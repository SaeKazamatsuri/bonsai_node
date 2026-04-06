from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .bonsai_manager import BonsaiServerManager

if TYPE_CHECKING:
    import numpy as np


def _normalize_tags(text: str) -> str:
    normalized = text.replace("\r", " ").replace("\n", " ").strip()
    return ",".join(part.strip() for part in normalized.split(",") if part.strip())


def _split_tags(text: str) -> list[str]:
    normalized = _normalize_tags(text)
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _validate_instruction(instruction_ja: str) -> str:
    stripped = instruction_ja.strip()
    if not stripped:
        raise ValueError("instruction_ja を入力してください。")
    return stripped


def _require_numpy() -> object:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "numpy が必要です。`pip install numpy sentence-transformers torch` を実行してください。"
        ) from exc
    return np


def _require_sentence_transformer() -> object:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers が必要です。`pip install sentence-transformers torch numpy` を実行してください。"
        ) from exc
    return SentenceTransformer


@dataclass(frozen=True)
class TagMetadata:
    name: str
    words: list[str]
    category: int
    post_count: int
    is_deprecated: bool


@dataclass(frozen=True)
class TagSearchResult:
    metadata: TagMetadata
    similarity: float
    score: float


@dataclass(frozen=True)
class TagIndex:
    model_name: str
    index_version: int
    source_path: str
    source_mtime_ns: int
    source_size: int
    metadata: list[TagMetadata]
    vectors: "np.ndarray"

    @property
    def tag_set(self) -> set[str]:
        return {item.name for item in self.metadata}


class TagEmbeddingCatalog:
    _instance: "TagEmbeddingCatalog | None" = None
    _lock = threading.Lock()

    INDEX_VERSION = 1
    MODEL_NAME = "intfloat/multilingual-e5-small"
    INDEX_META_NAME = "tag_index_meta.json"
    INDEX_VECTOR_NAME = "tag_index_vectors.npz"
    ENCODE_BATCH_SIZE = 256
    SEARCH_MULTIPLIER = 6
    MIN_INTERNAL_CANDIDATES = 256

    CATEGORY_LABELS = {
        0: "general",
        1: "artist",
        3: "copyright",
        4: "character",
        5: "meta",
    }
    CATEGORY_PROFILES: dict[str, dict[int, float]] = {
        "balanced": {
            0: 1.12,
            1: 0.95,
            3: 1.00,
            4: 1.12,
            5: 0.72,
        },
        "character_focus": {
            0: 1.00,
            1: 0.90,
            3: 1.05,
            4: 1.28,
            5: 0.65,
        },
        "style_pose_focus": {
            0: 1.20,
            1: 0.92,
            3: 0.88,
            4: 0.92,
            5: 0.60,
        },
    }

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._source_path = base_dir / "tags.json"
        self._meta_path = base_dir / self.INDEX_META_NAME
        self._vector_path = base_dir / self.INDEX_VECTOR_NAME
        self._index: TagIndex | None = None
        self._model: object | None = None
        self._instance_lock = threading.RLock()

    @classmethod
    def instance(cls) -> "TagEmbeddingCatalog":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(Path(__file__).resolve().parent)
            return cls._instance

    @classmethod
    def category_profile_names(cls) -> tuple[str, ...]:
        return tuple(cls.CATEGORY_PROFILES.keys())

    def filter_existing_tags(self, tags: list[str]) -> list[str]:
        index = self.load_or_build_index(rebuild=False)
        filtered: list[str] = []
        seen: set[str] = set()
        tag_set = index.tag_set
        for tag in tags:
            if tag not in tag_set or tag in seen:
                continue
            filtered.append(tag)
            seen.add(tag)
        return filtered

    def search(self, instruction_ja: str, limit: int, category_profile: str, rebuild: bool) -> list[TagSearchResult]:
        if limit < 1:
            raise ValueError("max_candidates は 1 以上にしてください。")
        if category_profile not in self.CATEGORY_PROFILES:
            raise ValueError(f"未対応の category_profile です: {category_profile}")

        index = self.load_or_build_index(rebuild=rebuild)
        query_vector = self._encode_query(instruction_ja)
        candidates = self._collect_candidates(index=index, query_vector=query_vector, limit=limit)
        if not candidates:
            raise RuntimeError("instruction_ja に一致する tags.json の候補が見つかりません。")

        profile = self.CATEGORY_PROFILES[category_profile]
        rescored: list[TagSearchResult] = []
        for similarity, metadata in candidates:
            score = self._score_candidate(metadata=metadata, similarity=similarity, profile=profile)
            rescored.append(TagSearchResult(metadata=metadata, similarity=similarity, score=score))

        rescored.sort(
            key=lambda item: (
                -item.score,
                -item.similarity,
                -item.metadata.post_count,
                item.metadata.name,
            )
        )
        return rescored[:limit]

    def load_or_build_index(self, rebuild: bool) -> TagIndex:
        with self._instance_lock:
            if rebuild:
                self._index = self._build_index()
                return self._index

            if self._index is not None and self._is_index_fresh(self._index):
                return self._index

            disk_index = self._load_index_from_disk()
            if disk_index is not None and self._is_index_fresh(disk_index):
                self._index = disk_index
                return disk_index

            self._index = self._build_index()
            return self._index

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model

        sentence_transformer = _require_sentence_transformer()
        self._model = sentence_transformer(self.MODEL_NAME)
        return self._model

    def _encode_query(self, instruction_ja: str) -> "np.ndarray":
        np_module = _require_numpy()
        model = self._load_model()
        encoded = model.encode(
            [f"query: {instruction_ja.strip()}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np_module.asarray(encoded[0], dtype=np_module.float32)

    def _collect_candidates(
        self,
        index: TagIndex,
        query_vector: "np.ndarray",
        limit: int,
    ) -> list[tuple[float, TagMetadata]]:
        np_module = _require_numpy()
        similarities = index.vectors @ query_vector
        internal_limit = min(
            len(index.metadata),
            max(limit * self.SEARCH_MULTIPLIER, self.MIN_INTERNAL_CANDIDATES),
        )
        top_indices = np_module.argpartition(similarities, -internal_limit)[-internal_limit:]
        sorted_indices = top_indices[np_module.argsort(similarities[top_indices])[::-1]]

        results: list[tuple[float, TagMetadata]] = []
        for raw_index in sorted_indices.tolist():
            similarity = float(similarities[raw_index])
            metadata = index.metadata[int(raw_index)]
            results.append((similarity, metadata))
        return results

    def _score_candidate(self, metadata: TagMetadata, similarity: float, profile: dict[int, float]) -> float:
        category_weight = profile.get(metadata.category, 0.82)
        deprecated_penalty = -0.25 if metadata.is_deprecated else 0.0
        post_count_boost = min(0.18, math.log10(max(metadata.post_count, 1) + 1.0) * 0.03)
        exact_word_bonus = 0.03 if metadata.name in metadata.words else 0.0
        return (similarity * category_weight) + post_count_boost + exact_word_bonus + deprecated_penalty

    def _build_index(self) -> TagIndex:
        source_signature = self._source_signature()
        metadata = self._load_metadata_from_source()
        vectors = self._encode_passages(metadata)
        index = TagIndex(
            model_name=self.MODEL_NAME,
            index_version=self.INDEX_VERSION,
            source_path=str(self._source_path),
            source_mtime_ns=source_signature["source_mtime_ns"],
            source_size=source_signature["source_size"],
            metadata=metadata,
            vectors=vectors,
        )
        self._save_index(index)
        return index

    def _encode_passages(self, metadata: list[TagMetadata]) -> "np.ndarray":
        np_module = _require_numpy()
        model = self._load_model()
        passages = [self._build_passage_text(item) for item in metadata]
        vectors = model.encode(
            passages,
            batch_size=self.ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np_module.asarray(vectors, dtype=np_module.float32)

    def _save_index(self, index: TagIndex) -> None:
        np_module = _require_numpy()
        meta_payload = {
            "model_name": index.model_name,
            "index_version": index.index_version,
            "source_path": index.source_path,
            "source_mtime_ns": index.source_mtime_ns,
            "source_size": index.source_size,
            "metadata": [asdict(item) for item in index.metadata],
        }
        with self._meta_path.open("w", encoding="utf-8") as file:
            json.dump(meta_payload, file, ensure_ascii=False)
        np_module.savez_compressed(self._vector_path, vectors=index.vectors)

    def _load_index_from_disk(self) -> TagIndex | None:
        if not self._meta_path.is_file() or not self._vector_path.is_file():
            return None

        np_module = _require_numpy()
        with self._meta_path.open("r", encoding="utf-8") as file:
            raw_meta = json.load(file)

        if not isinstance(raw_meta, dict):
            return None

        metadata_raw = raw_meta.get("metadata")
        if not isinstance(metadata_raw, list):
            return None

        metadata: list[TagMetadata] = []
        for item in metadata_raw:
            if not isinstance(item, dict):
                return None
            name = item.get("name")
            words = item.get("words")
            category = item.get("category")
            post_count = item.get("post_count")
            is_deprecated = item.get("is_deprecated")
            if (
                not isinstance(name, str)
                or not isinstance(words, list)
                or not all(isinstance(word, str) for word in words)
                or not isinstance(category, int)
                or not isinstance(post_count, int)
                or not isinstance(is_deprecated, bool)
            ):
                return None
            metadata.append(
                TagMetadata(
                    name=name,
                    words=list(words),
                    category=category,
                    post_count=post_count,
                    is_deprecated=is_deprecated,
                )
            )

        with np_module.load(self._vector_path) as loaded:
            vectors = np_module.asarray(loaded["vectors"], dtype=np_module.float32)

        if len(metadata) != int(vectors.shape[0]):
            return None

        return TagIndex(
            model_name=str(raw_meta.get("model_name", "")),
            index_version=int(raw_meta.get("index_version", 0)),
            source_path=str(raw_meta.get("source_path", "")),
            source_mtime_ns=int(raw_meta.get("source_mtime_ns", 0)),
            source_size=int(raw_meta.get("source_size", 0)),
            metadata=metadata,
            vectors=vectors,
        )

    def _is_index_fresh(self, index: TagIndex) -> bool:
        source_signature = self._source_signature()
        return (
            index.index_version == self.INDEX_VERSION
            and index.model_name == self.MODEL_NAME
            and index.source_path == str(self._source_path)
            and index.source_mtime_ns == source_signature["source_mtime_ns"]
            and index.source_size == source_signature["source_size"]
            and len(index.metadata) > 0
        )

    def _source_signature(self) -> dict[str, int]:
        if not self._source_path.is_file():
            raise FileNotFoundError(f"tags.json が見つかりません: {self._source_path}")
        stat = self._source_path.stat()
        return {
            "source_mtime_ns": stat.st_mtime_ns,
            "source_size": stat.st_size,
        }

    def _load_metadata_from_source(self) -> list[TagMetadata]:
        with self._source_path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

        if not isinstance(raw, list):
            raise RuntimeError("tags.json の内容が不正です。")

        metadata: list[TagMetadata] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            words = item.get("words")
            category = item.get("category")
            post_count = item.get("post_count")
            is_deprecated = item.get("is_deprecated")
            if (
                not isinstance(name, str)
                or not isinstance(words, list)
                or not all(isinstance(word, str) for word in words)
                or not isinstance(category, int)
                or not isinstance(post_count, int)
                or not isinstance(is_deprecated, bool)
            ):
                continue
            metadata.append(
                TagMetadata(
                    name=name,
                    words=list(words),
                    category=category,
                    post_count=post_count,
                    is_deprecated=is_deprecated,
                )
            )

        if not metadata:
            raise RuntimeError("tags.json からタグを読み込めませんでした。")
        return metadata

    def _build_passage_text(self, metadata: TagMetadata) -> str:
        category_label = self.CATEGORY_LABELS.get(metadata.category, "other")
        words_text = " ".join(metadata.words)
        return (
            f"passage: tag {metadata.name} "
            f"words {words_text} "
            f"category {category_label}"
        ).strip()


class BonsaiChatNode:
    CATEGORY = "LLM/Bonsai"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)

    DEFAULT_SYSTEM_PROMPT = (
        "あなたは画像生成向けタグ生成アシスタントです。"
        "入力された日本語の指示を読み取り、内容に合う短いタグを英語中心で生成してください。"
        "出力は1行のカンマ区切りタグのみとし、説明文、番号、改行、前置きは禁止です。"
        "人物、構図、背景、色を含めてください。"
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str, dict[str, object]]]]:
        return {
            "required": {
                "instruction_ja": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": cls.DEFAULT_SYSTEM_PROMPT},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "max_tokens": ("INT", {"default": 128, "min": 1, "max": 2048}),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": ("INT", {"default": 20, "min": 1, "max": 200}),
            }
        }

    def run(
        self,
        instruction_ja: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
    ) -> tuple[str]:
        manager = BonsaiServerManager.instance()
        text = manager.chat(
            system_prompt=system_prompt,
            user_prompt=self._build_user_prompt(instruction_ja),
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
        )
        return (_normalize_tags(text),)

    @staticmethod
    def _build_user_prompt(instruction_ja: str) -> str:
        stripped = _validate_instruction(instruction_ja)
        return (
            "次の日本語指示を、画像生成向けのカンマ区切りタグへ変換してください。\n"
            f"指示: {stripped}"
        )


class BonsaiCsvTagSelectorNode:
    CATEGORY = "LLM/Bonsai"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)

    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_MAX_TOKENS = 128
    DEFAULT_TOP_P = 0.9
    DEFAULT_TOP_K = 20
    CONTEXT_MARGIN_TOKENS = 256
    DEFAULT_SYSTEM_PROMPT = (
        "あなたは tags.json の候補から画像生成タグを選ぶアシスタントです。"
        "必ず候補一覧に含まれるタグだけを選んでください。"
        "候補にないタグを新規生成してはいけません。"
        "出力は1行のカンマ区切りタグのみとし、説明文、番号、改行、前置きは禁止です。"
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[object, dict[str, object]]]]:
        return {
            "required": {
                "instruction_ja": ("STRING", {"multiline": True, "default": ""}),
                "max_candidates": ("INT", {"default": 200, "min": 1, "max": 2000}),
                "max_selected_tags": ("INT", {"default": 32, "min": 1, "max": 256}),
                "category_profile": (
                    TagEmbeddingCatalog.category_profile_names(),
                    {"default": "balanced"},
                ),
                "rebuild_index": ("BOOLEAN", {"default": False}),
            }
        }

    def run(
        self,
        instruction_ja: str,
        max_candidates: int,
        max_selected_tags: int,
        category_profile: str,
        rebuild_index: bool,
    ) -> tuple[str]:
        stripped_instruction = _validate_instruction(instruction_ja)
        if max_selected_tags < 1:
            raise ValueError("max_selected_tags は 1 以上にしてください。")

        catalog = TagEmbeddingCatalog.instance()
        candidates = catalog.search(
            instruction_ja=stripped_instruction,
            limit=max_candidates,
            category_profile=category_profile,
            rebuild=bool(rebuild_index),
        )
        candidate_tags = [item.metadata.name for item in candidates]

        manager = BonsaiServerManager.instance()
        user_prompt = self._build_fitted_user_prompt(
            manager=manager,
            instruction_ja=stripped_instruction,
            candidates=candidates,
            max_selected_tags=max_selected_tags,
            category_profile=category_profile,
        )
        text = manager.chat(
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=self.DEFAULT_TEMPERATURE,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            top_p=self.DEFAULT_TOP_P,
            top_k=self.DEFAULT_TOP_K,
        )

        normalized_tags = self._normalize_selected_tags(
            text=text,
            candidate_tags=candidate_tags,
            max_selected_tags=max_selected_tags,
            catalog=catalog,
        )
        if not normalized_tags:
            raise RuntimeError("候補内のタグを選択できませんでした。")
        return (",".join(normalized_tags),)

    @staticmethod
    def _build_candidate_line(item: TagSearchResult) -> str:
        return (
            f"- {item.metadata.name} | "
            f"category={TagEmbeddingCatalog.CATEGORY_LABELS.get(item.metadata.category, 'other')} | "
            f"post_count={item.metadata.post_count}"
        )

    @classmethod
    def _build_fitted_user_prompt(
        cls,
        manager: BonsaiServerManager,
        instruction_ja: str,
        candidates: list[TagSearchResult],
        max_selected_tags: int,
        category_profile: str,
    ) -> str:
        ctx_size = manager.get_context_size()
        system_tokens = manager.estimate_token_count(cls.DEFAULT_SYSTEM_PROMPT)
        fixed_tokens = system_tokens + cls.DEFAULT_MAX_TOKENS + cls.CONTEXT_MARGIN_TOKENS

        prompt_header = (
            "次の日本語指示に合うタグを候補一覧から選んでください。\n"
            f"指示: {instruction_ja}\n"
            f"category_profile: {category_profile}\n"
            f"最大選択数: {max_selected_tags}\n"
            "ルール:\n"
            "- 候補一覧にあるタグだけを使う\n"
            "- 指示に本当に必要なタグだけを選ぶ\n"
            "- 出力は1行のカンマ区切りタグのみ\n"
            "- category や post_count は参考情報であり、出力には含めない\n"
            "候補一覧:\n"
        )
        prompt_tokens = manager.estimate_token_count(prompt_header)
        available_candidate_tokens = ctx_size - fixed_tokens - prompt_tokens
        if available_candidate_tokens <= 0:
            raise RuntimeError(
                "Bonsai のコンテキスト長が不足しています。config.json の ctx_size を増やすか、入力文を短くしてください。"
            )

        selected_lines: list[str] = []
        used_candidate_tokens = 0
        for item in candidates:
            line = cls._build_candidate_line(item)
            line_tokens = manager.estimate_token_count(f"{line}\n")
            if selected_lines and (used_candidate_tokens + line_tokens) > available_candidate_tokens:
                break
            if not selected_lines and line_tokens > available_candidate_tokens:
                raise RuntimeError(
                    "候補一覧を 1 件も収められません。config.json の ctx_size を増やしてください。"
                )
            selected_lines.append(line)
            used_candidate_tokens += line_tokens

        if not selected_lines:
            raise RuntimeError("候補一覧を構築できませんでした。")

        return prompt_header + "\n".join(selected_lines)

    @staticmethod
    def _build_user_prompt(
        instruction_ja: str,
        candidates: list[TagSearchResult],
        max_selected_tags: int,
        category_profile: str,
    ) -> str:
        candidate_lines = "\n".join(BonsaiCsvTagSelectorNode._build_candidate_line(item) for item in candidates)
        return (
            "次の日本語指示に合うタグを候補一覧から選んでください。\n"
            f"指示: {instruction_ja}\n"
            f"category_profile: {category_profile}\n"
            f"最大選択数: {max_selected_tags}\n"
            "ルール:\n"
            "- 候補一覧にあるタグだけを使う\n"
            "- 指示に本当に必要なタグだけを選ぶ\n"
            "- 出力は1行のカンマ区切りタグのみ\n"
            "- category や post_count は参考情報であり、出力には含めない\n"
            "候補一覧:\n"
            f"{candidate_lines}"
        )

    @staticmethod
    def _normalize_selected_tags(
        text: str,
        candidate_tags: list[str],
        max_selected_tags: int,
        catalog: TagEmbeddingCatalog,
    ) -> list[str]:
        normalized_text = _normalize_tags(text)
        raw_tags = [part.strip() for part in normalized_text.split(",") if part.strip()]
        valid_catalog_tags = catalog.filter_existing_tags(raw_tags)
        candidate_tag_set = set(candidate_tags)
        selected_tag_set = {tag for tag in valid_catalog_tags if tag in candidate_tag_set}
        ordered_selected_tags = [tag for tag in candidate_tags if tag in selected_tag_set]
        return ordered_selected_tags[:max_selected_tags]
