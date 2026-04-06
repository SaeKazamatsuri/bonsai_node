from __future__ import annotations

import csv
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from .bonsai_manager import BonsaiServerManager


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


@dataclass(frozen=True)
class TagEntry:
    tag: str
    count: int


class TagCatalog:
    _instance: "TagCatalog | None" = None
    _lock = threading.Lock()

    def __init__(self, csv_path: Path) -> None:
        self._csv_path = csv_path
        self._entries = self._load_entries(csv_path)
        self._tag_set = {entry.tag for entry in self._entries}

    @classmethod
    def instance(cls) -> "TagCatalog":
        with cls._lock:
            if cls._instance is None:
                csv_path = Path(__file__).resolve().parent / "tags.csv"
                cls._instance = cls(csv_path)
            return cls._instance

    @property
    def tag_set(self) -> set[str]:
        return self._tag_set

    def find_candidates(self, instruction_ja: str, limit: int) -> list[TagEntry]:
        if limit < 1:
            raise ValueError("max_candidates は 1 以上にしてください。")

        return self.find_candidates_for_queries([instruction_ja], limit)

    def find_candidates_for_queries(self, queries: list[str], limit: int) -> list[TagEntry]:
        if limit < 1:
            raise ValueError("max_candidates は 1 以上にしてください。")

        prepared_queries = self._prepare_queries(queries)
        scored_entries: list[tuple[int, int, str, TagEntry]] = []

        for entry in self._entries:
            score = self._score_entry(entry, prepared_queries)
            if score <= 0:
                continue
            scored_entries.append((score, entry.count, entry.tag, entry))

        if not scored_entries:
            raise ValueError("instruction_ja に一致する tags.csv の候補が見つかりません。")

        scored_entries.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return [item[3] for item in scored_entries[:limit]]

    def filter_existing_tags(self, tags: list[str]) -> list[str]:
        filtered: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            if tag not in self._tag_set or tag in seen:
                continue
            filtered.append(tag)
            seen.add(tag)
        return filtered

    @staticmethod
    def _load_entries(csv_path: Path) -> list[TagEntry]:
        if not csv_path.is_file():
            raise FileNotFoundError(f"tags.csv が見つかりません: {csv_path}")

        entries: list[TagEntry] = []
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) < 2:
                    continue
                tag = row[0].strip()
                count_text = row[1].strip()
                if not tag:
                    continue
                try:
                    count = int(count_text)
                except ValueError as exc:
                    raise RuntimeError(f"tags.csv の頻度が整数ではありません: {tag}") from exc
                entries.append(TagEntry(tag=tag, count=count))
        if not entries:
            raise RuntimeError("tags.csv からタグを読み込めませんでした。")
        return entries

    @staticmethod
    def _normalize_search_text(text: str) -> str:
        lowered = text.strip().lower()
        return re.sub(r"\s+", " ", lowered)

    @staticmethod
    def _normalize_tag(tag: str) -> str:
        lowered = tag.strip().lower()
        replaced = lowered.replace("_", " ")
        return re.sub(r"\s+", " ", replaced)

    def _build_search_tokens(self, text: str) -> list[str]:
        token_candidates = re.findall(r"[0-9a-zA-Z_]+|[ぁ-んァ-ヶー一-龠]{2,}", text)
        normalized_tokens: list[str] = []
        seen_tokens: set[str] = set()
        for token in token_candidates:
            normalized_token = self._normalize_tag(token)
            if len(normalized_token) < 2 or normalized_token in seen_tokens:
                continue
            normalized_tokens.append(normalized_token)
            seen_tokens.add(normalized_token)
        return normalized_tokens

    def _prepare_queries(self, queries: list[str]) -> list[tuple[str, list[str]]]:
        prepared: list[tuple[str, list[str]]] = []
        seen_queries: set[str] = set()
        for query in queries:
            normalized_query = self._normalize_search_text(query)
            if not normalized_query or normalized_query in seen_queries:
                continue
            prepared.append((normalized_query, self._build_search_tokens(normalized_query)))
            seen_queries.add(normalized_query)
        return prepared

    def _score_entry(self, entry: TagEntry, queries: list[tuple[str, list[str]]]) -> int:
        normalized_tag = self._normalize_tag(entry.tag)
        if not normalized_tag:
            return 0

        score = 0
        for normalized_query, tokens in queries:
            if normalized_query and normalized_query in normalized_tag:
                score += 100 + len(normalized_query)
            for token in tokens:
                if token == normalized_tag:
                    score += 80 + len(token)
                    continue
                if token in normalized_tag:
                    score += 20 + len(token)
        return score


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
    DEFAULT_CREATIVITY = 0.3
    DEFAULT_SYSTEM_PROMPT = (
        "あなたは tags.csv 候補から画像生成タグを選ぶアシスタントです。"
        "必ず候補一覧に含まれるタグだけを選んでください。"
        "候補にないタグを新規生成してはいけません。"
        "出力は1行のカンマ区切りタグのみとし、説明文、番号、改行、前置きは禁止です。"
    )
    SEARCH_SYSTEM_PROMPT = (
        "あなたは日本語の画像指示を、tags.csv 検索用の短い英語タグ候補へ変換するアシスタントです。"
        "出力は1行のカンマ区切りタグのみとし、説明文、番号、改行、前置きは禁止です。"
        "人物、構図、背景、色、雰囲気、作品ジャンルを含めてよいです。"
        "抽象的な指示でも、画像タグ検索に有効な具体語へ言い換えてください。"
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str, dict[str, object]]]]:
        return {
            "required": {
                "instruction_ja": ("STRING", {"multiline": True, "default": ""}),
                "max_candidates": ("INT", {"default": 200, "min": 1, "max": 2000}),
                "max_selected_tags": ("INT", {"default": 32, "min": 1, "max": 256}),
                "creativity": (
                    "FLOAT",
                    {"default": cls.DEFAULT_CREATIVITY, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    def run(
        self,
        instruction_ja: str,
        max_candidates: int,
        max_selected_tags: int,
        creativity: float,
    ) -> tuple[str]:
        stripped_instruction = _validate_instruction(instruction_ja)
        if max_selected_tags < 1:
            raise ValueError("max_selected_tags は 1 以上にしてください。")

        manager = BonsaiServerManager.instance()
        creativity_value = self._clamp_creativity(creativity)
        search_queries = self._build_search_queries(
            manager=manager,
            instruction_ja=stripped_instruction,
            creativity=creativity_value,
        )

        catalog = TagCatalog.instance()
        candidates = catalog.find_candidates_for_queries(search_queries, max_candidates)
        candidate_tags = [entry.tag for entry in candidates]

        text = manager.chat(
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            user_prompt=self._build_user_prompt(
                instruction_ja=stripped_instruction,
                candidate_tags=candidate_tags,
                max_selected_tags=max_selected_tags,
                creativity=creativity_value,
            ),
            temperature=self._selection_temperature(creativity_value),
            max_tokens=self.DEFAULT_MAX_TOKENS,
            top_p=self._selection_top_p(creativity_value),
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
    def _build_user_prompt(
        instruction_ja: str,
        candidate_tags: list[str],
        max_selected_tags: int,
        creativity: float,
    ) -> str:
        candidate_lines = "\n".join(f"- {tag}" for tag in candidate_tags)
        creativity_instruction = (
            "指示にかなり忠実に選んでください。連想は最小限にしてください。"
            if creativity < 0.34
            else "指示に沿いつつ、画像として自然な補完を少しだけ行ってください。"
            if creativity < 0.67
            else "指示の意図を広めに解釈し、雰囲気や構図の補完も積極的に行ってください。"
        )
        return (
            "次の日本語指示に合うタグを候補一覧から選んでください。\n"
            f"指示: {instruction_ja}\n"
            f"独創性: {creativity:.2f}\n"
            f"最大選択数: {max_selected_tags}\n"
            "ルール:\n"
            "- 候補一覧にあるタグだけを使う\n"
            "- 必要なタグだけを選ぶ\n"
            "- 出力は1行のカンマ区切りタグのみ\n"
            f"- {creativity_instruction}\n"
            "候補一覧:\n"
            f"{candidate_lines}"
        )

    @staticmethod
    def _normalize_selected_tags(
        text: str,
        candidate_tags: list[str],
        max_selected_tags: int,
        catalog: TagCatalog,
    ) -> list[str]:
        normalized_text = _normalize_tags(text)
        raw_tags = [part.strip() for part in normalized_text.split(",") if part.strip()]
        valid_catalog_tags = catalog.filter_existing_tags(raw_tags)
        candidate_tag_set = set(candidate_tags)
        selected_tag_set = {tag for tag in valid_catalog_tags if tag in candidate_tag_set}
        ordered_selected_tags = [tag for tag in candidate_tags if tag in selected_tag_set]
        return ordered_selected_tags[:max_selected_tags]

    def _build_search_queries(
        self,
        manager: BonsaiServerManager,
        instruction_ja: str,
        creativity: float,
    ) -> list[str]:
        search_hint_text = manager.chat(
            system_prompt=self.SEARCH_SYSTEM_PROMPT,
            user_prompt=self._build_search_prompt(instruction_ja, creativity),
            temperature=self._search_temperature(creativity),
            max_tokens=self.DEFAULT_MAX_TOKENS,
            top_p=self._search_top_p(creativity),
            top_k=self.DEFAULT_TOP_K,
        )
        search_hints = _split_tags(search_hint_text)
        return [instruction_ja, *search_hints]

    @staticmethod
    def _build_search_prompt(instruction_ja: str, creativity: float) -> str:
        creativity_instruction = (
            "元の指示に近い直接的な検索語を優先してください。"
            if creativity < 0.34
            else "元の指示に加えて、近い言い換えや補助的な検索語も出してください。"
            if creativity < 0.67
            else "抽象語を具体的な視覚要素へ展開し、雰囲気や演出の検索語も出してください。"
        )
        return (
            "次の日本語指示から、tags.csv を探すための短い英語タグ候補を作成してください。\n"
            f"指示: {instruction_ja}\n"
            f"独創性: {creativity:.2f}\n"
            "ルール:\n"
            "- 1行のカンマ区切りのみ\n"
            "- 英語中心の短いタグ候補\n"
            "- 同義語や言い換えを含めてよい\n"
            "- 画像タグ検索に不向きな長文は禁止\n"
            f"- {creativity_instruction}"
        )

    @staticmethod
    def _clamp_creativity(creativity: float) -> float:
        return max(0.0, min(1.0, creativity))

    @classmethod
    def _search_temperature(cls, creativity: float) -> float:
        return 0.15 + (0.55 * creativity)

    @classmethod
    def _search_top_p(cls, creativity: float) -> float:
        return min(0.98, 0.75 + (0.2 * creativity))

    @classmethod
    def _selection_temperature(cls, creativity: float) -> float:
        return cls.DEFAULT_TEMPERATURE + (0.35 * creativity)

    @classmethod
    def _selection_top_p(cls, creativity: float) -> float:
        return min(0.98, cls.DEFAULT_TOP_P + (0.08 * creativity))
