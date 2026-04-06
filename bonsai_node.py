from __future__ import annotations

import json
import math
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .bonsai_manager import BonsaiContextSizeError, BonsaiServerManager

if TYPE_CHECKING:
    import numpy as np


def _normalize_tags(text: str) -> str:
    normalized = text.replace("\r", " ").replace("\n", " ").strip()
    normalized = re.sub(r"(?i)^prompt\s*:\s*", "", normalized)
    return ",".join(part.strip() for part in normalized.split(",") if part.strip())


def _split_tags(text: str) -> list[str]:
    normalized = _normalize_tags(text)
    return [part.strip() for part in normalized.split(",") if part.strip()]


def _display_tag(tag: str) -> str:
    return tag.replace("_", " ")


COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "black": ("黒", "黒い", "ブラック", "black"),
    "blue": ("青", "青い", "ブルー", "blue"),
    "white": ("白", "白い", "ホワイト", "white"),
    "red": ("赤", "赤い", "レッド", "red"),
    "green": ("緑", "緑色", "グリーン", "green"),
    "yellow": ("黄", "黄色", "イエロー", "yellow"),
    "pink": ("ピンク", "pink"),
    "purple": ("紫", "紫色", "パープル", "purple"),
    "grey": ("灰", "灰色", "グレー", "grey", "gray"),
    "brown": ("茶", "茶色", "ブラウン", "brown"),
}
KNOWN_COLOR_PREFIXES: set[str] = set(COLOR_ALIASES.keys())
CLOTHING_TERMS: tuple[str, ...] = (
    "dress",
    "shirt",
    "skirt",
    "jacket",
    "coat",
    "hoodie",
    "sweater",
    "vest",
    "kimono",
    "serafuku",
    "suit",
    "underwear",
    "underwear_only",
    "bra",
    "sports_bra",
    "panties",
    "shoes",
    "sleeves",
    "collar",
    "ribbon",
    "bow",
    "necktie",
    "neckerchief",
    "socks",
    "thighhighs",
    "pantyhose",
    "gloves",
    "cardigan",
    "cloak",
    "cape",
    "boots",
    "sandals",
    "pants",
    "shorts",
    "buruma",
    "leotard",
    "bodysuit",
    "bikini",
    "one-piece_swimsuit",
)
SORTED_CLOTHING_TERMS: tuple[str, ...] = tuple(sorted(CLOTHING_TERMS, key=len, reverse=True))
CANONICAL_CLOTHING_FAMILIES: dict[str, str] = {
    "underwear_only": "underwear",
    "sports_bra": "bra",
    "bikini": "swimwear",
    "one-piece_swimsuit": "swimwear",
}
INSTRUCTION_CONCEPT_ALIASES: dict[str, tuple[str, ...]] = {
    "underwear": ("下着", "下着姿", "ランジェリー", "lingerie", "underwear"),
    "underwear_only": ("下着のみ", "下着だけ", "underwear only"),
    "bra": ("ブラ", "ブラジャー", "bra"),
    "panties": ("パンツ", "パンティ", "パンティー", "ショーツ", "panties"),
    "dress": ("ドレス", "ワンピース", "dress"),
    "shirt": ("シャツ", "shirt"),
    "skirt": ("スカート", "skirt"),
    "swimwear": ("水着", "ビキニ", "swimsuit", "bikini"),
    "thighhighs": ("ニーハイ", "thighhighs"),
    "pantyhose": ("パンスト", "ストッキング", "パンティストッキング", "pantyhose"),
    "full_body": ("全身", "全身像", "full body"),
    "upper_body": ("上半身", "バストアップ", "upper body"),
    "sitting": ("座り", "座って", "sitting"),
    "standing": ("立ち", "立って", "standing"),
    "flat_chest": ("貧乳", "ぺたんこ", "平らな胸", "flat chest"),
    "small_breasts": ("小さな胸", "small breasts"),
    "medium_breasts": ("普通の胸", "medium breasts"),
    "large_breasts": ("巨乳", "大きな胸", "large breasts", "big breasts"),
    "huge_breasts": ("爆乳", "huge breasts"),
    "gigantic_breasts": ("超爆乳", "gigantic breasts"),
}
UNDERWEAR_THEME_ALLOWED_FAMILIES: frozenset[str] = frozenset({"underwear", "bra", "panties"})
STRICT_TAG_DENYLIST: frozenset[str] = frozenset(
    {
        "quality",
        "masterpiece",
        "best_quality",
        "high_quality",
        "highres",
        "absurdres",
        "worst_quality",
        "lowres",
        "aesthetic",
        "official_style",
        "source_anime",
        "source_cartoon",
        "source_furry",
        "source_pony",
        "source_comic",
    }
)
STRICT_NOISE_EXACT_TAGS: frozenset[str] = frozenset(
    {
        "tag",
        "content_rating",
        "button_prompt",
        "name_tag",
        "price_tag",
        "pixiv_id",
        "pixiv_username",
        "instagram_logo",
        "instagram_username",
        "timestamp",
        "watermark",
        "signature",
    }
)
STRICT_NOISE_SUBSTRINGS: tuple[str, ...] = (
    "_username",
    "_id",
    "watermark",
    "signature",
    "logo",
    "prompt",
    "content_rating",
)
EXPLICIT_ONLY_CATEGORIES: frozenset[int] = frozenset({1, 5})
BREAST_SIZE_TAGS: frozenset[str] = frozenset(
    {
        "flat_chest",
        "small_breasts",
        "medium_breasts",
        "large_breasts",
        "huge_breasts",
        "gigantic_breasts",
    }
)
FACE_EXACT_TAGS: frozenset[str] = frozenset(
    {
        "blush",
        "smile",
        "open_mouth",
        "closed_mouth",
        "parted_lips",
        "looking_at_viewer",
        "teeth",
        "tongue",
        "nose_blush",
    }
)
HAIR_HEAD_EXACT_TAGS: frozenset[str] = frozenset(
    {
        "bangs",
        "ahoge",
        "sidelocks",
        "twintails",
        "ponytail",
        "braid",
        "hair_ornament",
        "hair_ribbon",
        "hair_bow",
        "headband",
        "hat",
    }
)
ACCESSORY_EXACT_TAGS: frozenset[str] = frozenset(
    {
        "jewelry",
        "bracelet",
        "earrings",
        "necklace",
        "choker",
        "ring",
        "anklet",
    }
)
BODY_EXACT_TAGS: frozenset[str] = frozenset(
    {
        "breasts",
        "navel",
        "cleavage",
        "collarbone",
        "thighs",
        "stomach",
        "barefoot",
        "legs",
        "arms",
        "shoulders",
    }
)
POSE_EXACT_TAGS: frozenset[str] = frozenset(
    {
        "sitting",
        "standing",
        "kneeling",
        "wariza",
        "lying",
        "crouching",
        "leaning",
        "full_body",
        "upper_body",
        "cowboy_shot",
        "profile",
        "from_side",
        "from_behind",
        "clothes_lift",
        "shirt_lift",
        "lifted_by_self",
    }
)
BACKGROUND_EXACT_TAGS: frozenset[str] = frozenset(
    {
        "simple_background",
        "white_background",
        "black_background",
        "transparent_background",
        "outdoors",
        "indoors",
    }
)
PROMPT_BUCKET_ORDER: tuple[str, ...] = (
    "subject",
    "identity",
    "body",
    "face",
    "hair_head",
    "clothes",
    "pose",
    "background",
    "other",
)
PROMPT_BUCKET_LABELS: dict[str, str] = {
    "subject": "主題・人数",
    "identity": "版権・キャラ",
    "body": "身体・全体属性",
    "face": "顔・表情・目線",
    "hair_head": "髪・頭部",
    "clothes": "服・装飾",
    "pose": "ポーズ・構図・動作",
    "background": "背景・環境",
    "other": "その他",
}
OUTPUT_BUCKET_ORDER: tuple[str, ...] = (
    "subject",
    "identity",
    "body",
    "face",
    "hair_head",
    "clothes",
    "pose",
    "other",
    "background",
)
SUBJECT_COUNT_PATTERN = re.compile(r"^(?P<count>\d+)(girl|girls|boy|boys|other|others)$")
POSE_CONFLICT_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"sitting", "standing", "kneeling", "lying", "crouching", "leaning", "wariza"}),
    frozenset({"full_body", "upper_body", "cowboy_shot"}),
    frozenset({"profile", "from_side", "from_behind"}),
)
SDXL_DANBOORU_SYSTEM_PROMPT = (
    "あなたは、Stable Diffusion XL（SDXL）向けのプロンプトを生成する専門エージェントです。"
    "Danbooruタグ形式をベースに、構造化されたタグ列のみを出力してください。"
    "目的: Danbooruタグに準拠した、明確で再現性の高いプロンプトを生成し、不要な修飾（品質・レンダリング・雰囲気）を排除して純粋な構造情報のみを記述すること。"
    "出力形式は必ず `Prompt: <tag1>, <tag2>, <tag3>, ...` としてください。"
    "ネガティブプロンプトは禁止です。"
    "masterpiece, best_quality, high_quality などの品質タグは禁止です。"
    "cinematic_lighting, depth_of_field, volumetric_lighting などのレンダリング表現は禁止です。"
    "beautiful, cute, aesthetic などの抽象的・主観的表現は禁止です。"
    "自然言語の文章は禁止です。"
    "タグはすべて英語、lowercase、単語区切りはアンダースコア、カンマ区切りにしてください。"
    "重要度の高い順に並べ、1girl, 1boy などの主体タグは必ず先頭に配置してください。"
    "タグ構造は、1. 主体 2. キャラクター属性 3. 表情 4. 服装 5. ポーズ・動作 6. 構図 7. 背景・場所 の順に従ってください。"
    "ユーザーの入力が曖昧な場合は一般的なDanbooruタグに補完し、存在しないタグは作らず既存タグに近似し、冗長なタグは避けて簡潔にまとめてください。"
)


def _validate_instruction(instruction_ja: str) -> str:
    stripped = instruction_ja.strip()
    if not stripped:
        raise ValueError("instruction_ja を入力してください。")
    return stripped


def _extract_requested_colors(instruction_ja: str) -> set[str]:
    lowered = instruction_ja.casefold()
    requested: set[str] = set()
    for color, aliases in COLOR_ALIASES.items():
        if any(alias.casefold() in lowered for alias in aliases):
            requested.add(color)
    return requested


def _split_color_tag(tag: str) -> tuple[str | None, str]:
    parts = tag.split("_", 1)
    if len(parts) != 2:
        return (None, tag)
    prefix, remainder = parts
    if prefix not in KNOWN_COLOR_PREFIXES:
        return (None, tag)
    return (prefix, remainder)


def _canonical_clothing_family(term: str) -> str:
    return CANONICAL_CLOTHING_FAMILIES.get(term, term)


def _extract_clothing_family(tag: str) -> str | None:
    for term in SORTED_CLOTHING_TERMS:
        if tag == term or tag.endswith(f"_{term}"):
            return _canonical_clothing_family(term)
    return None


def _is_clothing_tag(tag: str) -> bool:
    return _extract_clothing_family(tag) is not None


def _instruction_mentions_aliases(instruction_ja: str, aliases: tuple[str, ...]) -> bool:
    lowered = instruction_ja.casefold()
    return any(alias.casefold() in lowered for alias in aliases)


def _instruction_mentions_concept(instruction_ja: str, concept: str) -> bool:
    aliases = INSTRUCTION_CONCEPT_ALIASES.get(concept)
    if aliases is None:
        return False
    return _instruction_mentions_aliases(instruction_ja, aliases)


def _instruction_requests_underwear_theme(instruction_ja: str) -> bool:
    return any(
        _instruction_mentions_concept(instruction_ja, concept)
        for concept in ("underwear", "underwear_only", "bra", "panties")
    )


def _is_specific_breast_detail_tag(tag: str) -> bool:
    if tag == "breasts" or tag in BREAST_SIZE_TAGS:
        return False
    return tag.endswith("_breasts")


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


def _normalized_tag_key(tag: str) -> str:
    return "_".join(part for part in tag.strip().replace("\r", " ").replace("\n", " ").split() if part).casefold()


def _instruction_mentions_text(instruction_ja: str, text: str) -> bool:
    if not text:
        return False
    return text.casefold() in instruction_ja.casefold()


def _instruction_explicitly_mentions_tag(instruction_ja: str, metadata: TagMetadata) -> bool:
    if _instruction_mentions_text(instruction_ja, metadata.name):
        return True
    if _instruction_mentions_text(instruction_ja, _display_tag(metadata.name)):
        return True
    if _instruction_mentions_concept(instruction_ja, metadata.name):
        return True
    clothing_family = _extract_clothing_family(metadata.name)
    if clothing_family is not None and _instruction_mentions_concept(instruction_ja, clothing_family):
        return True
    words_text = " ".join(metadata.words)
    return _instruction_mentions_text(instruction_ja, words_text)


def _is_denied_by_strict_policy(tag: str) -> bool:
    return tag in STRICT_TAG_DENYLIST


def _is_low_signal_tag(tag: str) -> bool:
    if tag in STRICT_NOISE_EXACT_TAGS:
        return True
    return any(pattern in tag for pattern in STRICT_NOISE_SUBSTRINGS)


def _is_subject_tag(tag: str) -> bool:
    return (
        tag == "solo"
        or tag.endswith("_focus")
        or tag.startswith("multiple_")
        or SUBJECT_COUNT_PATTERN.match(tag) is not None
        or (_person_count_value(tag) is not None)
    )


def _is_face_tag(tag: str) -> bool:
    if tag in FACE_EXACT_TAGS:
        return True
    return tag.endswith(("_eyes", "_mouth", "_lips", "_eyebrows"))


def _is_hair_head_tag(tag: str) -> bool:
    if tag in HAIR_HEAD_EXACT_TAGS:
        return True
    return tag.endswith(("_hair", "_bangs", "_braid", "_ponytail", "_twintails"))


def _is_accessory_tag(tag: str) -> bool:
    if tag in ACCESSORY_EXACT_TAGS:
        return True
    return tag.endswith(("_earrings", "_necklace", "_bracelet", "_choker", "_ring"))


def _is_body_tag(tag: str) -> bool:
    if tag in BODY_EXACT_TAGS or tag in BREAST_SIZE_TAGS:
        return True
    return tag.endswith(("_breasts", "_body"))


def _is_pose_tag(tag: str) -> bool:
    if tag in POSE_EXACT_TAGS:
        return True
    return tag.endswith(("_shot", "_view", "_angle", "_pose", "_lift"))


def _is_background_tag(tag: str) -> bool:
    if tag in BACKGROUND_EXACT_TAGS:
        return True
    return tag.endswith("_background")


def _classify_tag_bucket(tag: str, metadata: TagMetadata | None) -> str:
    if _is_subject_tag(tag):
        return "subject"
    if metadata is not None and metadata.category in {3, 4}:
        return "identity"
    if _is_body_tag(tag):
        return "body"
    if _is_face_tag(tag):
        return "face"
    if _is_hair_head_tag(tag):
        return "hair_head"
    if _is_clothing_tag(tag) or _is_accessory_tag(tag):
        return "clothes"
    if _is_pose_tag(tag):
        return "pose"
    if _is_background_tag(tag):
        return "background"
    return "other"


def _should_allow_metadata_by_default(instruction_ja: str, metadata: TagMetadata) -> bool:
    if _is_denied_by_strict_policy(metadata.name):
        return False
    if metadata.name in BREAST_SIZE_TAGS and not _instruction_explicitly_mentions_tag(instruction_ja, metadata):
        return False
    if _is_specific_breast_detail_tag(metadata.name) and not _instruction_explicitly_mentions_tag(instruction_ja, metadata):
        return False
    clothing_family = _extract_clothing_family(metadata.name)
    if (
        clothing_family is not None
        and _instruction_requests_underwear_theme(instruction_ja)
        and clothing_family not in UNDERWEAR_THEME_ALLOWED_FAMILIES
        and not _instruction_explicitly_mentions_tag(instruction_ja, metadata)
    ):
        return False
    if metadata.category in EXPLICIT_ONLY_CATEGORIES and not _instruction_explicitly_mentions_tag(instruction_ja, metadata):
        return False
    return True


def _person_count_value(tag: str) -> int | None:
    if tag == "solo":
        return 1
    match = SUBJECT_COUNT_PATTERN.match(tag)
    if match is not None:
        return int(match.group("count"))
    if tag.startswith("multiple_"):
        return 2
    if not any(token in tag for token in ("girl", "boy", "other")):
        return None
    numbers = [int(value) for value in re.findall(r"\d+", tag)]
    if not numbers:
        return None
    return sum(numbers)


def _filter_subject_conflicts(ordered_selected_tags: list[str]) -> list[str]:
    explicit_subject_count = sum(
        count
        for tag in ordered_selected_tags
        if tag != "solo" and (count := _person_count_value(tag)) is not None
    )
    inferred_count = explicit_subject_count
    if inferred_count == 0 and any(tag.startswith("multiple_") for tag in ordered_selected_tags):
        inferred_count = 2
    if inferred_count <= 1:
        return ordered_selected_tags
    return [tag for tag in ordered_selected_tags if tag != "solo"]


def _filter_breast_size_conflicts(ordered_selected_tags: list[str]) -> list[str]:
    kept_tags: list[str] = []
    seen_size = False
    for tag in ordered_selected_tags:
        if tag in BREAST_SIZE_TAGS:
            if seen_size:
                continue
            seen_size = True
        kept_tags.append(tag)
    return kept_tags


def _filter_background_conflicts(ordered_selected_tags: list[str]) -> list[str]:
    kept_tags: list[str] = []
    seen_specific_background = False
    for tag in ordered_selected_tags:
        color_prefix, base_tag = _split_color_tag(tag)
        if base_tag == "background" and color_prefix is not None:
            if seen_specific_background:
                continue
            seen_specific_background = True
        kept_tags.append(tag)
    return kept_tags


def _filter_pose_conflicts(ordered_selected_tags: list[str]) -> list[str]:
    kept_tags: list[str] = []
    seen_group_indexes: set[int] = set()
    for tag in ordered_selected_tags:
        matched_group_index: int | None = None
        for group_index, group in enumerate(POSE_CONFLICT_GROUPS):
            if tag in group:
                matched_group_index = group_index
                break
        if matched_group_index is not None:
            if matched_group_index in seen_group_indexes:
                continue
            seen_group_indexes.add(matched_group_index)
        kept_tags.append(tag)
    return kept_tags


def _filter_underwear_theme_conflicts(ordered_selected_tags: list[str], instruction_ja: str) -> list[str]:
    if not _instruction_requests_underwear_theme(instruction_ja):
        return ordered_selected_tags

    kept_tags: list[str] = []
    seen_underwear_families: set[str] = set()
    for tag in ordered_selected_tags:
        clothing_family = _extract_clothing_family(tag)
        if clothing_family is None:
            kept_tags.append(tag)
            continue
        if (
            clothing_family not in UNDERWEAR_THEME_ALLOWED_FAMILIES
            and not _instruction_mentions_concept(instruction_ja, clothing_family)
        ):
            continue
        if clothing_family in UNDERWEAR_THEME_ALLOWED_FAMILIES:
            if clothing_family in seen_underwear_families:
                continue
            seen_underwear_families.add(clothing_family)
        kept_tags.append(tag)
    return kept_tags


def _apply_output_bucket_order(ordered_selected_tags: list[str], metadata_by_tag: dict[str, TagMetadata]) -> list[str]:
    bucketed: dict[str, list[str]] = {bucket: [] for bucket in OUTPUT_BUCKET_ORDER}
    for tag in ordered_selected_tags:
        bucket = _classify_tag_bucket(tag, metadata_by_tag.get(tag))
        bucketed.setdefault(bucket, []).append(tag)

    ordered: list[str] = []
    for bucket in OUTPUT_BUCKET_ORDER:
        ordered.extend(bucketed.get(bucket, []))
    return ordered


def _build_candidate_lookup(candidate_tags: list[str]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for tag in candidate_tags:
        lookup[_normalized_tag_key(tag)] = tag
        lookup[_normalized_tag_key(_display_tag(tag))] = tag
    return lookup


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
        requested_colors = _extract_requested_colors(instruction_ja)
        query_vector = self._encode_query(instruction_ja)
        candidates = self._collect_candidates(index=index, query_vector=query_vector, limit=limit)
        if not candidates:
            raise RuntimeError("instruction_ja に一致する tags.json の候補が見つかりません。")

        profile = self.CATEGORY_PROFILES[category_profile]
        rescored: list[TagSearchResult] = []
        for similarity, metadata in candidates:
            score = self._score_candidate(
                instruction_ja=instruction_ja,
                metadata=metadata,
                similarity=similarity,
                profile=profile,
                requested_colors=requested_colors,
            )
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

    def _score_candidate(
        self,
        instruction_ja: str,
        metadata: TagMetadata,
        similarity: float,
        profile: dict[int, float],
        requested_colors: set[str],
    ) -> float:
        category_weight = profile.get(metadata.category, 0.82)
        deprecated_penalty = -0.25 if metadata.is_deprecated else 0.0
        post_count_boost = min(0.18, math.log10(max(metadata.post_count, 1) + 1.0) * 0.03)
        exact_word_bonus = 0.03 if metadata.name in metadata.words else 0.0
        strict_policy_penalty = 0.0
        if not _should_allow_metadata_by_default(instruction_ja, metadata):
            strict_policy_penalty -= 0.75
        if _is_low_signal_tag(metadata.name):
            strict_policy_penalty -= 0.28
        color_bonus = 0.0
        color_prefix, _ = _split_color_tag(metadata.name)
        if requested_colors and color_prefix is not None:
            if color_prefix in requested_colors:
                color_bonus = 0.14
            elif _is_clothing_tag(metadata.name):
                color_bonus = -0.18
            else:
                color_bonus = -0.06
        explicit_mention_bonus = 0.12 if _instruction_explicitly_mentions_tag(instruction_ja, metadata) else 0.0
        return (
            (similarity * category_weight)
            + post_count_boost
            + exact_word_bonus
            + deprecated_penalty
            + strict_policy_penalty
            + color_bonus
            + explicit_mention_bonus
        )

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


class BonsaiCsvTagSelectorNode:
    CATEGORY = "LLM/Bonsai"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)

    DEFAULT_MAX_TOKENS = 128
    CONTEXT_MARGIN_TOKENS = 256
    MIN_CANDIDATE_COUNT = 8
    DEFAULT_SYSTEM_PROMPT = (
        SDXL_DANBOORU_SYSTEM_PROMPT
        + " 必ず候補一覧に含まれるタグだけを選んでください。"
        + " 候補にないタグを新規生成してはいけません。"
        + " qualityタグ、画風タグ、artist/metaタグは、指示で明示されない限り選んではいけません。"
        + " 題材に直接関係するタグだけを選び、人物属性、顔髪、服、ポーズ、背景を必要な範囲で過不足なく構成してください。"
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
                "temperature": (
                    "FLOAT",
                    {"default": 0.55, "min": 0.0, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "top_k": ("INT", {"default": 40, "min": 1, "max": 200}),
                "rebuild_index": ("BOOLEAN", {"default": False}),
            }
        }

    def run(
        self,
        instruction_ja: str,
        max_candidates: int,
        max_selected_tags: int,
        category_profile: str,
        temperature: float,
        top_p: float,
        top_k: int,
        rebuild_index: bool,
    ) -> tuple[str]:
        tags = self._run_strict_selection(
            instruction_ja=instruction_ja,
            max_candidates=max_candidates,
            max_selected_tags=max_selected_tags,
            category_profile=category_profile,
            temperature=temperature,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            top_p=top_p,
            top_k=top_k,
            rebuild_index=rebuild_index,
            auxiliary_prompt=None,
        )
        return (tags,)

    @classmethod
    def _run_strict_selection(
        cls,
        instruction_ja: str,
        max_candidates: int,
        max_selected_tags: int,
        category_profile: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        rebuild_index: bool,
        auxiliary_prompt: str | None,
    ) -> str:
        stripped_instruction = _validate_instruction(instruction_ja)
        if max_selected_tags < 1:
            raise ValueError("max_selected_tags は 1 以上にしてください。")

        catalog = TagEmbeddingCatalog.instance()
        searched_candidates = catalog.search(
            instruction_ja=stripped_instruction,
            limit=max_candidates,
            category_profile=category_profile,
            rebuild=bool(rebuild_index),
        )
        candidates = cls._filter_candidates_for_prompt(
            instruction_ja=stripped_instruction,
            candidates=searched_candidates,
        )
        if not candidates:
            raise RuntimeError("strict policy 適用後に候補タグが残りませんでした。")

        candidate_tags = [item.metadata.name for item in candidates]
        metadata_by_tag = {item.metadata.name: item.metadata for item in candidates}

        manager = BonsaiServerManager.instance()
        text = cls._chat_with_retry(
            manager=manager,
            instruction_ja=stripped_instruction,
            candidates=candidates,
            max_selected_tags=max_selected_tags,
            category_profile=category_profile,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            auxiliary_prompt=auxiliary_prompt,
        )

        normalized_tags = cls._normalize_selected_tags(
            text=text,
            instruction_ja=stripped_instruction,
            candidate_tags=candidate_tags,
            metadata_by_tag=metadata_by_tag,
            max_selected_tags=max_selected_tags,
            catalog=catalog,
        )
        if not normalized_tags:
            raise RuntimeError("候補内のタグを選択できませんでした。")
        display_tags = [_display_tag(tag) for tag in normalized_tags]
        return ",".join(display_tags)

    @staticmethod
    def _build_candidate_line(item: TagSearchResult) -> str:
        return (
            f"- {_display_tag(item.metadata.name)} | "
            f"category={TagEmbeddingCatalog.CATEGORY_LABELS.get(item.metadata.category, 'other')} | "
            f"post_count={item.metadata.post_count}"
        )

    @classmethod
    def _filter_candidates_for_prompt(
        cls,
        instruction_ja: str,
        candidates: list[TagSearchResult],
    ) -> list[TagSearchResult]:
        filtered = [
            item
            for item in candidates
            if _should_allow_metadata_by_default(instruction_ja, item.metadata)
            and (not _is_low_signal_tag(item.metadata.name) or _instruction_explicitly_mentions_tag(instruction_ja, item.metadata))
        ]
        if filtered:
            return filtered
        return [item for item in candidates if not _is_denied_by_strict_policy(item.metadata.name)]

    @classmethod
    def _chat_with_retry(
        cls,
        manager: BonsaiServerManager,
        instruction_ja: str,
        candidates: list[TagSearchResult],
        max_selected_tags: int,
        category_profile: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        auxiliary_prompt: str | None,
    ) -> str:
        candidate_count = len(candidates)
        candidate_limit = candidate_count
        context_size_override: int | None = None
        last_error: BonsaiContextSizeError | None = None

        while candidate_limit >= 1:
            prompt_candidates = candidates[:candidate_limit]
            user_prompt = cls._build_fitted_user_prompt(
                manager=manager,
                instruction_ja=instruction_ja,
                candidates=prompt_candidates,
                max_selected_tags=max_selected_tags,
                category_profile=category_profile,
                max_tokens=max_tokens,
                auxiliary_prompt=auxiliary_prompt,
                context_size_override=context_size_override,
            )
            try:
                return manager.chat(
                    system_prompt=cls.DEFAULT_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    top_k=top_k,
                )
            except BonsaiContextSizeError as exc:
                last_error = exc
                context_size_override = exc.context_size
                next_limit = min(
                    candidate_limit - 1,
                    max(cls.MIN_CANDIDATE_COUNT, (candidate_limit * exc.context_size) // max(exc.prompt_tokens, 1)),
                )
                if next_limit >= candidate_limit:
                    next_limit = candidate_limit - 1
                candidate_limit = next_limit

        if last_error is not None:
            raise RuntimeError(
                "Bonsai の実コンテキスト長に合わせて候補数を削減しても収まりませんでした。"
                "instruction_ja を短くするか、サーバー側の ctx_size をさらに増やしてください。"
            ) from last_error
        raise RuntimeError("Bonsai へのリクエスト生成に失敗しました。")

    @classmethod
    def _build_fitted_user_prompt(
        cls,
        manager: BonsaiServerManager,
        instruction_ja: str,
        candidates: list[TagSearchResult],
        max_selected_tags: int,
        category_profile: str,
        max_tokens: int,
        auxiliary_prompt: str | None,
        context_size_override: int | None = None,
    ) -> str:
        ctx_size = context_size_override if context_size_override is not None else manager.get_context_size()
        system_tokens = manager.estimate_token_count(cls.DEFAULT_SYSTEM_PROMPT)
        fixed_tokens = system_tokens + max_tokens + cls.CONTEXT_MARGIN_TOKENS

        auxiliary_section = ""
        if auxiliary_prompt is not None and auxiliary_prompt.strip():
            auxiliary_section = (
                "補助指示:\n"
                f"{auxiliary_prompt.strip()}\n"
                "補助指示は参考情報であり、候補制約と禁止ルールより優先してはいけません。\n"
            )

        prompt_header = (
            "次の日本語指示に合うタグを候補一覧から選んでください。\n"
            f"指示: {instruction_ja}\n"
            f"category_profile: {category_profile}\n"
            f"最大選択数: {max_selected_tags}\n"
            "ルール:\n"
            "- 候補一覧にあるタグだけを使う\n"
            "- 指示に直接関係するタグだけを選ぶ\n"
            "- 指示にない胸サイズや胸まわりの細部タグは補完しない\n"
            "- 同じ部位に競合する衣装タグを同時に選ばない\n"
            "- full_body と upper_body のような競合構図タグを同時に選ばない\n"
            "- 品質タグ、画風タグ、artist/meta タグは明示要求がない限り選ばない\n"
            "- 人数/主題 -> 版権/キャラ -> 身体属性 -> 顔/目線 -> 髪/頭部 -> 服/装飾 -> ポーズ/構図 -> 背景の順で考える\n"
            "- 出力は `Prompt: tag1, tag2, ...` の1行のみ\n"
            "- category や post_count は参考情報であり、出力には含めない\n"
            f"{auxiliary_section}"
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
        emitted_bucket_headers: set[str] = set()
        grouped_candidates = cls._group_candidates_by_bucket(candidates)
        for bucket in PROMPT_BUCKET_ORDER:
            bucket_items = grouped_candidates.get(bucket, [])
            if not bucket_items:
                continue
            header_line = f"[{PROMPT_BUCKET_LABELS[bucket]}]"
            header_tokens = manager.estimate_token_count(f"{header_line}\n")
            for item in bucket_items:
                candidate_line = cls._build_candidate_line(item)
                candidate_tokens = manager.estimate_token_count(f"{candidate_line}\n")
                additional_tokens = candidate_tokens
                lines_to_add: list[str] = []
                if bucket not in emitted_bucket_headers:
                    additional_tokens += header_tokens
                    lines_to_add.append(header_line)
                lines_to_add.append(candidate_line)

                if selected_lines and (used_candidate_tokens + additional_tokens) > available_candidate_tokens:
                    break
                if not selected_lines and additional_tokens > available_candidate_tokens:
                    raise RuntimeError(
                        "候補一覧を 1 件も収められません。config.json の ctx_size を増やしてください。"
                    )

                if bucket not in emitted_bucket_headers:
                    emitted_bucket_headers.add(bucket)
                selected_lines.extend(lines_to_add)
                used_candidate_tokens += additional_tokens

        if not selected_lines:
            raise RuntimeError("候補一覧を構築できませんでした。")

        return prompt_header + "\n".join(selected_lines)

    @staticmethod
    def _group_candidates_by_bucket(candidates: list[TagSearchResult]) -> dict[str, list[TagSearchResult]]:
        grouped: dict[str, list[TagSearchResult]] = {bucket: [] for bucket in PROMPT_BUCKET_ORDER}
        for item in candidates:
            bucket = _classify_tag_bucket(item.metadata.name, item.metadata)
            grouped.setdefault(bucket, []).append(item)
        return grouped

    @staticmethod
    def _normalize_selected_tags(
        text: str,
        instruction_ja: str,
        candidate_tags: list[str],
        metadata_by_tag: dict[str, TagMetadata],
        max_selected_tags: int,
        catalog: TagEmbeddingCatalog,
    ) -> list[str]:
        normalized_text = _normalize_tags(text)
        raw_tags = [part.strip() for part in normalized_text.split(",") if part.strip()]
        candidate_lookup = _build_candidate_lookup(candidate_tags)
        resolved_tags = [candidate_lookup.get(_normalized_tag_key(tag), tag) for tag in raw_tags]
        valid_catalog_tags = catalog.filter_existing_tags(resolved_tags)
        candidate_tag_set = set(candidate_tags)
        ordered_selected_tags = [tag for tag in valid_catalog_tags if tag in candidate_tag_set]
        strict_filtered_tags: list[str] = []
        for tag in ordered_selected_tags:
            metadata = metadata_by_tag.get(tag)
            if metadata is None:
                continue
            if not _should_allow_metadata_by_default(instruction_ja, metadata):
                continue
            if _is_low_signal_tag(tag) and not _instruction_explicitly_mentions_tag(instruction_ja, metadata):
                continue
            strict_filtered_tags.append(tag)

        filtered_tags = BonsaiCsvTagSelectorNode._filter_conflicting_color_tags(
            ordered_selected_tags=strict_filtered_tags,
            instruction_ja=instruction_ja,
        )
        filtered_tags = _filter_background_conflicts(filtered_tags)
        filtered_tags = _filter_breast_size_conflicts(filtered_tags)
        filtered_tags = _filter_pose_conflicts(filtered_tags)
        filtered_tags = _filter_underwear_theme_conflicts(filtered_tags, instruction_ja=instruction_ja)
        filtered_tags = _filter_subject_conflicts(filtered_tags)
        ordered_output = _apply_output_bucket_order(
            ordered_selected_tags=filtered_tags,
            metadata_by_tag=metadata_by_tag,
        )
        return ordered_output[:max_selected_tags]

    @staticmethod
    def _filter_conflicting_color_tags(ordered_selected_tags: list[str], instruction_ja: str) -> list[str]:
        requested_colors = _extract_requested_colors(instruction_ja)
        if not requested_colors:
            return ordered_selected_tags

        kept_tags: list[str] = []
        requested_color_bases: set[str] = set()
        for tag in ordered_selected_tags:
            color_prefix, base_tag = _split_color_tag(tag)
            if color_prefix is None:
                kept_tags.append(tag)
                continue
            if color_prefix in requested_colors:
                kept_tags.append(tag)
                requested_color_bases.add(base_tag)
                continue
            if _is_clothing_tag(tag) and base_tag in requested_color_bases:
                continue
            kept_tags.append(tag)
        return kept_tags


class BonsaiChatNode:
    CATEGORY = "LLM/Bonsai"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)

    DEFAULT_MAX_CANDIDATES = 256
    DEFAULT_MAX_SELECTED_TAGS = 32
    DEFAULT_CATEGORY_PROFILE = "balanced"
    DEFAULT_SYSTEM_PROMPT = SDXL_DANBOORU_SYSTEM_PROMPT

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
        auxiliary_prompt = self._normalize_auxiliary_prompt(system_prompt)
        tags = BonsaiCsvTagSelectorNode._run_strict_selection(
            instruction_ja=instruction_ja,
            max_candidates=self.DEFAULT_MAX_CANDIDATES,
            max_selected_tags=self.DEFAULT_MAX_SELECTED_TAGS,
            category_profile=self.DEFAULT_CATEGORY_PROFILE,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            rebuild_index=False,
            auxiliary_prompt=auxiliary_prompt,
        )
        return (tags,)

    @classmethod
    def _normalize_auxiliary_prompt(cls, system_prompt: str) -> str | None:
        stripped = system_prompt.strip()
        if not stripped or stripped == cls.DEFAULT_SYSTEM_PROMPT:
            return None
        return stripped
