from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_PACKAGE_NAME = "bonsai_node_testpkg"


def _load_module(module_name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"モジュールを読み込めません: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_bonsai_node_module() -> types.ModuleType:
    package = sys.modules.get(TEST_PACKAGE_NAME)
    if package is None:
        package = types.ModuleType(TEST_PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]  # type: ignore[attr-defined]
        sys.modules[TEST_PACKAGE_NAME] = package

    manager_module_name = f"{TEST_PACKAGE_NAME}.bonsai_manager"
    node_module_name = f"{TEST_PACKAGE_NAME}.bonsai_node"

    if manager_module_name not in sys.modules:
        _load_module(manager_module_name, REPO_ROOT / "bonsai_manager.py")
    if node_module_name not in sys.modules:
        _load_module(node_module_name, REPO_ROOT / "bonsai_node.py")

    return sys.modules[node_module_name]


MODULE = load_bonsai_node_module()


class FakeManager:
    def __init__(self, response: str, context_size: int = 8192) -> None:
        self._response = response
        self._context_size = context_size
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
        return self._response

    def get_context_size(self) -> int:
        return self._context_size

    @staticmethod
    def estimate_token_count(text: str) -> int:
        if not text:
            return 0
        return max(1, (len(text.encode("utf-8")) + 2) // 3)


class FakeCatalog:
    def __init__(self, candidates: list[object], existing_tags: set[str] | None = None) -> None:
        self._candidates = candidates
        self._existing_tags = existing_tags or {item.metadata.name for item in candidates}

    def search(self, instruction_ja: str, limit: int, category_profile: str, rebuild: bool) -> list[object]:
        return list(self._candidates[:limit])

    def filter_existing_tags(self, tags: list[str]) -> list[str]:
        filtered: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            if tag not in self._existing_tags or tag in seen:
                continue
            filtered.append(tag)
            seen.add(tag)
        return filtered


def make_result(name: str, category: int = 0, post_count: int = 1000) -> object:
    metadata = MODULE.TagMetadata(
        name=name,
        words=name.replace("_", " ").split(),
        category=category,
        post_count=post_count,
        is_deprecated=False,
    )
    return MODULE.TagSearchResult(metadata=metadata, similarity=0.9, score=0.9)


class BonsaiNodeTests(unittest.TestCase):
    def test_chat_node_uses_strict_selection_pipeline(self) -> None:
        candidates = [
            make_result("1girl"),
            make_result("solo"),
            make_result("blush"),
            make_result("brown_hair"),
            make_result("simple_background"),
            make_result("black_background"),
            make_result("quality"),
        ]
        fake_catalog = FakeCatalog(candidates)
        fake_manager = FakeManager("masterpiece, 1girl, solo, blush, brown hair, simple background, black background, quality")

        with patch.object(MODULE.TagEmbeddingCatalog, "instance", return_value=fake_catalog), patch.object(
            MODULE.BonsaiServerManager, "instance", return_value=fake_manager
        ):
            node = MODULE.BonsaiChatNode()
            result = node.run(
                instruction_ja="黒背景で、頬を赤らめた女の子",
                system_prompt="masterpiece を入れてください",
                temperature=0.4,
                max_tokens=128,
                top_p=0.9,
                top_k=20,
            )

        self.assertEqual(result, ("1girl,solo,blush,brown hair,simple background,black background",))
        self.assertEqual(fake_manager.calls[0]["system_prompt"], MODULE.BonsaiCsvTagSelectorNode.DEFAULT_SYSTEM_PROMPT)
        self.assertIn("補助指示", str(fake_manager.calls[0]["user_prompt"]))

    def test_normalize_selected_tags_filters_non_candidates_and_duplicates(self) -> None:
        candidate_tags = ["1girl", "blush", "brown_hair"]
        metadata_by_tag = {item.metadata.name: item.metadata for item in [make_result(tag) for tag in candidate_tags]}
        fake_catalog = FakeCatalog([make_result(tag) for tag in candidate_tags], existing_tags={"1girl", "blush", "brown_hair"})

        normalized = MODULE.BonsaiCsvTagSelectorNode._normalize_selected_tags(
            text="1girl, 1girl, best quality, unknown tag, blush, brown hair",
            instruction_ja="女の子を描く",
            candidate_tags=candidate_tags,
            metadata_by_tag=metadata_by_tag,
            max_selected_tags=16,
            catalog=fake_catalog,
        )

        self.assertEqual(normalized, ["1girl", "blush", "brown_hair"])

    def test_conflict_pruning_handles_solo_breast_size_and_background_color(self) -> None:
        candidate_tags = [
            "1girl",
            "solo",
            "1boy",
            "large_breasts",
            "medium_breasts",
            "simple_background",
            "white_background",
            "black_background",
        ]
        candidate_results = [make_result(tag) for tag in candidate_tags]
        metadata_by_tag = {item.metadata.name: item.metadata for item in candidate_results}
        fake_catalog = FakeCatalog(candidate_results)

        normalized = MODULE.BonsaiCsvTagSelectorNode._normalize_selected_tags(
            text="1girl, solo, 1boy, large breasts, medium breasts, simple background, white background, black background",
            instruction_ja="男女2人を白背景で",
            candidate_tags=candidate_tags,
            metadata_by_tag=metadata_by_tag,
            max_selected_tags=16,
            catalog=fake_catalog,
        )

        self.assertEqual(normalized, ["1girl", "1boy", "large_breasts", "simple_background", "white_background"])

    def test_simple_background_survives_while_quality_is_removed(self) -> None:
        candidate_tags = ["simple_background", "black_background", "quality"]
        candidate_results = [make_result(tag) for tag in candidate_tags]
        metadata_by_tag = {item.metadata.name: item.metadata for item in candidate_results}
        fake_catalog = FakeCatalog(candidate_results)

        normalized = MODULE.BonsaiCsvTagSelectorNode._normalize_selected_tags(
            text="simple background, black background, quality",
            instruction_ja="単純な黒背景",
            candidate_tags=candidate_tags,
            metadata_by_tag=metadata_by_tag,
            max_selected_tags=16,
            catalog=fake_catalog,
        )

        self.assertEqual(normalized, ["simple_background", "black_background"])

    def test_selector_returns_bucketed_deepdanbooru_like_output(self) -> None:
        candidates = [
            make_result("1girl"),
            make_result("solo"),
            make_result("navel"),
            make_result("blush"),
            make_result("brown_hair"),
            make_result("short_hair"),
            make_result("black_shirt"),
            make_result("panties"),
            make_result("bra"),
            make_result("shirt_lift"),
            make_result("wariza"),
            make_result("simple_background"),
            make_result("black_background"),
            make_result("quality"),
        ]
        fake_catalog = FakeCatalog(candidates)
        fake_manager = FakeManager(
            "quality, black shirt, bra, navel, black background, shirt lift, 1girl, solo, blush, brown hair, short hair, panties, wariza, simple background"
        )

        with patch.object(MODULE.TagEmbeddingCatalog, "instance", return_value=fake_catalog), patch.object(
            MODULE.BonsaiServerManager, "instance", return_value=fake_manager
        ):
            node = MODULE.BonsaiCsvTagSelectorNode()
            result = node.run(
                instruction_ja="下着が見えるようにシャツをめくった女の子を座らせる",
                max_candidates=64,
                max_selected_tags=32,
                category_profile="balanced",
                temperature=0.4,
                top_p=0.95,
                top_k=40,
                rebuild_index=False,
            )

        self.assertEqual(
            result,
            ("1girl,solo,navel,blush,brown hair,short hair,black shirt,bra,panties,shirt lift,wariza,black background,simple background",),
        )


if __name__ == "__main__":
    unittest.main()
