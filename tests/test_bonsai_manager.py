from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_PACKAGE_NAME = "bonsai_manager_testpkg"


def _load_module(module_name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"モジュールを読み込めません: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_bonsai_manager_module() -> types.ModuleType:
    package = sys.modules.get(TEST_PACKAGE_NAME)
    if package is None:
        package = types.ModuleType(TEST_PACKAGE_NAME)
        package.__path__ = [str(REPO_ROOT)]  # type: ignore[attr-defined]
        sys.modules[TEST_PACKAGE_NAME] = package

    manager_module_name = f"{TEST_PACKAGE_NAME}.bonsai_manager"
    if manager_module_name not in sys.modules:
        _load_module(manager_module_name, REPO_ROOT / "bonsai_manager.py")

    return sys.modules[manager_module_name]


MODULE = load_bonsai_manager_module()


class BonsaiManagerLifecycleTests(unittest.TestCase):
    def test_setup_lifecycle_does_not_warmup_server(self) -> None:
        manager = Mock()

        with patch.object(MODULE.BonsaiServerManager, "instance", return_value=manager), patch.object(
            MODULE,
            "_get_prompt_server_instance",
            return_value=None,
        ), patch.object(MODULE.atexit, "register"):
            MODULE.setup_bonsai_lifecycle()

        manager.warmup_async.assert_not_called()
        manager.stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
