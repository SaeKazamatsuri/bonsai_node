from __future__ import annotations

from bonsai_manager import BonsaiServerManager


def main() -> None:
    manager = BonsaiServerManager.instance()
    print(manager.status())


if __name__ == "__main__":
    main()
