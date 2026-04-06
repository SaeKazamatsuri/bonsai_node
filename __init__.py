from .bonsai_manager import setup_bonsai_lifecycle
from .bonsai_node import BonsaiChatNode, BonsaiCsvTagSelectorNode

setup_bonsai_lifecycle()

NODE_CLASS_MAPPINGS = {
    "BonsaiChatNode": BonsaiChatNode,
    "BonsaiCsvTagSelectorNode": BonsaiCsvTagSelectorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BonsaiChatNode": "Bonsai Tag Generator",
    "BonsaiCsvTagSelectorNode": "Bonsai Semantic Tag Selector",
}
