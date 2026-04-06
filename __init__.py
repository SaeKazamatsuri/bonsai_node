from .bonsai_manager import setup_bonsai_lifecycle
from .bonsai_node import BonsaiChatNode

setup_bonsai_lifecycle()

NODE_CLASS_MAPPINGS = {
    "BonsaiChatNode": BonsaiChatNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BonsaiChatNode": "Bonsai Tag Generator",
}
