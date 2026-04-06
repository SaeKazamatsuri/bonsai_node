from __future__ import annotations

from .bonsai_manager import BonsaiServerManager


class BonsaiChatNode:
    CATEGORY = "LLM/Bonsai"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("tags",)

    DEFAULT_SYSTEM_PROMPT = (
        "あなたは画像生成向けタグ生成アシスタントです。"
        "入力された日本語の指示を読み取り、内容に合う短いタグを英語中心で生成してください。"
        "出力は1行のカンマ区切りタグのみとし、説明文、番号、改行、前置きは禁止です。"
        "人物、構図、背景、雰囲気、品質、色、ライティングが必要なら含めてください。"
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
        return (self._normalize_tags(text),)

    @staticmethod
    def _build_user_prompt(instruction_ja: str) -> str:
        stripped = instruction_ja.strip()
        if not stripped:
            raise ValueError("instruction_ja を入力してください。")
        return (
            "次の日本語指示を、画像生成向けのカンマ区切りタグへ変換してください。\n"
            f"指示: {stripped}"
        )

    @staticmethod
    def _normalize_tags(text: str) -> str:
        normalized = text.replace("\r", " ").replace("\n", " ").strip()
        return ",".join(part.strip() for part in normalized.split(",") if part.strip())
