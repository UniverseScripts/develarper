# handlers/code_gen.py
"""Code generation handler — local-first with lean API fallback.

Strategy (mirrors the AiTesting "Surgical Strike" approach):
  easy/medium  → local Qwen 3B (0 API tokens) → validate via ast.parse
                 → lean API fallback only if validation fails
  hard         → direct lean API call (skip wasted local attempt)
"""

import logging

from engines.local_slm import LocalSLMEngine
from engines.remote_llm import RemoteLLMEngine
from handlers._base import load_prompt_template
from handlers.code_utils import (
    HARD,
    classify_code_difficulty,
    extract_code,
    validate_code,
)

logger = logging.getLogger(__name__)


class CodeGenHandler:
    """Generates Python code with minimal API token spend."""

    def __init__(self) -> None:
        self.local = LocalSLMEngine.get_instance()
        self.remote = RemoteLLMEngine()
        self.local_prompt = load_prompt_template("local_code_gen.txt")
        self.remote_prompt = load_prompt_template("remote_code.txt")

    async def handle(self, prompt: str) -> str:
        difficulty = classify_code_difficulty(prompt)
        logger.info("CodeGen difficulty=%s", difficulty)

        # Hard tasks skip the local attempt — it almost always fails and
        # burns wall-clock time for nothing.
        if difficulty == HARD:
            raw = await self.remote.generate(
                prompt=prompt,
                category="API_CODE",
                system_prompt=self.remote_prompt,
                max_tokens=500,
            )
            return extract_code(raw)

        # Local-first for easy/medium (0 tokens)
        local_raw = self.local.generate(
            prompt=f"{prompt}\n\nImplement as Python code.",
            system_prompt=self.local_prompt,
            max_tokens=1024,
        )

        if local_raw != "__ESCALATE__":
            code = extract_code(local_raw)
            result = validate_code(code)
            if result["is_valid"]:
                logger.info("CodeGen solved locally (0 tokens)")
                return code
            logger.info("CodeGen local invalid: %s → API fallback", result["reason"])

        # Lean one-shot API fallback
        raw = await self.remote.generate(
            prompt=prompt,
            category="API_CODE",
            system_prompt=self.remote_prompt,
            max_tokens=500,
        )
        return extract_code(raw)
