"""Gemini text-replacement anonymizer.

Reuses the shared :class:`~generator.llm_client.GeminiCachedClient` for Vertex AI
access, SQLite response caching, and retry/back-off — the same infrastructure the
generation pipeline uses — so benchmark reruns are free and resumable.
"""

from __future__ import annotations

from experiments.benchmark.anonymizers.base import Anonymizer
from experiments.benchmark.prompts import TEXT_INSTRUCTION, TextResponse
from generator.llm_client import GeminiCachedClient


class GeminiAnonymizer(Anonymizer):
    """Gemini-based de-identifier that emits full PHI-replaced text."""

    def __init__(
        self,
        model_name: str = "gemini-3.7-flash",
        cache_db: str = "./experiments/results/benchmark/gemini_anon_cache.db",
        *,
        temperature: float = 0.1,
        retry_attempts: int = 3,
        retry_delay_seconds: int = 5,
    ):
        self.name = f"gemini_{model_name}"
        self.model_name = model_name
        self.temperature = temperature
        self._client = GeminiCachedClient(
            db_path=cache_db,
            model_name=model_name,
            retry_attempts=retry_attempts,
            retry_delay_seconds=retry_delay_seconds,
        )

    def anonymize_text(self, text: str) -> str:
        result = self._client.generate_structured(
            system_instruction=TEXT_INSTRUCTION,
            user_content=text,
            response_schema=TextResponse,
            # Cache key is mode-scoped so it never collides with generation-stage
            # entries that may share the same underlying cache database.
            cache_key=f"benchmark-anonymize|text|{text}",
            temperature=self.temperature,
        )
        return result.get("text", "")
