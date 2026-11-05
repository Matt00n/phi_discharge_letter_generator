"""LM Studio (local open-source LLM) text-replacement anonymizer.

Runs an open-source LLM served locally by `LM Studio <https://lmstudio.ai>`_
through its OpenAI-compatible HTTP server. This is a **drop-in backend swap** for
:class:`~experiments.benchmark.anonymizers.gemini.GeminiAnonymizer`: it uses the
exact same de-identification prompt (``TEXT_INSTRUCTION``), the same
structured-output schema (``TextResponse``), the same mode-scoped cache key, and
the same SQLite response cache + retry/back-off policy. Only the model provider
differs, so the LM Studio results are directly comparable to the Gemini baseline.

A single class serves all local models; the concrete model is selected by the
``model`` identifier (the model key LM Studio exposes at ``GET /v1/models``), so
the three benchmark LLMs (SauerkrautLM, Qwen3, Gemma) are three configurations of
this one method, mirroring how ``gemini`` is one method parameterised by ``model``.

``httpx`` is imported lazily so the harness and config import without the
experiment dependency group installed.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sqlite3
import time
from typing import Any

from pydantic import BaseModel

from experiments.benchmark.anonymizers.base import Anonymizer
from experiments.benchmark.prompts import TEXT_INSTRUCTION, TextResponse

#: Default LM Studio OpenAI-compatible endpoint.
DEFAULT_BASE_URL = "http://localhost:1234/v1"

#: HTTP status codes worth retrying (rate limit, and "model still loading").
_RETRIABLE_STATUS = frozenset({408, 429, 503})


def _slug(model: str) -> str:
    """Filesystem-safe identifier derived from an LM Studio model key."""
    tail = model.split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tail).strip("-") or "model"


class LMStudioCachedClient:
    """Cached, retrying client for LM Studio's OpenAI-compatible chat API.

    Mirrors :class:`generator.llm_client.GeminiCachedClient`: same SQLite cache
    schema, same ``generate_structured`` contract, same exponential back-off with
    jitter — so behaviour matches the Gemini baseline apart from the transport.
    """

    def __init__(
        self,
        db_path: str = "./lmstudio_anon_cache.db",
        model_name: str = "local-model",
        *,
        base_url: str = DEFAULT_BASE_URL,
        retry_attempts: int = 3,
        retry_delay_seconds: int = 5,
        timeout: float = 600.0,
    ) -> None:
        self.db_path = db_path
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.retry_attempts = retry_attempts
        self.retry_delay_seconds = retry_delay_seconds
        self.timeout = timeout
        self._init_db()

    def _init_db(self) -> None:
        """Initialise the SQLite cache (same schema as the Gemini client)."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS structured_cache (
                cache_key TEXT PRIMARY KEY,
                model_name TEXT,
                response_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()

    def _cache_get(self, key_hash: str) -> dict[str, Any] | None:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        row = conn.execute(
            "SELECT response_json FROM structured_cache WHERE cache_key = ?",
            (key_hash,),
        ).fetchone()
        conn.close()
        return json.loads(row[0]) if row else None

    def _cache_put(self, key_hash: str, result: dict[str, Any]) -> None:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT OR REPLACE INTO structured_cache "
            "(cache_key, model_name, response_json) VALUES (?, ?, ?)",
            (key_hash, self.model_name, json.dumps(result)),
        )
        conn.commit()
        conn.close()

    def generate_structured(
        self,
        *,
        system_instruction: str,
        user_content: str,
        response_schema: type[BaseModel],
        cache_key: str,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        retries: int | None = None,
        delay: int | None = None,
    ) -> dict[str, Any]:
        """Cached structured chat completion against LM Studio.

        Same contract as :meth:`GeminiCachedClient.generate_structured`. The
        JSON-schema of ``response_schema`` is passed as an OpenAI
        ``response_format`` so LM Studio constrains decoding to valid JSON.
        """
        import httpx

        retries = self.retry_attempts if retries is None else retries
        delay = self.retry_delay_seconds if delay is None else delay
        key_hash = hashlib.sha256(f"{self.model_name}|{cache_key}".encode()).hexdigest()

        cached = self._cache_get(key_hash)
        if cached is not None:
            return cached

        schema = response_schema.model_json_schema()
        # Strict mode rejects unspecified keys; be explicit for constrained decoding.
        schema.setdefault("additionalProperties", False)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        url = f"{self.base_url}/chat/completions"
        transient = (
            httpx.TransportError,
            httpx.TimeoutException,
            ConnectionError,
            TimeoutError,
            OSError,
        )

        for attempt in range(retries):
            try:
                response = httpx.post(url, json=payload, timeout=self.timeout)
                if response.status_code in _RETRIABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retriable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                result = json.loads(content)
                self._cache_put(key_hash, result)
                return result
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status not in _RETRIABLE_STATUS:
                    print(f"LM Studio permanent error (status={status}): {e}")
                    raise
                print(
                    f"LM Studio status {status} on attempt {attempt + 1}/{retries}: {e}"
                )
                if attempt < retries - 1:
                    self._sleep_backoff(attempt, delay)
                else:
                    raise
            except transient as e:
                print(
                    f"LM Studio transient error on attempt {attempt + 1}/{retries}: {e}"
                )
                if attempt < retries - 1:
                    self._sleep_backoff(attempt, delay)
                else:
                    raise
            except Exception as e:  # non-retriable (e.g. malformed JSON)
                print(f"LM Studio permanent error: {e}")
                raise

        raise RuntimeError("Failed structured generation after retries.")

    @staticmethod
    def _sleep_backoff(attempt: int, delay: int) -> None:
        base = delay * (2**attempt)
        sleep_secs = base + random.uniform(0, delay)
        print(f"  Retrying in {sleep_secs:.1f}s...")
        time.sleep(sleep_secs)


class LMStudioAnonymizer(Anonymizer):
    """Local open-source LLM de-identifier served through LM Studio.

    Structurally identical to :class:`GeminiAnonymizer` — same prompt, schema, and
    cache key — so only the model backend differs.
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = DEFAULT_BASE_URL,
        cache_db: str = "./experiments/results/benchmark/lmstudio_anon_cache.db",
        *,
        temperature: float = 0.1,
        retry_attempts: int = 3,
        retry_delay_seconds: int = 5,
        timeout: float = 600.0,
        max_tokens: int | None = 16384,
    ) -> None:
        self.name = f"lmstudio_{_slug(model)}"
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = LMStudioCachedClient(
            db_path=cache_db,
            model_name=model,
            base_url=base_url,
            retry_attempts=retry_attempts,
            retry_delay_seconds=retry_delay_seconds,
            timeout=timeout,
        )

    def anonymize_text(self, text: str) -> str:
        result = self._client.generate_structured(
            system_instruction=TEXT_INSTRUCTION,
            user_content=text,
            response_schema=TextResponse,
            # Same mode-scoped key as the Gemini method so the two never collide
            # if they ever share a cache database.
            cache_key=f"benchmark-anonymize|text|{text}",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return result.get("text", "")
