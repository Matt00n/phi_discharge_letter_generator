import hashlib
import json
import os
import random as _random
import sqlite3
import time
from typing import Any, Dict

from google import genai
from google.genai import types

# Errors that are transient and safe to retry (uses google-genai's own error types,
# no separate google-api-core dependency required).
from google.genai.errors import ClientError, ServerError
from pydantic import BaseModel

# 5xx, network-level, and timeout errors are always retriable.
_TRANSIENT_EXCEPTIONS = (
    ServerError,
    ConnectionError,
    TimeoutError,
    OSError,
)
# ClientError is retriable only for 429 (rate limit) and 408/499 (timeout/cancelled).
_RETRIABLE_CLIENT_CODES = frozenset({408, 429, 499})


class GeminiCachedClient:
    def __init__(
        self,
        db_path: str = "./gemini_cache.db",
        model_name: str = "gemini-3.7-flash",
        *,
        retry_attempts: int = 3,
        retry_delay_seconds: int = 2,
    ):
        self.db_path = db_path
        self.model_name = model_name
        # Default retry policy for structured calls; overridable per call. These
        # defaults are the values the pipeline has always used in practice.
        self.retry_attempts = retry_attempts
        self.retry_delay_seconds = retry_delay_seconds

        api_key = os.environ.get("GEMINI_API_KEY")
        vertex_project = os.environ.get("GCP_PROJECT_ID")
        vertex_location = os.environ.get("GCP_LOCATION", "eu")
        if api_key and not vertex_project:
            self.client = genai.Client(api_key=api_key)
        else:
            if not vertex_project:
                raise ValueError(
                    "Set GCP_PROJECT_ID (Vertex AI) or GEMINI_API_KEY "
                    "(Google AI Studio) in your environment."
                )
            self.client = genai.Client(
                vertexai=True,
                project=vertex_project,
                location=vertex_location if vertex_location else None,
            )

        self._init_db()

    def _init_db(self):
        """Initializes SQLite database for local response caching."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()
        cursor.execute(
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

    def generate_structured(
        self,
        *,
        system_instruction: str,
        user_content: str,
        response_schema: type[BaseModel],
        cache_key: str,
        temperature: float = 0.7,
        retries: int | None = None,
        delay: int | None = None,
    ) -> Dict[str, Any]:
        """Generic cached structured-output call.

        ``cache_key`` should uniquely identify the request (e.g. a hash of the
        prompt + schema name + model). Results are cached so reruns are free and
        interrupted batches resume instantly.

        ``retries``/``delay`` fall back to the client's configured retry policy
        (``retry_attempts``/``retry_delay_seconds``) when not given explicitly.
        """
        retries = self.retry_attempts if retries is None else retries
        delay = self.retry_delay_seconds if delay is None else delay
        key_hash = hashlib.sha256(
            f"{self.model_name}|{cache_key}".encode("utf-8")
        ).hexdigest()

        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        row = conn.execute(
            "SELECT response_json FROM structured_cache WHERE cache_key = ?",
            (key_hash,),
        ).fetchone()
        conn.close()
        if row:
            return json.loads(row[0])

        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=temperature,
                        system_instruction=system_instruction,
                    ),
                )
                result = json.loads(response.text)
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    "INSERT OR REPLACE INTO structured_cache (cache_key, model_name, response_json) VALUES (?, ?, ?)",
                    (key_hash, self.model_name, json.dumps(result)),
                )
                conn.commit()
                conn.close()
                return result
            except _TRANSIENT_EXCEPTIONS as e:
                print(f"API transient error on attempt {attempt + 1}/{retries}: {e}")
                if attempt < retries - 1:
                    base = delay * (2**attempt)
                    jitter = _random.uniform(0, delay)
                    sleep_secs = base + jitter
                    print(f"  Retrying in {sleep_secs:.1f}s...")
                    time.sleep(sleep_secs)
                else:
                    raise
            except ClientError as e:
                if e.code not in _RETRIABLE_CLIENT_CODES:
                    print(f"API permanent error (code={e.code}): {e}")
                    raise
                is_rate_limit = e.code == 429
                print(
                    f"API {'rate-limit' if is_rate_limit else 'client'} error (code={e.code}) on attempt {attempt + 1}/{retries}: {e}"
                )
                if attempt < retries - 1:
                    base = delay * (2**attempt)
                    jitter = _random.uniform(0, delay)
                    # Honour Retry-After if the exception carries it.
                    retry_after = getattr(e, "retry_delay", None)
                    if retry_after is not None:
                        sleep_secs = (
                            max(float(retry_after.total_seconds()), base) + jitter
                        )
                    else:
                        sleep_secs = base + jitter
                    print(f"  Retrying in {sleep_secs:.1f}s...")
                    time.sleep(sleep_secs)
                else:
                    raise
            except Exception as e:  # non-retriable — fail immediately
                print(f"API permanent error: {e}")
                raise

        raise RuntimeError("Failed structured generation after retries.")
