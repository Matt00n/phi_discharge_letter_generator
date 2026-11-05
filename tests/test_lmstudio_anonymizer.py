"""Tests for the LM Studio (local open-source LLM) de-identification method.

The live LM Studio server is not required: ``httpx.post`` is monkeypatched so the
transport, caching, retry, and prompt wiring are exercised offline. These assert
that the method reuses the *exact* Gemini setup (prompt, schema, cache key).
"""

import json as _json

import pytest

httpx = pytest.importorskip("httpx")

from experiments.benchmark.anonymizers import build_anonymizer  # noqa: E402
from experiments.benchmark.anonymizers.lmstudio import (  # noqa: E402
    LMStudioAnonymizer,
    _slug,
)
from experiments.benchmark.config import AnonymizerConfig  # noqa: E402
from experiments.benchmark.prompts import TEXT_INSTRUCTION  # noqa: E402


class _FakeResponse:
    def __init__(self, content: str, status: int = 200):
        self._content = content
        self.status_code = status
        self.request = None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def _make_anon(tmp_path, **kwargs) -> LMStudioAnonymizer:
    return LMStudioAnonymizer(
        model=kwargs.pop("model", "google/gemma-4-e2b"),
        cache_db=str(tmp_path / "cache.db"),
        retry_delay_seconds=0,  # keep retry back-off instant in tests
        **kwargs,
    )


def test_slug():
    assert _slug("google/gemma-4-e2b") == "gemma-4-e2b"
    assert _slug("bartowski/Llama-3.1-SauerkrautLM-8b-Instruct-GGUF") == (
        "Llama-3.1-SauerkrautLM-8b-Instruct-GGUF"
    )


def test_registered_via_config(tmp_path):
    built = build_anonymizer(
        AnonymizerConfig(
            name="lmstudio",
            lmstudio_model="google/gemma-4-e2b",
            cache_db=tmp_path / "cache.db",
        )
    )
    assert isinstance(built, LMStudioAnonymizer)
    assert built.name == "lmstudio_gemma-4-e2b"


def test_uses_gemini_prompt_and_schema(tmp_path, monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return _FakeResponse(_json.dumps({"text": "Patient [REDACTED]."}))

    monkeypatch.setattr(httpx, "post", fake_post)
    anon = _make_anon(tmp_path)
    out = anon.anonymize_text("Patient Wilfried Hecht.")

    assert out == "Patient [REDACTED]."
    payload = captured["payload"]
    # Exact same prompt as the Gemini method.
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == TEXT_INSTRUCTION
    assert payload["messages"][1]["content"] == "Patient Wilfried Hecht."
    # Structured output constrained to the shared TextResponse schema.
    rf = payload["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "TextResponse"
    assert rf["json_schema"]["schema"]["properties"]["text"]["type"] == "string"
    assert payload["model"] == "google/gemma-4-e2b"


def test_cache_avoids_second_call(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_post(url, json=None, timeout=None):
        calls["n"] += 1
        return _FakeResponse(_json.dumps({"text": "[REDACTED]"}))

    monkeypatch.setattr(httpx, "post", fake_post)
    anon = _make_anon(tmp_path)
    assert anon.anonymize_text("hello") == "[REDACTED]"
    assert anon.anonymize_text("hello") == "[REDACTED]"
    assert calls["n"] == 1  # second call served from cache


def test_retries_on_transient_error(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_post(url, json=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused")
        return _FakeResponse(_json.dumps({"text": "[REDACTED]"}))

    monkeypatch.setattr(httpx, "post", fake_post)
    anon = _make_anon(tmp_path)
    assert anon.anonymize_text("hi") == "[REDACTED]"
    assert calls["n"] == 2
