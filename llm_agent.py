"""Local Ollama decision-support client with strict structured-output validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlparse

import requests

import config


SYSTEM_PROMPT = """You are an offline pedagogical decision-support assistant.
You receive aggregate classroom engagement metrics, never personal identities or
biometric images. Produce exactly three practical, non-intrusive active-learning
interventions for the instructor. Each intervention must be immediately usable,
respect learner autonomy, avoid diagnosis or surveillance language, and take no
more than five minutes. Adapt each intervention to the observed signal pattern.
Return JSON only, matching this schema exactly:
{"interventions":[{"title":"string","action":"string","rationale":"string"},
{"title":"string","action":"string","rationale":"string"},
{"title":"string","action":"string","rationale":"string"}]}
"""


class LocalLLMError(RuntimeError):
    """Raised when a local LLM response is unavailable or unsafe to consume."""


@dataclass(frozen=True)
class TeachingSuggestion:
    """One validated intervention suitable for terminal or GUI rendering."""

    title: str
    action: str
    rationale: str


class LocalTeachingAgent:
    """Calls only a loopback Ollama server and validates its JSON response."""

    def __init__(
        self,
        endpoint: str = config.OLLAMA_GENERATE_ENDPOINT,
        model_tag: str = config.OLLAMA_MODEL_TAG,
        timeout_seconds: int = config.OLLAMA_REQUEST_TIMEOUT_SECONDS,
        http_post: Optional[Callable[..., Any]] = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("The local LLM endpoint must use an HTTP loopback address.")
        self.endpoint = endpoint
        self.model_tag = model_tag
        self.timeout_seconds = timeout_seconds
        self._http_post = http_post or requests.post

    @staticmethod
    def _parse_suggestions(payload: Mapping[str, Any]) -> List[TeachingSuggestion]:
        raw_suggestions = payload.get("interventions")
        if not isinstance(raw_suggestions, list) or len(raw_suggestions) != 3:
            raise LocalLLMError("Ollama output must contain exactly three interventions.")
        suggestions: List[TeachingSuggestion] = []
        for item in raw_suggestions:
            if not isinstance(item, dict):
                raise LocalLLMError("Every intervention must be an object.")
            title = item.get("title")
            action = item.get("action")
            rationale = item.get("rationale")
            if not all(isinstance(value, str) and value.strip() for value in (title, action, rationale)):
                raise LocalLLMError("Every intervention needs non-empty title, action, and rationale.")
            suggestions.append(
                TeachingSuggestion(title=title.strip(), action=action.strip(), rationale=rationale.strip())
            )
        return suggestions

    def request_suggestions(self, metrics_payload: Mapping[str, Any]) -> List[TeachingSuggestion]:
        """Send aggregate metrics to local Ollama and return three safe suggestions."""
        body: Dict[str, Any] = {
            "model": self.model_tag,
            "system": SYSTEM_PROMPT,
            "prompt": json.dumps(dict(metrics_payload), sort_keys=True),
            "format": "json",
            "stream": False,
            "think": False,
            "options": {"temperature": 0.25, "num_predict": 420},
        }
        try:
            response = self._http_post(self.endpoint, json=body, timeout=self.timeout_seconds)
            response.raise_for_status()
            response_body = response.json()
            generated_text = response_body.get("response")
            if not isinstance(generated_text, str):
                raise LocalLLMError("Ollama did not return a textual response.")
            parsed = json.loads(generated_text)
            if not isinstance(parsed, dict):
                raise LocalLLMError("Ollama response JSON must be an object.")
            return self._parse_suggestions(parsed)
        except (requests.RequestException, ValueError, json.JSONDecodeError) as error:
            raise LocalLLMError(f"Local Ollama request failed: {error}") from error
