"""The model client. This is the only module that knows which provider is in use.

It talks to any OpenAI-compatible chat endpoint (Groq, Ollama, vLLM, OpenAI),
asks for JSON that matches a Pydantic schema, validates every reply, and sends
a reply that does not fit back to the model with the error, a limited number
of times. Open-source models do not always honour a schema, so nothing is used
unvalidated.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Protocol, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from rca.config import JSON_MODES, Settings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Base class. The message is written to be shown to the user."""


class LLMConfigError(LLMError):
    """Wrong key, unknown model, or a request the endpoint will not accept."""


class LLMReplyError(LLMError):
    """The model answered, but not with something usable."""


class LLMRateLimitError(LLMError):
    """The provider's rate limit would need a longer wait than we allow."""


class LLM(Protocol):
    description: str
    usage: dict[str, dict[str, int]]
    on_event: Callable[[str], None] | None

    def structured(self, agent: str, schema: type[T], system: str, user: str) -> T: ...


JSON_ONLY = "Reply with one JSON object and nothing else: no explanation and no markdown fences."
JSON_RULES = JSON_ONLY + " It must match this JSON schema, with every listed key present:\n"
REPAIR = (
    "Your previous reply could not be used: {error}\n"
    "Reply again with only the corrected JSON object."
)

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)
_RETRY_IN = re.compile(r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?", re.I)

MAX_SINGLE_WAIT_S = 90.0


def extract_json(text: str) -> str:
    """Pull the JSON object out of a model reply that may wrap it in reasoning, fences or prose."""
    text = _THINK.sub("", text or "").strip()
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("the reply contained no JSON object")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"the reply was not valid JSON ({exc.msg} at position {exc.pos})") from exc
    return json.dumps(obj)


def tidy(value: Any) -> Any:
    """Remove artefacts models leave in text: a backslash before a quote, and stray whitespace."""
    if isinstance(value, str):
        return value.replace("\\'", "'").replace('\\"', '"').strip()
    if isinstance(value, list):
        return [tidy(item) for item in value]
    if isinstance(value, dict):
        return {key: tidy(item) for key, item in value.items()}
    return value


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = []
        for err in exc.errors()[:6]:
            where = ".".join(str(p) for p in err["loc"]) or "(root)"
            parts.append(f"{where}: {err['msg']}")
        return "; ".join(parts)
    return str(exc)[:400]


def _error_text(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    text = str(exc)
    if body:
        try:
            text += " " + json.dumps(body)
        except (TypeError, ValueError):
            text += " " + str(body)
    return text


def _failed_generation(exc: Exception) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        inner = body.get("error") if isinstance(body.get("error"), dict) else body
        value = inner.get("failed_generation")
        if isinstance(value, str):
            return value
    return ""


def retry_after_seconds(exc: Exception) -> float | None:
    """How long the provider asked us to wait, from the retry-after header or the error text."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw = headers.get("retry-after")
        if raw:
            try:
                return max(0.0, float(raw))
            except ValueError:
                pass
    match = _RETRY_IN.search(_error_text(exc))
    if match and any(match.groups()):
        hours, minutes, seconds = (float(g) if g else 0.0 for g in match.groups())
        return hours * 3600 + minutes * 60 + seconds
    return None


class OpenAICompatLLM:
    """Structured calls against an OpenAI-compatible chat completions endpoint."""

    def __init__(self, settings: Settings, client: Any = None, *,
                 sleep: Callable[[float], None] = time.sleep,
                 on_event: Callable[[str], None] | None = None) -> None:
        self.settings = settings
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key or "not-set",
                max_retries=0,  # retries are handled here so waits can be reported
                timeout=settings.request_timeout_s,
            )
        self.client = client
        self._sleep = sleep
        self.on_event = on_event
        self.usage: dict[str, dict[str, int]] = {}
        self._mode_index: dict[str, int] = {}
        self._dropped: dict[str, set[str]] = {}
        self._tokens_param = settings.max_tokens_param

    # ---- description and bookkeeping ------------------------------------

    @property
    def description(self) -> str:
        models = sorted(set(self.settings.models.values()))
        host = urlparse(self.settings.llm_base_url).netloc or self.settings.llm_base_url
        return f"{', '.join(models)} via {host}"

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())

    def _emit(self, message: str) -> None:
        if self.on_event:
            self.on_event(message)

    def _bump(self, agent: str, key: str, amount: int = 1) -> None:
        stats = self.usage.setdefault(agent, {})
        stats[key] = stats.get(key, 0) + amount

    # ---- structured-output mode -----------------------------------------

    def _mode(self, model: str) -> str:
        if self.settings.json_mode != "auto":
            return self.settings.json_mode
        return JSON_MODES[self._mode_index.get(model, 0)]

    def _downgrade(self, model: str, detail: str) -> bool:
        """Step down one structured-output mode if the endpoint rejected the current one."""
        if self.settings.json_mode != "auto":
            return False
        index = self._mode_index.get(model, 0)
        keywords = ("response_format", "json_schema", "json schema", "strict", "structured output",
                    "json_object", "json mode", "schema")
        if index >= len(JSON_MODES) - 1 or not any(k in detail for k in keywords):
            return False
        self._mode_index[model] = index + 1
        self._emit(f"{model} does not accept {JSON_MODES[index]}; using {JSON_MODES[index + 1]}")
        return True

    @staticmethod
    def _response_format(mode: str, schema: type[BaseModel], schema_json: dict) -> dict | None:
        if mode == "json_schema_strict":
            return {"type": "json_schema",
                    "json_schema": {"name": schema.__name__, "strict": True, "schema": schema_json}}
        if mode == "json_schema":
            return {"type": "json_schema", "json_schema": {"name": schema.__name__, "schema": schema_json}}
        if mode == "json_object":
            return {"type": "json_object"}
        return None

    def _drop_parameter(self, model: str, detail: str) -> bool:
        """Stop sending an optional parameter this endpoint or model does not accept."""
        dropped = self._dropped.setdefault(model, set())
        if "max_completion_tokens" in detail and self._tokens_param == "max_completion_tokens":
            self._tokens_param = "max_tokens"
            return True
        for name in ("reasoning_effort", "temperature"):
            if name in detail and name not in dropped:
                dropped.add(name)
                return True
        return False

    # ---- one request, with waits and fallbacks --------------------------

    def _system(self, system: str, mode: str, schema_json: dict) -> dict:
        """The system message. When the endpoint enforces the schema itself (strict mode) the schema
        is not repeated in the prompt, which saves a few hundred tokens on every call."""
        if mode == "json_schema_strict":
            return {"role": "system", "content": system.strip() + "\n\n" + JSON_ONLY}
        return {"role": "system", "content": system.strip() + "\n\n" + JSON_RULES + json.dumps(schema_json)}

    def _request(self, agent: str, model: str, schema: type[BaseModel], schema_json: dict,
                 system: str, conversation: list[dict]) -> str:
        from openai import (APIConnectionError, APIStatusError, AuthenticationError, BadRequestError,
                            NotFoundError, PermissionDeniedError, RateLimitError)

        waited = 0.0
        rate_hits = 0
        transient = 0
        budget = self.settings.max_output_tokens
        while True:
            dropped = self._dropped.get(model, set())
            mode = self._mode(model)
            kwargs: dict[str, Any] = {"model": model,
                                      "messages": [self._system(system, mode, schema_json)] + conversation,
                                      self._tokens_param: budget}
            if self.settings.temperature is not None and "temperature" not in dropped:
                kwargs["temperature"] = self.settings.temperature
            if self.settings.reasoning_effort and "reasoning_effort" not in dropped:
                kwargs["reasoning_effort"] = self.settings.reasoning_effort
            response_format = self._response_format(mode, schema, schema_json)
            if response_format:
                kwargs["response_format"] = response_format
            try:
                response = self.client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                wait = retry_after_seconds(exc)
                if wait is None:
                    wait = min(60.0, 5.0 * 2 ** rate_hits)
                rate_hits += 1
                wait += 0.5
                if wait > MAX_SINGLE_WAIT_S or waited + wait > self.settings.max_rate_wait_s:
                    raise LLMRateLimitError(
                        f"The rate limit for {model} was reached and the provider asks for a wait of "
                        f"about {wait:.0f} seconds. Try again later, or set a different model in .env."
                    ) from exc
                self._bump(agent, "rate_limit_waits")
                self._emit(f"Rate limit reached on {model}; waiting {wait:.0f}s")
                self._sleep(wait)
                waited += wait
                continue
            except BadRequestError as exc:
                detail = _error_text(exc).lower()
                if any(k in detail for k in ("json_validate_failed", "failed_generation",
                                             "failed to generate json", "failed to validate json")):
                    # The provider enforced the schema and the model's output failed it.
                    self._bump(agent, "failed_generations")
                    return _failed_generation(exc)
                if self._drop_parameter(model, detail) or self._downgrade(model, detail):
                    continue
                raise LLMConfigError(
                    f"The model endpoint rejected the request for {model}: {_error_text(exc)[:300]}"
                ) from exc
            except AuthenticationError as exc:
                raise LLMConfigError("The LLM API key was rejected. Check LLM_API_KEY in .env.") from exc
            except PermissionDeniedError as exc:
                raise LLMConfigError(f"This API key is not allowed to use {model}.") from exc
            except NotFoundError as exc:
                raise LLMConfigError(
                    f"Model '{model}' was not found at {self.settings.llm_base_url}. Check LLM_MODEL in .env."
                ) from exc
            except APIConnectionError as exc:  # includes timeouts
                transient += 1
                if transient > 2:
                    raise LLMError(
                        f"Could not reach the model endpoint at {self.settings.llm_base_url}: {exc}"
                    ) from exc
                self._sleep(2.0 * transient)
                continue
            except APIStatusError as exc:
                if exc.status_code >= 500 and transient < 2:
                    transient += 1
                    self._sleep(2.0 * transient)
                    continue
                raise LLMError(
                    f"The model endpoint returned an error ({exc.status_code}): {_error_text(exc)[:300]}"
                ) from exc

            self._bump(agent, "calls")
            usage = getattr(response, "usage", None)
            if usage is not None:
                self._bump(agent, "input_tokens", getattr(usage, "prompt_tokens", 0) or 0)
                self._bump(agent, "output_tokens", getattr(usage, "completion_tokens", 0) or 0)
            choice = response.choices[0]
            if choice.finish_reason == "length":
                # Start small and grow only when needed: a large limit on every call wastes
                # a provider's per-minute token budget.
                if budget < self.settings.max_output_tokens * 2:
                    budget = self.settings.max_output_tokens * 2
                    self._bump(agent, "cut_off_retries")
                    self._emit(f"The reply from {model} was cut off; asking again with a larger limit")
                    continue
                raise LLMReplyError(
                    f"The reply from {model} was cut off before it finished, even at {budget} tokens. "
                    "Raise LLM_MAX_OUTPUT_TOKENS in .env, or use shorter notes."
                )
            return choice.message.content or ""

    # ---- public API ------------------------------------------------------

    def structured(self, agent: str, schema: type[T], system: str, user: str) -> T:
        """Ask the model for JSON matching `schema`; validate it, repairing a bad reply if needed."""
        model = self.settings.models[agent]
        schema_json = schema.model_json_schema()
        conversation: list[dict] = [{"role": "user", "content": user}]
        attempts = self.settings.max_repairs + 1
        last_error = "no reply"
        for attempt in range(attempts):
            text = self._request(agent, model, schema, schema_json, system, conversation)
            try:
                return schema.model_validate(tidy(json.loads(extract_json(text))))
            except ValueError as exc:  # pydantic's ValidationError is a ValueError
                last_error = _short_error(exc)
                if attempt + 1 < attempts:
                    self._bump(agent, "repairs")
                    conversation = conversation + [
                        {"role": "assistant", "content": text or "(empty reply)"},
                        {"role": "user", "content": REPAIR.format(error=last_error)},
                    ]
        raise LLMReplyError(
            f"The {agent} agent's model ({model}) did not return a usable answer after "
            f"{attempts} attempts. Last problem: {last_error}"
        )


def build_llm(settings: Settings, on_event: Callable[[str], None] | None = None) -> LLM:
    if settings.llm_backend == "demo":
        from rca.llm_fake import demo_llm

        return demo_llm()
    if not settings.llm_api_key:
        raise LLMConfigError(
            "No LLM API key is configured. Set LLM_API_KEY in .env (a free Groq key works), "
            "or set LLM_BACKEND=demo to try the app offline with a rule-based stand-in."
        )
    return OpenAICompatLLM(settings, on_event=on_event)
