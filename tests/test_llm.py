"""The model client: validate-and-repair, structured-output fallbacks, rate-limit waits."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import BaseModel

from rca.config import load_settings
from rca.llm import (LLMConfigError, LLMRateLimitError, LLMReplyError, OpenAICompatLLM, build_llm,
                     extract_json, retry_after_seconds)


class Answer(BaseModel):
    city: str
    population: int


GOOD = json.dumps({"city": "Pune", "population": 7})


def completion(text: str, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )


def api_error(cls, status: int, message: str, headers: dict | None = None, body: dict | None = None):
    request = httpx2.Request("POST", "https://api.example.test/v1/chat/completions")
    response = httpx2.Response(status, request=request, headers=headers or {})
    return cls(message, response=response, body=body or {"error": {"message": message}})


class FakeClient:
    """Stands in for openai.OpenAI: replays a script of replies and errors, and records each request."""

    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def settings():
    return load_settings({"LLM_API_KEY": "test-key", "LLM_MODEL": "test-model"}, use_dotenv=False)


def make(settings, script, **kwargs):
    waits: list[float] = []
    events: list[str] = []
    client = FakeClient(script)
    llm = OpenAICompatLLM(settings, client, sleep=waits.append, on_event=events.append, **kwargs)
    return llm, client, waits, events


def ask(llm):
    return llm.structured("extractor", Answer, "You answer questions.", "Which city?")


# ---- parsing -------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    GOOD,
    f"```json\n{GOOD}\n```",
    f"<think>The user wants a city. Maybe {{not json}}.</think>\n{GOOD}",
    f"Here is the answer:\n{GOOD}\nHope that helps.",
])
def test_json_is_found_inside_wrapping(text):
    assert json.loads(extract_json(text)) == {"city": "Pune", "population": 7}


@pytest.mark.parametrize("text", ["", "no json here", '{"city": "Pune", ', "<think>cut off before the answer"])
def test_unusable_replies_raise(text):
    with pytest.raises(ValueError):
        extract_json(text)


# ---- validate and repair -------------------------------------------------------

def test_valid_reply_needs_one_call(settings):
    llm, client, _, _ = make(settings, [completion(GOOD)])
    assert ask(llm) == Answer(city="Pune", population=7)
    request = client.requests[0]
    assert request["model"] == "test-model"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert "population" not in request["messages"][0]["content"], \
        "the endpoint enforces the schema, so it is not repeated in the prompt"
    assert request["max_completion_tokens"] == settings.max_output_tokens == 4000
    assert llm.usage["extractor"] == {"calls": 1, "input_tokens": 100, "output_tokens": 20}


def test_bad_reply_is_sent_back_with_the_error(settings):
    wrong = json.dumps({"city": "Pune", "population": "lots"})
    llm, client, _, _ = make(settings, [completion(wrong), completion(GOOD)])
    assert ask(llm).population == 7
    repair = client.requests[1]["messages"]
    assert repair[0]["role"] == "system" and repair[1] == {"role": "user", "content": "Which city?"}
    assert repair[-2] == {"role": "assistant", "content": wrong}
    assert "population" in repair[-1]["content"], "the model is told what was wrong"
    assert llm.usage["extractor"]["repairs"] == 1


def test_model_artefacts_are_cleaned_from_every_string(settings):
    backslash = chr(92)  # models sometimes write doesn\'t, with a real backslash in the text
    messy = f" CI doesn{backslash}'t check {backslash}\"pool{backslash}\" size "
    assert backslash in messy
    reply = json.dumps({"city": messy, "population": 7})
    llm, _, _, _ = make(settings, [completion(reply)])
    assert ask(llm).city == "CI doesn't check \"pool\" size"


def test_gives_up_after_the_repair_limit(settings):
    llm, client, _, _ = make(settings, [completion("nonsense")] * 3)
    with pytest.raises(LLMReplyError) as error:
        ask(llm)
    assert len(client.requests) == settings.max_repairs + 1 == 3
    assert "3 attempts" in str(error.value)


def test_a_cut_off_reply_is_retried_once_with_a_larger_limit(settings):
    cut = completion('{"city": "Pu', finish_reason="length")
    llm, client, _, events = make(settings, [cut, completion(GOOD)])
    assert ask(llm).city == "Pune"
    assert [r["max_completion_tokens"] for r in client.requests] == [4000, 8000]
    assert llm.usage["extractor"]["cut_off_retries"] == 1 and "cut off" in events[0]

    llm, client, _, _ = make(settings, [cut, cut])
    with pytest.raises(LLMReplyError, match="cut off"):
        ask(llm)
    assert len(client.requests) == 2, "it is reported, never parsed as if it were complete"


# ---- structured-output fallbacks -----------------------------------------------

def test_falls_back_when_the_endpoint_rejects_a_mode(settings):
    unsupported = api_error(openai.BadRequestError, 400, "response_format json_schema strict is not supported")
    llm, client, _, events = make(settings, [unsupported, unsupported, completion(GOOD)])
    assert ask(llm).city == "Pune"
    formats = [r.get("response_format", {}).get("type") for r in client.requests]
    assert formats == ["json_schema", "json_schema", "json_object"]
    in_prompt = ['"population"' in r["messages"][0]["content"] for r in client.requests]
    assert in_prompt == [False, True, True], "without strict enforcement the schema goes into the prompt"
    assert client.requests[0]["response_format"]["json_schema"].get("strict") is True
    assert "strict" not in client.requests[1]["response_format"]["json_schema"]
    assert any("json_object" in event for event in events)
    # the working mode is remembered for the next call
    client.script.append(completion(GOOD))
    ask(llm)
    assert client.requests[-1]["response_format"]["type"] == "json_object"


def test_provider_side_schema_failure_counts_as_a_bad_reply(settings):
    failed = api_error(openai.BadRequestError, 400, "Failed to validate JSON",
                       body={"error": {"code": "json_validate_failed", "failed_generation": "{oops"}})
    llm, client, _, _ = make(settings, [failed, completion(GOOD)])
    assert ask(llm).city == "Pune"
    assert client.requests[1]["response_format"]["json_schema"]["strict"] is True, "the mode is not downgraded"
    assert client.requests[1]["messages"][-2]["content"] == "{oops"
    assert llm.usage["extractor"]["failed_generations"] == 1


def test_unsupported_parameters_are_dropped(settings):
    cfg = replace(settings, reasoning_effort="low")
    errors = [api_error(openai.BadRequestError, 400, "unknown parameter: max_completion_tokens"),
              api_error(openai.BadRequestError, 400, "reasoning_effort is not supported by this model")]
    llm, client, _, _ = make(cfg, errors + [completion(GOOD)])
    assert ask(llm).city == "Pune"
    final = client.requests[-1]
    assert "max_tokens" in final and "max_completion_tokens" not in final and "reasoning_effort" not in final


def test_other_bad_requests_are_reported(settings):
    llm, _, _, _ = make(settings, [api_error(openai.BadRequestError, 400, "context length exceeded")])
    with pytest.raises(LLMConfigError, match="context length exceeded"):
        ask(llm)


# ---- rate limits and other errors ----------------------------------------------

def test_waits_as_long_as_the_provider_asks(settings):
    limited = api_error(openai.RateLimitError, 429, "slow down", headers={"retry-after": "7"})
    llm, _, waits, events = make(settings, [limited, completion(GOOD)])
    assert ask(llm).city == "Pune"
    assert waits == [7.5]
    assert "waiting 8s" in events[0] or "waiting 7s" in events[0]
    assert llm.usage["extractor"]["rate_limit_waits"] == 1


def test_wait_time_is_read_from_the_error_text():
    error = api_error(openai.RateLimitError, 429, "Rate limit reached. Please try again in 2m3.5s.")
    assert retry_after_seconds(error) == 123.5
    assert retry_after_seconds(api_error(openai.RateLimitError, 429, "try again in 850ms")) is None


def test_a_long_wait_is_reported_instead_of_hanging(settings):
    limited = api_error(openai.RateLimitError, 429, "daily limit", headers={"retry-after": "3600"})
    llm, client, waits, _ = make(settings, [limited])
    with pytest.raises(LLMRateLimitError, match="rate limit"):
        ask(llm)
    assert waits == [] and len(client.requests) == 1


def test_total_wait_is_capped(settings):
    cfg = replace(settings, max_rate_wait_s=20)
    limited = [api_error(openai.RateLimitError, 429, "slow", headers={"retry-after": "8"}) for _ in range(5)]
    llm, _, waits, _ = make(cfg, limited)
    with pytest.raises(LLMRateLimitError):
        ask(llm)
    assert waits == [8.5, 8.5], "it stops once the next wait would pass the cap"


@pytest.mark.parametrize("error,message", [
    (api_error(openai.AuthenticationError, 401, "invalid key"), "API key was rejected"),
    (api_error(openai.NotFoundError, 404, "no such model"), "was not found"),
])
def test_configuration_errors_are_explained(settings, error, message):
    llm, _, _, _ = make(settings, [error])
    with pytest.raises(LLMConfigError, match=message):
        ask(llm)


def test_server_errors_are_retried(settings):
    down = api_error(openai.InternalServerError, 503, "overloaded")
    llm, client, waits, _ = make(settings, [down, down, completion(GOOD)])
    assert ask(llm).city == "Pune" and len(client.requests) == 3 and waits == [2.0, 4.0]


def test_each_role_can_have_its_own_model():
    cfg = load_settings({"LLM_API_KEY": "k", "LLM_MODEL": "base", "LLM_MODEL_DRAFTER": "writer",
                         "LLM_MODEL_EXTRACTOR": "reader"}, use_dotenv=False)
    assert cfg.models == {"extractor": "reader", "planner": "base", "clarifier": "base", "drafter": "writer",
                          "reviewer": "base", "extractor_lists": "reader"}, "list passes follow the Extractor"
    cfg = load_settings({"LLM_API_KEY": "k", "LLM_MODEL_EXTRACTOR_LISTS": "lists"}, use_dotenv=False)
    assert cfg.models["extractor_lists"] == "lists" and cfg.models["extractor"] == "openai/gpt-oss-120b"
    llm, client, _, _ = make(cfg, [completion(GOOD)])
    llm.structured("extractor_lists", Answer, "s", "u")
    assert client.requests[0]["model"] == "lists"


def test_missing_key_is_explained():
    with pytest.raises(LLMConfigError, match="LLM_API_KEY"):
        build_llm(load_settings({}, use_dotenv=False))
    assert "not an AI model" in build_llm(load_settings({"LLM_BACKEND": "demo"}, use_dotenv=False)).description
