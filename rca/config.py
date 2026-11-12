"""Settings, read from environment variables (and a .env file if one exists)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

AGENTS = ("extractor", "planner", "clarifier", "drafter", "reviewer")
# The Extractor's timeline and action-item passes can use a different model from its
# fact passes. On a provider with a per-model rate limit that halves the first wait.
MODEL_ROLES = AGENTS + ("extractor_lists",)

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-120b"

# Order in which structured-output modes are tried when LLM_JSON_MODE=auto.
JSON_MODES = ("json_schema_strict", "json_schema", "json_object", "none")


@dataclass(frozen=True)
class ConfluenceSettings:
    base_url: str  # https://<site>.atlassian.net/wiki
    email: str
    api_token: str
    space_key: str
    parent_path: str = ""  # "Knowledge Base/RCAs"; empty means the space root
    create_missing_parents: bool = False


@dataclass(frozen=True)
class Settings:
    llm_backend: str  # "openai" (any OpenAI-compatible endpoint) or "demo"
    llm_base_url: str
    llm_api_key: str
    models: dict[str, str]  # role (see MODEL_ROLES) -> model id
    temperature: float | None
    reasoning_effort: str | None
    json_mode: str  # "auto" or one of JSON_MODES
    max_output_tokens: int  # first limit for a reply; doubled once if a reply is cut off
    max_tokens_param: str  # "max_completion_tokens" or "max_tokens"
    max_repairs: int
    request_timeout_s: float
    max_rate_wait_s: float
    data_dir: Path
    chunk_chars: int
    questions_per_round: int
    max_clarify_rounds: int
    max_revisions: int
    confluence: ConfluenceSettings | None

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rca.db"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "output"

    @property
    def llm_ready(self) -> bool:
        return self.llm_backend == "demo" or bool(self.llm_api_key)


def _flag(value: str | None, default: bool = False) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def normalize_confluence_url(url: str) -> str:
    """Confluence Cloud serves its REST API under /wiki; accept the site URL with or without it."""
    url = url.strip().rstrip("/")
    if url and not url.endswith("/wiki"):
        url += "/wiki"
    return url


def load_settings(env: Mapping[str, str] | None = None, *, use_dotenv: bool = True) -> Settings:
    if env is None:
        if use_dotenv:
            load_dotenv()
        env = os.environ

    def get(name: str, default: str = "") -> str:
        return (env.get(name) or default).strip()

    default_model = get("LLM_MODEL", DEFAULT_MODEL)
    models = {agent: get(f"LLM_MODEL_{agent.upper()}", default_model) for agent in AGENTS}
    models["extractor_lists"] = get("LLM_MODEL_EXTRACTOR_LISTS", models["extractor"])

    temperature_raw = get("LLM_TEMPERATURE", "0.1")
    temperature = None if temperature_raw.lower() in {"", "none", "default"} else float(temperature_raw)

    confluence = None
    conf_url, conf_token = get("CONFLUENCE_BASE_URL"), get("CONFLUENCE_API_TOKEN")
    conf_space = get("CONFLUENCE_SPACE_KEY")
    if conf_url and conf_token and conf_space and "your-site" not in conf_url:
        confluence = ConfluenceSettings(
            base_url=normalize_confluence_url(conf_url),
            email=get("CONFLUENCE_EMAIL"),
            api_token=conf_token,
            space_key=conf_space,
            parent_path=get("CONFLUENCE_PARENT_PATH").strip("/"),
            create_missing_parents=_flag(get("CONFLUENCE_CREATE_MISSING_PARENTS")),
        )

    api_key = get("LLM_API_KEY")
    if api_key in {"your-groq-key", "changeme"}:
        api_key = ""

    return Settings(
        llm_backend=get("LLM_BACKEND", "openai").lower(),
        llm_base_url=get("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
        llm_api_key=api_key,
        models=models,
        temperature=temperature,
        reasoning_effort=get("LLM_REASONING_EFFORT") or None,
        json_mode=get("LLM_JSON_MODE", "auto").lower(),
        max_output_tokens=int(get("LLM_MAX_OUTPUT_TOKENS", "4000")),
        max_tokens_param=get("LLM_MAX_TOKENS_PARAM", "max_completion_tokens"),
        max_repairs=int(get("LLM_MAX_REPAIRS", "2")),
        request_timeout_s=float(get("LLM_TIMEOUT_S", "120")),
        max_rate_wait_s=float(get("LLM_MAX_RATE_WAIT_S", "300")),
        data_dir=Path(get("RCA_DATA_DIR", "data")).expanduser().resolve(),
        chunk_chars=int(get("RCA_CHUNK_CHARS", "8000")),
        questions_per_round=int(get("RCA_QUESTIONS_PER_ROUND", "5")),
        max_clarify_rounds=int(get("RCA_MAX_CLARIFY_ROUNDS", "3")),
        max_revisions=int(get("RCA_MAX_REVISIONS", "2")),
        confluence=confluence,
    )
