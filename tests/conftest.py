"""Shared fixtures. Every test runs against a temporary data directory and a fixed clock."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from rca.config import load_settings
from rca.llm_fake import ScriptedLLM, demo_handler
from rca.service import RCAService

FIXTURES = Path(__file__).parent / "fixtures"
NOW = datetime(2026, 11, 11, 10, 30)

# Answers that complete the demo notes: two rounds of five, then the last two questions.
ROUND_1 = ("1. Sev2\n2. IST\n3. A typo cut the DB pool size to 5 and config CI has no check for it\n"
           "4. skip\n5. Ravi Kumar")
ROUND_2 = ("1. INC-4821\n2. none\n3. Ravi Kumar, by 2026-11-25\n4. skip\n"
           "5. The pool size was too small; The config had a typo")


@pytest.fixture
def demo_notes() -> str:
    return (FIXTURES / "demo_notes.txt").read_text()


@pytest.fixture
def settings(tmp_path):
    return load_settings({"LLM_BACKEND": "demo", "RCA_DATA_DIR": str(tmp_path / "data")}, use_dotenv=False)


@pytest.fixture
def make_service(settings):
    """Build a service around any handler; defaults to the rule-based demo handler."""

    def factory(handler=demo_handler, publish=None, **overrides):
        cfg = settings
        if overrides:
            from dataclasses import replace

            cfg = replace(settings, **overrides)
        llm = ScriptedLLM(handler)
        return RCAService(cfg, llm=llm, publish=publish, now=lambda: NOW)

    return factory


@pytest.fixture
def service(make_service):
    return make_service()


def to_approval(service: RCAService, notes: str, user: str = "Test User"):
    """Drive a session from notes to the approval step with the standard answers."""
    view = service.start(notes, user)
    for answer in (ROUND_1, ROUND_2, "skip all"):
        if not (view.pending and view.pending["kind"] == "questions"):
            break
        view = service.reply(view.session_id, answer)
    assert view.pending and view.pending["kind"] == "approval", view.error or view.status
    return view


def pdf_text(path: str) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() for page in PdfReader(path).pages)
