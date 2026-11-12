"""The session state that flows through the workflow and is checkpointed after every step."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class RCAState(TypedDict, total=False):
    # identity
    session_id: str
    incident_key: str  # stable key in long-term memory; reused when an incident is reopened
    version: int
    user: dict  # {"name": ..., "email": ...}

    # inputs and facts
    raw_inputs: list[dict]  # {"text", "source", "origin", "parts", "override", "extracted", "added_at"}
    facts: dict  # IncidentFacts.model_dump()
    rejected: list[dict]  # extracted items dropped by the evidence check

    # planning and the question loop
    plan: dict  # {"section_notes": {...}, "done": bool}
    similar: list[dict]  # similar past incidents from long-term memory
    extra_gaps: list[dict]  # questions raised by the Planner or Reviewer about existing fields
    gap_log: dict  # gap id -> {"asked": n, "outcome": "open" | "answered" | "declined"}
    clarifications_declined: list[dict]
    asked: list[dict]  # the batch of questions awaiting an answer
    reply: str  # the user's answer, before it has been read
    reply_by: str  # who gave that answer
    clarify_rounds: int
    review_round: bool  # the pending questions were raised by the Reviewer

    # drafting and review
    drafts: dict  # section -> [paragraphs]
    dirty: list[str]  # sections that must be redrafted
    feedback: str  # the user's change request, until it has been applied
    review: dict  # {"verdict", "issues", "missing_info"}
    review_action: str  # "revise" | "ask" | "render"
    revisions: int
    review_info_used: bool

    # output
    pdf_path: str
    decision: dict  # {"action": "approve" | "changes" | "cancel", "publish": bool, "feedback": str, "by": str}
    approved_by: str
    confluence: dict  # {"page_id", "url", "title", "version", "error"}
    status: str

    # chat transcript shown in the UI; nodes return only the new messages
    messages: Annotated[list[dict], operator.add]
