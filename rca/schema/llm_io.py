"""Shapes of the JSON each agent asks the model for.

They are deliberately small and flat: open-source models follow short schemas
far more reliably than one large nested document schema. Every model forbids
extra keys so strict structured-output modes accept the generated JSON schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, create_model

from rca.schema.field_specs import LIST_IDS, QUALITY_CHECK_IDS, SCALAR_IDS


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---- Extractor ---------------------------------------------------------------

class ExtractedFact(_Strict):
    field: str
    value: str
    evidence: str


class FactsOut(_Strict):
    facts: list[ExtractedFact]


def facts_schema_for(field_ids: tuple[str, ...]) -> type[FactsOut]:
    """FactsOut with `field` restricted to the given ids, so the model cannot invent field names."""
    allowed = Literal.__getitem__(tuple(field_ids))  # type: ignore[attr-defined]
    fact = create_model("ExtractedFact", __base__=_Strict, field=(allowed, ...), value=(str, ...), evidence=(str, ...))
    return create_model("FactsOut", __base__=_Strict, facts=(list[fact], ...))  # type: ignore[valid-type]


class TimelineOutEvent(_Strict):
    time: str
    event: str
    source: str
    evidence: str


class TimelineOut(_Strict):
    events: list[TimelineOutEvent]


class ActionOut(_Strict):
    action: str
    type: Literal["corrective", "preventive", "detective", "unknown"]
    owner: str
    due_date: str
    priority: str
    ticket: str
    evidence: str


class ActionsOut(_Strict):
    actions: list[ActionOut]


# ---- Planner -----------------------------------------------------------------

class QualityGap(_Strict):
    field: Literal.__getitem__(QUALITY_CHECK_IDS)  # type: ignore[valid-type]
    problem: str
    question: str


class SectionNote(_Strict):
    section: Literal["summary", "impact", "detection", "response", "root_cause"]
    focus: str


class PlanOut(_Strict):
    quality_gaps: list[QualityGap]
    section_notes: list[SectionNote]


# ---- Clarifier ---------------------------------------------------------------

class ParsedAnswer(_Strict):
    number: int
    kind: Literal["answer", "decline", "unclear"]
    value: str
    value2: str
    evidence: str


class ReplyOut(_Strict):
    answers: list[ParsedAnswer]
    skip_rest: bool
    extra_notes: str


# ---- Drafter -----------------------------------------------------------------

class SectionDraft(_Strict):
    paragraphs: list[str]


class SummaryDraft(_Strict):
    title: str
    paragraphs: list[str]


# ---- Reviewer ----------------------------------------------------------------

class ReviewIssue(_Strict):
    section: Literal["summary", "impact", "detection", "response", "root_cause"]
    problem: str
    fix: str


class MissingInfo(_Strict):
    field: Literal.__getitem__(SCALAR_IDS + LIST_IDS)  # type: ignore[valid-type]
    question: str


class ReviewOut(_Strict):
    verdict: Literal["approve", "revise", "needs_info"]
    issues: list[ReviewIssue]
    missing_info: list[MissingInfo]
