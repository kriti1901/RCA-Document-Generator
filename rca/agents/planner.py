"""Planner agent: judges whether the extracted facts are good enough and sets the focus per section."""

from __future__ import annotations

from rca.agents.common import DATA_NOT_INSTRUCTIONS, as_json, block, full_payload
from rca.llm import LLM
from rca.schema.facts import IncidentFacts
from rca.schema.field_specs import QUALITY_CHECK_IDS
from rca.schema.llm_io import PlanOut

SYSTEM = f"""You are planning a Root Cause Analysis (RCA) document from facts already extracted from incident notes.

{DATA_NOT_INSTRUCTIONS}

You have two jobs.

1. quality_gaps: flag at most 3 facts whose value is too vague to be useful in an RCA, or that contradicts another fact. Typical cases: a root cause that only restates the symptom ("the service was down"), an impact with no concrete effect, a fix described only as "fixed", a detection method that does not match the detection time in the timeline. Only these fields may be flagged: {", ".join(QUALITY_CHECK_IDS)}. For each, write `problem` in one sentence and a specific `question` to put to the person who wrote the notes. Do not flag facts that are missing; those are handled elsewhere. If the facts are specific enough, return an empty list.

2. section_notes: for each narrative section (summary, impact, detection, response, root_cause), one sentence on what it should emphasise for this incident, based only on the facts."""

MAX_QUALITY_GAPS = 3


def plan(llm: LLM, facts: IncidentFacts, durations: dict[str, str]) -> dict:
    """Returns {"quality_gaps": [...], "section_notes": {section: focus}}."""
    payload = full_payload(facts, durations, key="id")
    out = llm.structured("planner", PlanOut, SYSTEM, block("facts", as_json(payload)))
    gaps = []
    seen: set[str] = set()
    for gap in out.quality_gaps:
        # Only fields that actually have a value can be "too vague"; ignore the rest.
        if gap.field in seen or not facts.value(gap.field) or not gap.question.strip():
            continue
        seen.add(gap.field)
        gaps.append({"field": gap.field, "problem": gap.problem.strip(), "question": gap.question.strip()})
        if len(gaps) == MAX_QUALITY_GAPS:
            break
    notes = {note.section: note.focus.strip() for note in out.section_notes if note.focus.strip()}
    return {"quality_gaps": gaps, "section_notes": notes}
