"""Gap detection: compare the facts with the RCA layout and list what still has to be asked.

This is plain code, not a model call, so the same facts always produce the
same questions and nothing required can be silently skipped.
"""

from __future__ import annotations

from rca.schema.facts import IncidentFacts
from rca.schema.field_specs import (ACTION_OWNER_PRIORITY, ACTIONS_GAP_PRIORITY, ACTIONS_QUESTION,
                                    BY_ID, FIELDS, SECTION_TITLES, is_required)
from rca.timeutil import describe_date

MAX_ASKS_PER_GAP = 2  # a question is repeated once if the answer was unclear, then closed


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


YEAR_GAP_ID = "quality:incident_date"


def incident_date_of(facts: IncidentFacts) -> tuple[str, bool]:
    """The incident date as stated ('3 Nov 2026' or '5 Nov') and whether it includes the year."""
    best = ("", False)
    for field_id in ("incident_date", "started_at"):
        label, has_year = describe_date(facts.value(field_id))
        if label and has_year:
            return label, True
        if label and not best[0]:
            best = (label, False)
    return best


def field_gap(field_id: str) -> dict:
    spec = BY_ID[field_id]
    return {
        "id": f"field:{field_id}", "kind": "field", "field": field_id, "index": -1,
        "label": spec.label, "section": spec.section, "question": spec.ask,
        "answer_type": spec.kind, "priority": spec.priority,
    }


def extra_gap(kind: str, field_id: str, question: str, problem: str = "", current: str = "",
              label: str = "") -> dict:
    """A gap raised by the Planner ("quality") or the Reviewer ("review") about an existing field."""
    spec = BY_ID[field_id]
    if kind == "quality" and current:
        question = f'About "{spec.label}" (the notes say: "{_short(current)}"): {question}'
    return {
        "id": f"{kind}:{field_id}", "kind": kind, "field": field_id, "index": -1,
        "label": label or f"{spec.label} (more detail)", "section": spec.section, "question": question,
        "answer_type": spec.kind, "priority": spec.priority, "problem": problem,
    }


def open_gaps(facts: IncidentFacts, extra_gaps: list[dict], gap_log: dict[str, dict]) -> list[dict]:
    """Everything that still needs an answer, most important first."""
    severity = facts.severity
    gaps: list[dict] = []

    # The incident date can come from its own field or from the start time. A date
    # without a year gets one follow-up question; a year is never assumed.
    known_date, has_year = incident_date_of(facts)
    for spec in FIELDS:
        if not is_required(spec, severity):
            continue
        status = facts.scalars[spec.id].status if spec.kind == "scalar" else facts.lists[spec.id].status
        if spec.id == "incident_date" and known_date:
            continue
        if status == "missing":
            gaps.append(field_gap(spec.id))
    if known_date and not has_year and gap_log.get(YEAR_GAP_ID, {}).get("outcome", "open") == "open":
        gaps.append(extra_gap("quality", "incident_date",
                              "The year is not stated. What was the full date (day, month and year)?",
                              problem="The year of the incident is not stated.", current=known_date,
                              label="Incident date (year)"))

    if not facts.actions and facts.actions_status == "missing":
        gaps.append({
            "id": "actions", "kind": "actions", "field": "", "index": -1,
            "label": "Follow-up actions", "section": "actions", "question": ACTIONS_QUESTION,
            "answer_type": "actions", "priority": ACTIONS_GAP_PRIORITY,
        })
    for index, action in enumerate(facts.actions):
        if not action.owner and action.owner_status == "missing":
            gaps.append({
                "id": f"owner:{index}", "kind": "owner", "field": "", "index": index,
                "label": f"Owner of action: {_short(action.action, 60)}", "section": "actions",
                "question": f'Who owns the action "{_short(action.action)}", and what is the due date?',
                "answer_type": "owner", "priority": ACTION_OWNER_PRIORITY,
            })

    for gap in extra_gaps:
        if gap_log.get(gap["id"], {}).get("outcome", "open") == "open":
            gaps.append(dict(gap))

    for gap in gaps:
        gap["asked"] = gap_log.get(gap["id"], {}).get("asked", 0)
    gaps.sort(key=lambda g: (g["priority"], g["id"]))
    return gaps


def declined_gaps(facts: IncidentFacts, clarifications_declined: list[dict]) -> list[dict]:
    """Everything the user was asked for and did not provide, for the gaps section and long-term memory."""
    gaps: list[dict] = []
    for fid, fv in facts.scalars.items():
        if fv.status == "declined":
            gaps.append({**field_gap(fid), "by": fv.by, "at": fv.at, "note": fv.note})
    for fid, lv in facts.lists.items():
        if lv.status == "declined":
            gaps.append({**field_gap(fid), "by": lv.by, "at": lv.at, "note": lv.note})
    if facts.actions_status == "declined":
        gaps.append({"id": "actions", "label": "Follow-up actions", "section": "actions",
                     "by": facts.actions_by, "at": facts.actions_at, "note": ""})
    for index, action in enumerate(facts.actions):
        if action.owner_status == "declined":
            gaps.append({"id": f"owner:{index}", "label": f"Owner of action: {action.action}",
                         "section": "actions", "by": action.owner_by, "at": action.owner_at, "note": ""})
    for item in clarifications_declined:
        gaps.append({"id": item["id"], "label": item["label"], "section": item["section"],
                     "by": item["by"], "at": item["at"], "note": item.get("note", "")})
    return gaps


def select_batch(gaps: list[dict], round_index: int, per_round: int, max_rounds: int) -> list[dict]:
    """The questions for this round: a few at a time, then everything left in the final round.

    Asking everything that remains in the last round guarantees that no
    required detail is marked "not provided" without the user having been asked.
    """
    if round_index >= max_rounds - 1:
        return gaps
    return gaps[:per_round]


def section_title(section: str) -> str:
    return SECTION_TITLES.get(section, section.replace("_", " ").title())
