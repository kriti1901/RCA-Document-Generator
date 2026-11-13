"""Builds the view the PDF template (and the Confluence page) is filled from.

Everything here is deterministic: the layout, the "Not provided by the user"
markers, the durations and the gaps table all come from code, never from the
model. The model only supplies the narrative paragraphs.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from rca.schema.facts import FieldValue, IncidentFacts, ListValue
from rca.schema.field_specs import BY_ID, MILESTONE_IDS, fields_in_section, is_required
from rca.gaps import declined_gaps, incident_date_of, section_title
from rca.timeutil import DURATION_LABELS, Sequence, compute_milestones

NOT_PROVIDED = "Not provided by the user"
ORDERED_LISTS = ("five_whys", "mitigation_steps")  # shown as numbered steps


def format_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y").lstrip("0")
    except (ValueError, TypeError):
        return iso or ""


def not_provided(by: str, at: str) -> str:
    """The marker the requirements call for: who was asked, and when."""
    who = ", ".join(part for part in (by, format_date(at)) if part)
    return f"{NOT_PROVIDED} ({who})" if who else NOT_PROVIDED


def scalar_cell(fv: FieldValue) -> dict:
    if fv.provided:
        return {"state": "ok", "text": fv.value}
    if fv.status == "declined":
        return {"state": "declined", "text": not_provided(fv.by, fv.at)}
    if fv.status == "not_applicable":
        return {"state": "na", "text": "Not applicable"}
    return {"state": "missing", "text": "Not recorded"}


def list_cell(lv: ListValue) -> dict:
    if lv.provided:
        return {"state": "ok", "items": [item.value for item in lv.items]}
    if lv.status == "declined":
        return {"state": "declined", "text": not_provided(lv.by, lv.at), "items": []}
    return {"state": "missing", "text": "Not recorded", "items": []}


def _rows(facts: IncidentFacts, section: str, severity: str,
          skip: tuple[str, ...] = ("incident_date",)) -> list[dict]:
    """Label/value rows for a section: every required field, plus optional ones that have content."""
    rows = []
    for spec in fields_in_section(section):
        if spec.id in skip:
            continue
        if spec.kind == "scalar":
            fv = facts.scalars[spec.id]
            cell = scalar_cell(fv)
            shown = fv.status != "missing" or is_required(spec, severity)
        else:
            lv = facts.lists[spec.id]
            cell = list_cell(lv)
            shown = lv.status != "missing" or is_required(spec, severity)
        if shown:
            rows.append({"id": spec.id, "label": spec.label, "kind": spec.kind,
                         "ordered": spec.id in ORDERED_LISTS, **cell})
    return rows


def _timeline(facts: IncidentFacts, year: int) -> dict:
    values = {key: facts.value(key) for key in MILESTONE_IDS}
    tz_field = facts.scalars["timezone"]
    base_date = facts.value("incident_date")
    milestones = compute_milestones(values, facts.value("timezone"), default_year=year, base_date=base_date)
    rows = []
    for key in MILESTONE_IDS:
        fv = facts.scalars[key]
        point = milestones.points[key]
        cell = scalar_cell(fv)
        rows.append({"label": BY_ID[key].label, "state": cell["state"],
                     "local": point.local_text if fv.provided else cell["text"],
                     "utc": point.utc_text if fv.provided else ""})

    # Detailed events share the incident's date unless they state their own.
    sequence = Sequence(milestones.zone, year)
    sequence.seed([base_date, values.get("started_at", "")] + [event.time for event in facts.timeline])
    start = milestones.points["started_at"].local
    end = milestones.points["resolved_at"].local
    events = []
    for order, event in enumerate(facts.timeline):
        point = sequence.add(event.time, advance=False)
        local = point.local
        # A clock time earlier than the start, in an incident that ran past midnight, is the next day.
        if (local is not None and start is not None and end is not None and end.date() > start.date()
                and local.date() == start.date() and local.time() < start.time() and local.time() <= end.time()):
            local += timedelta(days=1)
            point.local = local
            if point.utc is not None:
                point.utc += timedelta(days=1)
        events.append({"time": point.local_text, "utc": point.utc_text, "event": event.event,
                       "source": event.source, "sort": (local or datetime.max, order)})
    if all(e["sort"][0] != datetime.max for e in events):
        events.sort(key=lambda e: e["sort"])

    if tz_field.provided:
        zone = milestones.zone
        caption = (f"Times were recorded in {zone.label}." if zone.tz is not None
                   else f'Times are as recorded ("{tz_field.value}"); this timezone could not be converted to UTC.')
        tz_state = "ok"
    elif tz_field.status == "declined":
        caption, tz_state = f"Timezone of the recorded times: {not_provided(tz_field.by, tz_field.at)}.", "declined"
    else:
        caption, tz_state = "Timezone of the recorded times: not recorded.", "missing"
    return {
        "rows": rows, "events": events, "caption": caption, "tz_state": tz_state,
        "show_utc": milestones.converted or any(e["utc"] for e in events),
        "durations": [{"label": DURATION_LABELS[name], "text": text} for name, text in milestones.durations.items()],
        "duration_map": milestones.durations,
    }


def _actions(facts: IncidentFacts) -> dict:
    rows = []
    for number, action in enumerate(facts.actions, start=1):
        if action.owner:
            owner = {"state": "ok", "text": action.owner}
        elif action.owner_status == "declined":
            owner = {"state": "declined", "text": not_provided(action.owner_by, action.owner_at)}
        else:
            owner = {"state": "missing", "text": "Not recorded"}
        rows.append({"number": number, "action": action.action, "type": action.type.capitalize() or "—",
                     "owner": owner, "due": action.due_date or "—", "ticket": action.ticket or "—"})
    if rows:
        return {"state": "ok", "rows": rows}
    if facts.actions_status == "declined":
        return {"state": "declined", "rows": [], "text": not_provided(facts.actions_by, facts.actions_at)}
    return {"state": "missing", "rows": [], "text": "No follow-up actions were recorded."}


def build_view(state: dict, *, now: datetime, final: bool, model_description: str = "") -> dict[str, Any]:
    facts = IncidentFacts.load(state.get("facts"))
    user = state.get("user") or {}
    drafts = state.get("drafts") or {}
    severity = facts.severity
    version = int(state.get("version", 1))
    approver = state.get("approved_by") or user.get("name", "")
    timeline = _timeline(facts, now.year)

    title = facts.value("title") or "Incident report"
    incident_id = facts.scalars["incident_id"]
    incident_date, _ = incident_date_of(facts)  # as stated: with the year only if someone gave it
    date_cell = ({"state": "ok", "text": incident_date} if incident_date
                 else scalar_cell(facts.scalars["incident_date"]))

    header_rows = [
        {"label": "Incident ID", **scalar_cell(incident_id)},
        {"label": "Severity", **scalar_cell(facts.scalars["severity"])},
        {"label": "Incident date", **date_cell},
        {"label": "Incident commander", **scalar_cell(facts.scalars["incident_commander"])},
        {"label": "Affected services", **_join(list_cell(facts.lists["affected_services"]))},
        {"label": "Document status", "state": "ok",
         "text": "Final" if final else "Draft, awaiting approval"},
        {"label": "Version", "state": "ok", "text": str(version)},
        {"label": "Prepared by", "state": "ok", "text": user.get("name", "")},
        {"label": "Date prepared", "state": "ok", "text": now.strftime("%d %b %Y").lstrip("0")},
    ]

    impact_rows = _rows(facts, "impact", severity)
    if "time_to_resolve" in timeline["duration_map"]:
        impact_rows.append({"id": "duration", "label": "Duration", "kind": "scalar", "state": "ok",
                            "text": timeline["duration_map"]["time_to_resolve"]})

    lessons = _rows(facts, "lessons", severity)
    gaps = [{"label": gap["label"], "section": section_title(gap["section"]), "by": gap["by"],
             "date": format_date(gap["at"]), "note": gap.get("note", "")}
            for gap in declined_gaps(facts, state.get("clarifications_declined", []))]

    related = [{"title": item.get("title") or "Untitled incident", "incident_id": item.get("incident_id", ""),
                "severity": item.get("severity", ""), "started_at": item.get("started_at", ""),
                "root_cause": item.get("root_cause", ""), "url": item.get("confluence_url", "")}
               for item in state.get("similar", [])]

    return {
        "title": title,
        "final": final,
        "severity": severity,
        "incident_ref": incident_id.value if incident_id.provided else "",
        "version": version,
        "header_rows": header_rows,
        "summary": drafts.get("summary", []),
        "impact": {"rows": impact_rows, "paragraphs": drafts.get("impact", [])},
        "timeline": timeline,
        "detection": {"rows": _rows(facts, "detection", severity), "paragraphs": drafts.get("detection", [])},
        "response": {"rows": _rows(facts, "response", severity), "paragraphs": drafts.get("response", [])},
        "root_cause": {"rows": _rows(facts, "root_cause", severity), "paragraphs": drafts.get("root_cause", [])},
        "lessons": lessons,
        "actions": _actions(facts),
        "related": related,
        "gaps": gaps,
        "revision": {"version": version, "date": now.strftime("%d %b %Y").lstrip("0"),
                     "author": user.get("name", ""),
                     "status": f"Approved by {approver}" if final else "Draft, awaiting approval"},
        "model_description": model_description,
        "author": user.get("name", ""),
        "approver": approver,
        "generated_at": now.strftime("%d %b %Y %H:%M").lstrip("0"),
        "not_provided": NOT_PROVIDED,
    }


def _join(cell: dict) -> dict:
    """Show a list field on one line (used in the header table)."""
    if cell["state"] == "ok":
        return {"state": "ok", "text": ", ".join(cell["items"])}
    return {"state": cell["state"], "text": cell["text"]}
