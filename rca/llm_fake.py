"""Stand-ins for the model, used by the tests and by the offline demo mode.

`ScriptedLLM` answers each call from a function you supply. `demo_llm()` is a
ScriptedLLM with a small rule-based handler: it reads "Label: value" lines,
lines that start with a time, and "Action: ..." lines. It is not an AI model
and only does something sensible with notes written that way. It exists so
the whole workflow can be run and tested without a key or a network.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from pydantic import BaseModel

Handler = Callable[[str, type[BaseModel], str, str], "dict | BaseModel"]


class ScriptedLLM:
    def __init__(self, handler: Handler, description: str = "scripted test double") -> None:
        self.handler = handler
        self.description = description
        self.usage: dict[str, dict[str, int]] = {}
        self.on_event = None
        self.calls: list[tuple[str, str]] = []  # (agent, schema name)

    def structured(self, agent: str, schema: type[BaseModel], system: str, user: str):
        self.calls.append((agent, schema.__name__))
        stats = self.usage.setdefault(agent, {})
        stats["calls"] = stats.get("calls", 0) + 1
        result = self.handler(agent, schema, system, user)
        return result if isinstance(result, BaseModel) else schema.model_validate(result)


# ---- the rule-based demo handler ---------------------------------------------

_ALIASES = {
    "title": "title", "incident": "title", "subject": "title",
    "incident id": "incident_id", "id": "incident_id", "ticket": "incident_id",
    "severity": "severity", "sev": "severity", "priority": "severity",
    "date": "incident_date", "incident date": "incident_date",
    "incident commander": "incident_commander", "commander": "incident_commander", "ic": "incident_commander",
    "affected services": "affected_services", "services": "affected_services",
    "service": "affected_services", "systems": "affected_services",
    "customer impact": "customer_impact", "impact": "customer_impact", "user impact": "customer_impact",
    "scale": "impact_scale", "scale of impact": "impact_scale", "affected users": "impact_scale",
    "business impact": "business_impact", "sla": "business_impact", "sla impact": "business_impact",
    "data loss": "data_loss",
    "start": "started_at", "started": "started_at", "start time": "started_at",
    "detected": "detected_at", "detection time": "detected_at",
    "mitigated": "mitigated_at", "mitigation time": "mitigated_at",
    "resolved": "resolved_at", "end": "resolved_at", "resolution time": "resolved_at",
    "timezone": "timezone", "time zone": "timezone", "tz": "timezone",
    "detection": "detection_method", "detected by": "detection_method", "how detected": "detection_method",
    "monitoring gaps": "monitoring_gaps",
    "mitigation": "mitigation_steps", "mitigation steps": "mitigation_steps",
    "fix": "fix_applied", "fix applied": "fix_applied", "resolution": "fix_applied",
    "responders": "responders", "people": "responders",
    "trigger": "trigger",
    "root cause": "root_cause", "cause": "root_cause",
    "contributing factors": "contributing_factors", "contributing factor": "contributing_factors",
    "5 whys": "five_whys", "whys": "five_whys",
    "went well": "went_well", "what went well": "went_well",
    "went wrong": "went_wrong", "what went wrong": "went_wrong",
    "got lucky": "got_lucky", "lucky": "got_lucky",
}
_LIST_FIELDS = {"affected_services", "mitigation_steps", "responders", "contributing_factors",
                "five_whys", "went_well", "went_wrong", "got_lucky"}
_COMMA_LISTS = {"affected_services", "responders"}

_LABELLED = re.compile(r"^\s*(?:[-*]\s*)?([A-Za-z0-9][A-Za-z0-9 /]{0,38}?)\s*:\s+(\S.*)$")
_TIMED = re.compile(r"^\s*(?:[-*]\s*)?((?:\d{4}-\d{2}-\d{2}\s+)?\d{1,2}:\d{2})\s*(?:[-–—]\s*)?(\S.*)$")
_ACTION = re.compile(r"^\s*(?:[-*]\s*)?(?:action|todo|follow[- ]?up)\s*:\s*(\S.*)$", re.I)
_OWNER = re.compile(r"\bowner\s*[:=]?\s*([A-Za-z][\w .'-]*?)\s*(?=[,;)]|\bdue\b|$)", re.I)
_DUE = re.compile(r"\bdue\s*[:=]?\s*([^,;)]+)", re.I)
_DECLINE = re.compile(r"^(skip|pass|n/?a|idk|(i )?(do not|don'?t|dont) know|not sure|unknown|no idea)[\s.!]*$", re.I)
_NUMBERED = re.compile(r"(?:^|(?<=\s))(\d{1,2})\s*[.):]\s+")


def _block(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", text, re.S)
    return match.group(1) if match else ""


def _labelled(line: str) -> tuple[str, str] | None:
    match = _LABELLED.match(line)
    if not match:
        return None
    field_id = _ALIASES.get(match.group(1).strip().lower())
    return (field_id, match.group(2).strip()) if field_id else None


def _facts(schema: type[BaseModel], user: str) -> dict:
    allowed = set(schema.model_json_schema()["$defs"]["ExtractedFact"]["properties"]["field"]["enum"])
    facts = []
    for line in _block(user, "notes").splitlines():
        hit = _labelled(line)
        if not hit or hit[0] not in allowed:
            continue
        field_id, value = hit
        if field_id in _LIST_FIELDS:
            pattern = r"[;,]" if field_id in _COMMA_LISTS else r";"
            for item in (part.strip() for part in re.split(pattern, value)):
                if item:
                    facts.append({"field": field_id, "value": item, "evidence": item})
        else:
            facts.append({"field": field_id, "value": value, "evidence": line.strip()})
    return {"facts": facts}


def _timeline(user: str) -> dict:
    events = []
    for line in _block(user, "notes").splitlines():
        match = _TIMED.match(line)
        if match and not _ACTION.match(line):
            events.append({"time": match.group(1), "event": match.group(2).strip(), "source": "",
                           "evidence": line.strip().lstrip("-* ")})
    return {"events": events}


def _action(text: str, evidence: str) -> dict:
    owner = _OWNER.search(text)
    due = _DUE.search(text)
    action = re.split(r"\s*[(,;]\s*(?:owner|due)\b", text, maxsplit=1, flags=re.I)[0].strip(" .")
    return {"action": action, "type": "unknown", "owner": owner.group(1).strip() if owner else "",
            "due_date": due.group(1).strip() if due else "", "priority": "", "ticket": "",
            "evidence": evidence}


def _actions(user: str) -> dict:
    lines = [line for line in _block(user, "notes").splitlines() if line.strip()]
    actions = [_action(m.group(1), line.strip()) for line in lines if (m := _ACTION.match(line))]
    # A short free-text answer to "what actions were agreed?": one action per line.
    if not actions and len(lines) <= 4 and not any(_labelled(line) or _TIMED.match(line) for line in lines):
        actions = [_action(line.strip(" -*"), line.strip()) for line in lines]
    return {"actions": actions}


def _plan(user: str) -> dict:
    facts = json.loads(_block(user, "facts") or "{}")
    gaps = []
    root_cause = facts.get("root_cause", "")
    if root_cause and len(root_cause.split()) <= 3:
        gaps.append({"field": "root_cause", "problem": "The root cause is stated in only a few words.",
                     "question": "What exactly went wrong, and why was it not caught earlier?"})
    notes = [{"section": name, "focus": "State the facts plainly and in order."}
             for name in ("summary", "impact", "detection", "response", "root_cause")]
    return {"quality_gaps": gaps, "section_notes": notes}


def _reply(user: str) -> dict:
    questions = [q for q in _block(user, "questions").splitlines() if q.strip()]
    reply = _block(user, "reply").strip()
    marks = list(_NUMBERED.finditer(reply))
    segments: list[tuple[int, str]] = []
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(reply)
        segments.append((int(mark.group(1)), reply[mark.end():end].strip()))
    if not segments:
        if len(questions) != 1:
            return {"answers": [], "skip_rest": False, "extra_notes": reply}
        segments = [(1, reply)]
    answers = []
    for number, text in segments:
        if not 1 <= number <= len(questions) or not text:
            continue
        question = questions[number - 1]
        if _DECLINE.match(text):
            answers.append({"number": number, "kind": "decline", "value": "", "value2": "", "evidence": text})
        elif "[owner and due date]" in question:
            owner, _, due = re.split(r"(?:,|;)?\s*\b(by|due|before)\b\s*:?\s*", text, maxsplit=1, flags=re.I) \
                if re.search(r"\b(by|due|before)\b", text, re.I) else (text, "", "")
            answers.append({"number": number, "kind": "answer", "value": owner.strip(" ,;."),
                            "value2": due.strip(" ,;."), "evidence": text})
        elif "[list]" in question:
            for item in (part.strip() for part in re.split(r";", text)):
                if item:
                    answers.append({"number": number, "kind": "answer", "value": item, "value2": "",
                                    "evidence": item})
        else:
            answers.append({"number": number, "kind": "answer", "value": text, "value2": "", "evidence": text})
    return {"answers": answers, "skip_rest": False, "extra_notes": ""}


def _sentences(facts: dict) -> str:
    parts = []
    for label, value in facts.items():
        if isinstance(value, list):
            value = "; ".join(str(v) if not isinstance(v, dict) else v.get("action", "") for v in value)
        parts.append(f"{label}: {str(value).rstrip('.')}.")
    return " ".join(parts)


def _section(user: str) -> dict:
    facts = json.loads(_block(user, "facts") or "{}")
    return {"paragraphs": [_sentences(facts)] if facts else []}


def _summary(user: str) -> dict:
    facts = json.loads(_block(user, "facts") or "{}")
    services = facts.get("Affected services")
    title = facts.get("Incident title") or (f"Incident affecting {', '.join(services)}" if services else "Incident report")
    return {"title": title, "paragraphs": [_sentences(facts)] if facts else []}


def demo_handler(agent: str, schema: type[BaseModel], system: str, user: str) -> dict:
    name = schema.__name__
    if name == "FactsOut":
        return _facts(schema, user)
    if name == "TimelineOut":
        return _timeline(user)
    if name == "ActionsOut":
        return _actions(user)
    if name == "PlanOut":
        return _plan(user)
    if name == "ReplyOut":
        return _reply(user)
    if name == "SectionDraft":
        return _section(user)
    if name == "SummaryDraft":
        return _summary(user)
    if name == "ReviewOut":
        return {"verdict": "approve", "issues": [], "missing_info": []}
    raise ValueError(f"demo handler has no rule for {name}")


def demo_llm() -> ScriptedLLM:
    return ScriptedLLM(demo_handler, description="offline demo (rule-based stand-in, not an AI model)")
