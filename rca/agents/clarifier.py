"""Clarifier agent: asks the user for what is missing and reads the reply.

Composing the questions is deterministic (they come from the field registry).
Reading a free-text reply is the part that needs a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rca.agents.common import DATA_NOT_INSTRUCTIONS, block
from rca.gaps import MAX_ASKS_PER_GAP
from rca.llm import LLM
from rca.schema.facts import FieldValue, IncidentFacts, ListItem, ListValue
from rca.schema.field_specs import BY_ID, normalize_severity
from rca.schema.llm_io import ParsedAnswer, ReplyOut
from rca.security.evidence import mentions, normalize, supported
from rca.timeutil import clean_time_value

SYSTEM = f"""You read a user's reply to numbered questions about an incident and sort it into answers.

{DATA_NOT_INSTRUCTIONS}

For each question the reply addresses, return one entry:
- number: the question number.
- kind: "answer" if the user gave the information; "decline" if they do not know it, do not have it, never measured it, will find out later, will not give it, or told you to skip it; "unclear" if they addressed it but you cannot tell what the answer is. "Don't know, nobody measured it" is a decline, not an answer.
- value: only the information asked for, using the user's own words and dropping filler. For "Who was the incident commander?" answered with "Arjun Mehta was IC", the value is "Arjun Mehta". "" for decline or unclear.
- value2: only for questions marked [owner and due date]: put the owner in `value` and the due date here ("" if none was given). For every other question, "".
- evidence: the part of the reply the entry comes from, copied word for word.

For a question marked [list], return one entry per item, all with the same number.
Leave out questions the reply does not address at all.
skip_rest: true only if the user says to skip all remaining questions or to go ahead and generate the document now.
extra_notes: any other information about the incident in the reply that answers none of the questions, copied word for word. Otherwise "".
Never add information the user did not write."""

_DECLINE = re.compile(
    r"^(skip( (it|this|that|these|them))?|pass|n/?a|idk|i (do not|don'?t) know|(do not|don'?t|dont) know"
    r"|not sure|unknown|no idea|not available|(do not|don'?t) have (it|that|this))[\s.!]*$", re.I)
_SKIP_REST = re.compile(
    r"^(skip (all|everything|the rest|the remaining|remaining)( of (them|these))?"
    r"|(just )?(generate|draft|create|write)( it| the (rca|document|doc|pdf))?( now)?"
    r"|go ahead|proceed|continue( anyway)?|that'?s all( i have)?|no more)[\s.!]*$", re.I)

# An "answer" that only says the information is not available is a decline, whatever the model
# called it. "none" and "no" are deliberately not here: they are real answers to yes/no questions.
_NON_ANSWER = re.compile(
    r"^(sorry[, ]+)?((i|we|they|honestly|really)[, ]+)*("
    r"(do not|don'?t|dont|did not|didn'?t) (really )?(know|have|remember|recall|measure|track)"
    r"|no idea|not sure|unsure|unknown|not known|not available|unavailable|n/?a\b|tbd|tbc"
    r"|not (measured|tracked|recorded|decided|assigned)( yet)?$"
    r"|(nobody|no one|no-one) (knows|measured|tracked|recorded)"
    r"|can(not|'?t| not) (say|share|tell|disclose|remember|recall)|prefer not|rather not"
    r"|((i|we)\s*)?(will|'ll|need to|have to) (confirm|check|find out|ask|get back)"
    r"|to be (confirmed|determined|decided)|skip|pass\b)", re.I)


def is_non_answer(text: str) -> bool:
    return bool(_NON_ANSWER.match(text.strip().strip("\"'.,;:!-– ")))


_TAGS = {"list": " [list]", "owner": " [owner and due date]", "actions": " [list]", "scalar": ""}


def format_questions(batch: list[dict], *, round_index: int, final: bool, intro: str = "") -> str:
    if intro:
        pass
    elif round_index == 0:
        intro = "To complete the RCA I need a few more details."
    elif final:
        intro = "Last questions:"
    else:
        intro = "Thanks. A few more:"
    lines = [intro, ""]
    lines += [f"{n}. {gap['question']}" for n, gap in enumerate(batch, start=1)]
    lines += ["", 'Answer by number. Say "skip" for anything you do not have, '
                  'or "skip all" to go ahead with what we have.']
    return "\n".join(lines)


def parse_reply(llm: LLM, batch: list[dict], reply: str) -> ReplyOut:
    """Sort the reply into per-question answers. Obvious cases are handled without a model call."""
    text = reply.strip()
    numbers = range(1, len(batch) + 1)
    if _SKIP_REST.match(text):
        return ReplyOut(answers=[], skip_rest=True, extra_notes="")
    if _DECLINE.match(text):
        return ReplyOut(
            answers=[ParsedAnswer(number=n, kind="decline", value="", value2="", evidence=text) for n in numbers],
            skip_rest=False, extra_notes="")
    if len(batch) == 1 and batch[0]["answer_type"] == "scalar" and not re.match(r"^\s*1\s*[.):]", text) \
            and len(text) <= 60 and "\n" not in text:
        # one short reply to one question needs no interpretation
        kind = "decline" if is_non_answer(text) else "answer"
        return ReplyOut(
            answers=[ParsedAnswer(number=1, kind=kind, value="" if kind == "decline" else text,
                                  value2="", evidence=text)],
            skip_rest=False, extra_notes="")
    questions = "\n".join(f"{n}.{_TAGS.get(gap['answer_type'], '')} {gap['question']}"
                          for n, gap in enumerate(batch, start=1))
    return llm.structured("clarifier", ReplyOut, SYSTEM,
                          block("questions", questions) + "\n" + block("reply", text))


@dataclass
class Applied:
    answered: list[str] = field(default_factory=list)  # gap labels
    declined: list[str] = field(default_factory=list)
    unclear: list[str] = field(default_factory=list)
    new_inputs: list[dict] = field(default_factory=list)  # text to run back through the Extractor
    clarifications_declined: list[dict] = field(default_factory=list)


def _split_items(values: list[str]) -> list[str]:
    items: list[str] = []
    for value in values:
        for part in re.split(r"[\n;]+", value):
            part = part.strip(" \t-•*")
            if part and normalize(part) not in {normalize(i) for i in items}:
                items.append(part)
    return items


def _decline_gap(facts: IncidentFacts, gap: dict, gap_log: dict, applied: Applied,
                 user: str, now: str, note: str = "") -> None:
    kind = gap["kind"]
    if kind == "field":
        if gap["answer_type"] == "list":
            facts.lists[gap["field"]] = ListValue(status="declined", by=user, at=now, note=note)
        else:
            facts.scalars[gap["field"]] = FieldValue(status="declined", by=user, at=now, note=note)
    elif kind == "actions":
        facts.actions_status, facts.actions_by, facts.actions_at = "declined", user, now
    elif kind == "owner":
        action = facts.actions[gap["index"]]
        action.owner_status, action.owner_by, action.owner_at = "declined", user, now
    else:  # quality / review: the field keeps the value it has; record that no more detail came
        applied.clarifications_declined.append({
            "id": gap["id"], "label": gap["label"], "section": gap["section"],
            "question": gap["question"], "by": user, "at": now, "note": note,
        })
    gap_log.setdefault(gap["id"], {"asked": 0})["outcome"] = "declined"
    applied.declined.append(gap["label"])


def decline_all(facts: IncidentFacts, gaps: list[dict], gap_log: dict, applied: Applied,
                user: str, now: str, note: str) -> None:
    for gap in gaps:
        _decline_gap(facts, gap, gap_log, applied, user, now, note)


def apply_reply(facts: IncidentFacts, batch: list[dict], parsed: ReplyOut, reply: str,
                gap_log: dict, user: str, now: str) -> Applied:
    """Write the parsed answers into the facts. Mutates `facts` and `gap_log`."""
    applied = Applied()
    for number, gap in enumerate(batch, start=1):
        log = gap_log.setdefault(gap["id"], {"asked": 0})
        log["asked"] = log.get("asked", 0) + 1
        log.setdefault("outcome", "open")
        entries = [a for a in parsed.answers if a.number == number]
        # An answer only counts if it is traceable to what the user actually typed, and if it
        # says something: "don't know" filed as an answer is still a decline.
        non_answers = [a for a in entries if a.kind == "answer" and is_non_answer(a.value)]
        answers = [a for a in entries if a.kind == "answer" and a.value.strip() and a not in non_answers
                   and (supported(a.evidence, reply) or mentions(a.value, reply))]
        declined = any(a.kind == "decline" for a in entries) or bool(non_answers)
        kind, answer_type = gap["kind"], gap["answer_type"]
        ok = False

        if answers:
            evidence = answers[0].evidence.strip() or answers[0].value.strip()
            if answer_type == "scalar":
                value = "; ".join(dict.fromkeys(a.value.strip() for a in answers))
                validator = BY_ID[gap["field"]].validator
                if validator == "severity":
                    value = normalize_severity(value)
                elif validator in ("datetime", "date"):
                    # the model may have completed the user's "5 nov" with a year of its own
                    value = clean_time_value(value, reply) or value
                if value:
                    facts.scalars[gap["field"]] = FieldValue(value=value, status="from_user",
                                                             evidence=evidence, by=user, at=now)
                    ok = True
            elif answer_type == "list":
                items = _split_items([a.value for a in answers])
                if items:
                    lv = facts.lists[gap["field"]]
                    kept = lv.items if (kind != "field" and lv.provided) else []
                    seen = {normalize(i.value) for i in kept}
                    kept += [ListItem(value=i, evidence=i, status="from_user", by=user)
                             for i in items if normalize(i) not in seen]
                    facts.lists[gap["field"]] = ListValue(items=kept, status="from_user", by=user, at=now)
                    ok = True
            elif answer_type == "owner":
                action = facts.actions[gap["index"]]
                action.owner = answers[0].value.strip()
                action.owner_status, action.owner_by, action.owner_at = "from_user", user, now
                due = answers[0].value2.strip()
                if due and mentions(due, reply):
                    action.due_date = due
                ok = True
            elif answer_type == "actions":
                # Free text describing actions: structure it with the Extractor's action pass.
                text = "\n".join(dict.fromkeys(a.evidence.strip() or a.value.strip() for a in answers))
                applied.new_inputs.append({"text": text, "source": "answer about follow-up actions",
                                           "origin": "user", "parts": ["actions"], "override": False})
                applied.answered.append(gap["label"])
                continue  # stays open until the extraction shows whether actions were found

        if ok:
            log["outcome"] = "answered"
            applied.answered.append(gap["label"])
        elif declined:
            _decline_gap(facts, gap, gap_log, applied, user, now)
        elif log["asked"] >= MAX_ASKS_PER_GAP:
            _decline_gap(facts, gap, gap_log, applied, user, now, note="no clear answer given")
        else:
            applied.unclear.append(gap["label"])

    extra = parsed.extra_notes.strip()
    if extra and supported(extra, reply):
        applied.new_inputs.append({"text": extra, "source": "additional notes from the user",
                                   "origin": "user", "parts": [], "override": False})
    return applied


def close_exhausted(facts: IncidentFacts, gaps: list[dict], gap_log: dict, applied: Applied,
                    user: str, now: str) -> None:
    """Close gaps that were already asked the maximum number of times without a usable answer."""
    for gap in gaps:
        if gap.get("asked", 0) >= MAX_ASKS_PER_GAP:
            _decline_gap(facts, gap, gap_log, applied, user, now, note="no clear answer given")


def summarize(applied: Applied) -> str:
    parts = []
    if applied.answered:
        parts.append(f"Recorded: {', '.join(applied.answered)}.")
    if applied.declined:
        parts.append(f"Marked as not provided: {', '.join(applied.declined)}.")
    if applied.unclear:
        parts.append(f"I could not find an answer for: {', '.join(applied.unclear)}. I will ask once more.")
    if applied.new_inputs:
        parts.append("I am reading the extra information you gave.")
    return " ".join(parts) or "Noted."
