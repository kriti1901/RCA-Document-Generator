"""Extractor agent: raw notes -> facts, each backed by a quote that code then verifies."""

from __future__ import annotations

from dataclasses import dataclass, field

from rca.agents.common import DATA_NOT_INSTRUCTIONS, block
from rca.llm import LLM
from rca.schema.facts import ActionItem, FieldValue, IncidentFacts, ListItem, TimelineEvent
from rca.schema.field_specs import BY_ID, MILESTONE_IDS, fields_in_group, normalize_severity
from rca.schema.llm_io import ActionsOut, TimelineOut, facts_schema_for
from rca.security.evidence import mentions, normalize, supported
from rca.timeutil import clean_time_value, parse_time

SYSTEM = f"""You extract facts from raw incident notes for a Root Cause Analysis (RCA) document.

{DATA_NOT_INSTRUCTIONS}

Rules:
1. Extract only what the notes state. Never guess, infer or fill in details that are not written.
2. For every item give `evidence`: a short quote copied word for word from the notes that supports it. Do not paraphrase the quote.
3. If the notes do not state something, leave it out. An empty list is a valid answer.
4. Keep each value short and factual.
5. Keep uncertainty. If the notes give something as a guess, an opinion or a suspicion ("probably", "X thinks", "might be"), say so in the value, for example "suspected: certificate rotation"."""

TIME_RULE = (
    "Write times in 24-hour time. If the notes state the full date including the year, write "
    "YYYY-MM-DD HH:MM. If they state a day and month but no year, write D Mon HH:MM (for example "
    "5 Nov 09:40). If they state only a time, write HH:MM. Never add a date or a year that is not "
    "written in the notes, and do not convert between timezones."
)

FACTS_TASK = """Extract the fields listed below. Put the field's id in `field`.
Fields marked [list] can appear several times, once per item.
{time_rule}
{hints}
{fields}

{notes}"""

TIMELINE_TASK = """List the timeline events in the notes: each thing that happened at a stated time, in the order it happened.
For each event:
- time: {time_rule}
- event: what happened, in one short sentence.
- source: who or what did or reported it, or "" if not stated.
- evidence: the supporting quote.
Leave out anything that has no stated time.

{notes}"""

ACTIONS_TASK = """List the follow-up actions in the notes: things to be done after the incident to fix its causes or prevent a repeat. Do not include steps that were taken during the incident itself.
For each action:
- action: what is to be done.
- type: corrective (fixes the cause), preventive (stops a repeat), detective (improves detection), or unknown.
- owner, due_date, priority, ticket: as stated in the notes, or "" if not stated.
- evidence: the supporting quote.

{notes}"""

GROUPS = ("overview", "analysis")
ALL_PARTS = ("overview", "analysis", "timeline", "actions")


@dataclass
class Extraction:
    """What one pass over one piece of text produced, after the evidence check."""

    scalars: dict[str, FieldValue] = field(default_factory=dict)
    list_items: dict[str, list[ListItem]] = field(default_factory=dict)
    events: list[TimelineEvent] = field(default_factory=list)
    actions: list[ActionItem] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)

    @property
    def count(self) -> int:
        return (len(self.scalars) + sum(len(v) for v in self.list_items.values())
                + len(self.events) + len(self.actions))


def split_chunks(text: str, max_chars: int) -> list[str]:
    """Split long notes on blank lines (then on lines) so each call stays small."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        pieces = [paragraph]
        if len(paragraph) > max_chars:
            pieces, line_buf = [], ""
            for line in paragraph.splitlines():
                while len(line) > max_chars:  # a single enormous line
                    pieces.append(line[:max_chars])
                    line = line[max_chars:]
                if line_buf and len(line_buf) + len(line) + 1 > max_chars:
                    pieces.append(line_buf)
                    line_buf = line
                else:
                    line_buf = f"{line_buf}\n{line}" if line_buf else line
            if line_buf:
                pieces.append(line_buf)
        for piece in pieces:
            if current and len(current) + len(piece) + 2 > max_chars:
                chunks.append(current)
                current = piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def _field_lines(group: str) -> str:
    lines = []
    for spec in fields_in_group(group):
        tag = " [list]" if spec.kind == "list" else ""
        lines.append(f"- {spec.id}{tag}: {spec.hint}")
    return block("fields", "\n".join(lines))


def extract(llm: LLM, text: str, *, origin: str = "notes", by: str = "", at: str = "",
            parts: tuple[str, ...] = ALL_PARTS, max_chars: int = 12000,
            known_services: tuple[str, ...] = ()) -> Extraction:
    """Run the extraction calls over `text` and keep only items whose quote is found in it.

    origin is "notes" for the pasted notes and "user" for text the user typed
    in answer to a question; it decides the status the facts are stored with.
    """
    status = "from_user" if origin == "user" else "from_notes"
    result = Extraction()
    hints = ""
    if known_services:
        hints = ("Service names seen in earlier incidents (use the same spelling if the notes mean one "
                 "of these): " + ", ".join(known_services[:30]) + "\n")

    def reject(kind: str, item: dict, reason: str) -> None:
        result.rejected.append({"kind": kind, "reason": reason, **item})

    for chunk in split_chunks(text, max_chars):
        notes = block("notes", chunk)

        for group in GROUPS:
            if group not in parts:
                continue
            ids = tuple(spec.id for spec in fields_in_group(group))
            out = llm.structured(
                "extractor", facts_schema_for(ids), SYSTEM,
                FACTS_TASK.format(time_rule=TIME_RULE, hints=hints, fields=_field_lines(group), notes=notes),
            )
            for fact in out.facts:
                value, quote = fact.value.strip(), fact.evidence.strip()
                item = {"field": fact.field, "value": value, "evidence": quote}
                if not value:
                    continue
                if not supported(quote, text):
                    reject("fact", item, "quote not found in the notes")
                    continue
                if value.endswith(("...", "\u2026")):
                    reject("fact", item, "the value was cut short")
                    continue
                spec = BY_ID[fact.field]
                if spec.validator == "severity":
                    value = normalize_severity(value) or normalize_severity(quote)
                    if not value:
                        reject("fact", item, "not a recognisable severity")
                        continue
                if spec.validator in ("datetime", "date"):
                    # A real quote can still be filed under the wrong field, and models like to
                    # complete a date with a year from their training data. Keep only what is stated.
                    value = clean_time_value(value, text)
                    parsed = parse_time(value)
                    wanted = parsed.has_day if spec.validator == "date" else (parsed.has_time or parsed.has_day)
                    if not wanted:
                        reject("fact", item, "not a date or time stated in the notes")
                        continue
                if spec.kind == "list":
                    items = result.list_items.setdefault(fact.field, [])
                    if normalize(value) not in {normalize(i.value) for i in items}:
                        items.append(ListItem(value=value, evidence=quote, status=status, by=by))
                elif fact.field not in result.scalars:  # first statement wins within one text
                    result.scalars[fact.field] = FieldValue(value=value, status=status, evidence=quote,
                                                            by=by, at=at)

        if "timeline" in parts:
            out_t = llm.structured("extractor_lists", TimelineOut, SYSTEM,
                                   TIMELINE_TASK.format(time_rule=TIME_RULE, notes=notes))
            for ev in out_t.events:
                item = {"time": ev.time, "event": ev.event, "evidence": ev.evidence}
                if not ev.time.strip() or not ev.event.strip():
                    continue
                if not supported(ev.evidence, text):
                    reject("timeline event", item, "quote not found in the notes")
                    continue
                when = clean_time_value(ev.time, text)
                if not parse_time(when).has_time:
                    reject("timeline event", item, "no time stated in the notes")
                    continue
                result.events.append(TimelineEvent(time=when, event=ev.event.strip(),
                                                   source=ev.source.strip(), evidence=ev.evidence.strip(),
                                                   status=status))

        if "actions" in parts:
            out_a = llm.structured("extractor_lists", ActionsOut, SYSTEM, ACTIONS_TASK.format(notes=notes))
            for act in out_a.actions:
                item = {"action": act.action, "evidence": act.evidence}
                if not act.action.strip():
                    continue
                if not supported(act.evidence, text):
                    reject("action", item, "quote not found in the notes")
                    continue
                owner = act.owner.strip()
                if owner and not mentions(owner, text):
                    reject("action owner", {"action": act.action, "owner": owner}, "name not found in the notes")
                    owner = ""
                result.actions.append(ActionItem(
                    action=act.action.strip(),
                    type="" if act.type == "unknown" else act.type,
                    owner=owner,
                    owner_status=status if owner else "missing",
                    owner_by=by if owner else "",
                    owner_at=at if owner else "",
                    due_date=act.due_date.strip(),
                    priority=act.priority.strip(),
                    ticket=act.ticket.strip(),
                    evidence=act.evidence.strip(),
                    status=status,
                ))
    return result


def merge(facts: IncidentFacts, new: Extraction, *, override: bool = False, by: str = "", at: str = "") -> int:
    """Fold an extraction into the facts. Returns how many items changed.

    Existing values are kept unless `override` is set, which is used for
    corrections the user typed: what the user says replaces what the notes said.
    """
    changed = 0
    for fid, fv in new.scalars.items():
        current = facts.scalars[fid]
        if override or not current.provided:
            facts.scalars[fid] = fv
            changed += 1
    for fid, items in new.list_items.items():
        lv = facts.lists[fid]
        seen = {normalize(i.value) for i in lv.items}
        added = [i for i in items if normalize(i.value) not in seen]
        if added:
            lv.items.extend(added)
            lv.status = added[0].status if not lv.provided else lv.status
            lv.by, lv.at, lv.note = by or lv.by, at or lv.at, ""
            changed += len(added)
    seen_events = {(normalize(e.time), normalize(e.event)) for e in facts.timeline}
    for ev in new.events:
        key = (normalize(ev.time), normalize(ev.event))
        if key not in seen_events:
            facts.timeline.append(ev)
            seen_events.add(key)
            changed += 1
    by_text = {normalize(a.action): a for a in facts.actions}
    for act in new.actions:
        existing = by_text.get(normalize(act.action))
        if existing is None:
            facts.actions.append(act)
            by_text[normalize(act.action)] = act
            changed += 1
        elif override:
            for attr in ("owner", "due_date", "priority", "ticket", "type"):
                if getattr(act, attr):
                    setattr(existing, attr, getattr(act, attr))
            if act.owner:
                existing.owner_status, existing.owner_by, existing.owner_at = act.owner_status, by, at
            changed += 1
    if facts.actions and facts.actions_status in ("missing", "declined"):
        facts.actions_status = facts.actions[0].status
        facts.actions_by, facts.actions_at = by, at
    return changed


__all__ = ["ALL_PARTS", "Extraction", "MILESTONE_IDS", "extract", "merge", "split_chunks"]
