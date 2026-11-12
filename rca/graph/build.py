"""The workflow: which agent runs when, where the user is asked, and where the loops stop.

Routing is ordinary code. The agents never decide what happens next; they
return data, and the functions here choose the next step from that data.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from rca.agents import clarifier, drafter, extractor, planner, reviewer
from rca.config import Settings
from rca.gaps import declined_gaps, extra_gap, open_gaps, select_batch
from rca.graph.state import RCAState
from rca.llm import LLM
from rca.memory.long_term import LongTermMemory
from rca.schema.facts import FieldValue, IncidentFacts
from rca.schema.field_specs import MILESTONE_IDS, NARRATIVE_SECTIONS
from rca.timeutil import compute_milestones

ALL_SECTIONS: tuple[str, ...] = ("summary",) + NARRATIVE_SECTIONS


class PublishError(RuntimeError):
    """Publishing failed; the message is written to be shown to the user."""


@dataclass
class Deps:
    settings: Settings
    llm: LLM
    memory: LongTermMemory
    render: Callable[..., dict]  # (state, final=False) -> {"pdf_path": ...}
    publish: Callable[[dict], dict] | None  # state -> {"page_id", "url", "title", "version"}
    now: Callable[[], datetime] = datetime.now


def durations_for(facts: IncidentFacts, year: int) -> dict[str, str]:
    values = {key: facts.value(key) for key in MILESTONE_IDS}
    return compute_milestones(values, facts.value("timezone"), default_year=year,
                              base_date=facts.value("incident_date")).durations


def similarity_text(facts: IncidentFacts) -> str:
    parts = [facts.value(fid) for fid in ("title", "root_cause", "trigger", "customer_impact")]
    parts += [item.value for item in facts.lists["affected_services"].items]
    return " ".join(p for p in parts if p)


def build_graph(deps: Deps, checkpointer):
    settings = deps.settings

    def iso() -> str:
        return deps.now().isoformat(timespec="seconds")

    def msg(text: str, role: str = "assistant") -> dict:
        return {"role": role, "text": text, "at": iso()}

    def load(state: RCAState) -> IncidentFacts:
        return IncidentFacts.load(state.get("facts"))

    def gaps_of(state: RCAState, facts: IncidentFacts | None = None) -> list[dict]:
        return open_gaps(facts or load(state), state.get("extra_gaps", []), state.get("gap_log", {}))

    def pending_inputs(state: RCAState) -> bool:
        return any(not item.get("extracted") for item in state.get("raw_inputs", []))

    # ---- nodes -----------------------------------------------------------

    def extract_node(state: RCAState) -> dict:
        facts = load(state)
        inputs = [dict(item) for item in state.get("raw_inputs", [])]
        rejected = list(state.get("rejected", []))
        gap_log = copy.deepcopy(state.get("gap_log", {}))
        user, now = state.get("reply_by") or state["user"]["name"], iso()
        first_pass = not any(item.get("extracted") for item in inputs)
        known_services = tuple(deps.memory.recall("service"))
        found = dropped = 0
        announce = False
        for item in inputs:
            if item.get("extracted"):
                continue
            from_user = item.get("origin") == "user"
            result = extractor.extract(
                deps.llm, item["text"], origin=item.get("origin", "notes"),
                by=user if from_user else "", at=now if from_user else "",
                parts=tuple(item.get("parts") or extractor.ALL_PARTS),
                max_chars=settings.chunk_chars, known_services=known_services,
            )
            found += extractor.merge(facts, result, override=bool(item.get("override")),
                                     by=user if from_user else "", at=now if from_user else "")
            dropped += len(result.rejected)
            rejected += result.rejected
            announce = announce or not item.get("override")
            item["extracted"] = True

        # An answer about follow-up actions that yielded none: stop asking after the limit.
        applied = clarifier.Applied()
        clarifier.close_exhausted(facts, open_gaps(facts, state.get("extra_gaps", []), gap_log),
                                  gap_log, applied, user, now)

        messages = []
        if first_pass:
            text = f"I read the notes and found {found} facts."
            if dropped:
                text += (f" I dropped {dropped} item{'s' if dropped != 1 else ''} because I could not "
                         "find a supporting quote in the notes.")
            messages.append(msg(text))
        elif announce:
            messages.append(msg(f"I added {found} item{'s' if found != 1 else ''} from what you wrote."
                                if found else "I found nothing new to add from what you wrote."))
        update = {
            "facts": facts.model_dump(), "raw_inputs": inputs, "rejected": rejected, "gap_log": gap_log,
            "clarifications_declined": state.get("clarifications_declined", []) + applied.clarifications_declined,
            "status": "extracting", "messages": messages,
        }
        if state.get("drafts"):
            update["dirty"] = list(ALL_SECTIONS)  # facts changed after a draft existed
        return update

    def plan_node(state: RCAState) -> dict:
        facts = load(state)
        result = planner.plan(deps.llm, facts, durations_for(facts, deps.now().year))
        extra = list(state.get("extra_gaps", []))
        known = {gap["id"] for gap in extra}
        for found in result["quality_gaps"]:
            gap = extra_gap("quality", found["field"], found["question"], found["problem"],
                            facts.value(found["field"]))
            if gap["id"] not in known:
                extra.append(gap)
        similar = deps.memory.find_similar(similarity_text(facts), exclude_key=state.get("incident_key", ""))
        return {"plan": {"section_notes": result["section_notes"], "done": True},
                "extra_gaps": extra, "similar": similar, "status": "planning"}

    def ask_node(state: RCAState) -> dict:
        # Everything before interrupt() runs again when the session resumes, so it
        # must be a pure function of the state: no model calls, no side effects.
        gaps = gaps_of(state)
        rounds = state.get("clarify_rounds", 0)
        from_review = bool(state.get("review_round"))
        if from_review:
            batch = gaps
            text = clarifier.format_questions(batch, round_index=rounds, final=True,
                                              intro="While reviewing the draft I found I need a little more:")
        else:
            batch = select_batch(gaps, rounds, settings.questions_per_round, settings.max_clarify_rounds)
            text = clarifier.format_questions(batch, round_index=rounds,
                                              final=rounds >= settings.max_clarify_rounds - 1)
        answer = interrupt({"kind": "questions", "text": text, "count": len(batch)})
        if isinstance(answer, dict):
            reply, reply_by = str(answer.get("text", "")), str(answer.get("by", ""))
        else:
            reply, reply_by = str(answer), ""
        return {"asked": batch, "reply": reply, "reply_by": reply_by, "status": "clarifying",
                "messages": [msg(text), msg(reply, "user")]}

    def absorb_node(state: RCAState) -> dict:
        facts = load(state)
        batch, reply = state.get("asked", []), state.get("reply", "")
        gap_log = copy.deepcopy(state.get("gap_log", {}))
        extra = state.get("extra_gaps", [])
        user, now = state.get("reply_by") or state["user"]["name"], iso()
        rounds = state.get("clarify_rounds", 0)
        was_final = bool(state.get("review_round")) or rounds >= settings.max_clarify_rounds - 1

        parsed = clarifier.parse_reply(deps.llm, batch, reply)
        applied = clarifier.apply_reply(facts, batch, parsed, reply, gap_log, user, now)
        awaiting_extraction = any("actions" in (item.get("parts") or []) for item in applied.new_inputs)
        if parsed.skip_rest:
            clarifier.decline_all(facts, open_gaps(facts, extra, gap_log), gap_log, applied, user, now,
                                  note="skipped")
            applied.unclear = []  # nothing will be asked again
        elif was_final:
            # The final round asked everything that was left. Whatever is still open was
            # asked and not answered, so "not provided by the user" is accurate.
            asked_ids = {gap["id"] for gap in batch}
            leftover = [gap for gap in open_gaps(facts, extra, gap_log)
                        if gap["id"] in asked_ids and not (gap["kind"] == "actions" and awaiting_extraction)]
            clarifier.decline_all(facts, leftover, gap_log, applied, user, now, note="no answer given")
            applied.unclear = []

        inputs = list(state.get("raw_inputs", []))
        for item in applied.new_inputs:
            inputs.append({**item, "extracted": False, "added_at": now})
        update = {
            "facts": facts.model_dump(), "gap_log": gap_log, "raw_inputs": inputs,
            "clarifications_declined": state.get("clarifications_declined", []) + applied.clarifications_declined,
            "clarify_rounds": rounds + 1, "asked": [], "reply": "", "review_round": False,
            "messages": [msg(clarifier.summarize(applied))],
        }
        if state.get("drafts") and (applied.answered or applied.declined):
            update["dirty"] = list(ALL_SECTIONS)
        return update

    def draft_node(state: RCAState) -> dict:
        facts = load(state)
        previous = state.get("drafts") or {}
        feedback = state.get("feedback", "")
        dirty = tuple(state.get("dirty") or ())
        sections = ALL_SECTIONS if (not previous or feedback) else dirty
        issues: dict[str, list[str]] = {}
        for issue in (state.get("review") or {}).get("issues", []):
            if issue["section"] in sections:
                issues.setdefault(issue["section"], []).append(f"{issue['problem']} {issue['fix']}".strip())
        result = drafter.draft(
            deps.llm, facts, durations_for(facts, deps.now().year), sections=sections, previous=previous,
            notes=(state.get("plan") or {}).get("section_notes", {}), issues=issues, feedback=feedback,
        )
        update: dict = {"drafts": result["sections"], "dirty": [], "feedback": "", "status": "drafting"}
        title = facts.scalars["title"]
        if result["title"] and (not title.provided or title.status == "derived"):
            facts.scalars["title"] = FieldValue(value=result["title"], status="derived",
                                                note="written by the Drafter from the facts", at=iso())
            update["facts"] = facts.model_dump()
        return update

    def review_node(state: RCAState) -> dict:
        facts = load(state)
        result = reviewer.review(deps.llm, facts, durations_for(facts, deps.now().year), state.get("drafts") or {})
        revisions = state.get("revisions", 0)
        update: dict = {"review": result, "status": "reviewing", "review_action": "render"}
        if result["issues"] and revisions < settings.max_revisions:
            update.update(review_action="revise", revisions=revisions + 1,
                          dirty=sorted({issue["section"] for issue in result["issues"]}))
        elif result["missing_info"] and not state.get("review_info_used"):
            extra = list(state.get("extra_gaps", []))
            known = {gap["id"] for gap in extra}
            added = False
            for info in result["missing_info"]:
                fid = info["field"]
                status = facts.scalars[fid].status if fid in facts.scalars else facts.lists[fid].status
                gap = extra_gap("review", fid, info["question"])
                if status != "declined" and gap["id"] not in known:
                    extra.append(gap)
                    added = True
            if added:
                update.update(review_action="ask", extra_gaps=extra, review_info_used=True, review_round=True)
        return update

    def render_node(state: RCAState) -> dict:
        rendered = deps.render(state, final=False)
        open_issues = (state.get("review") or {}).get("issues", [])
        text = "The draft RCA is ready. Please review the PDF, then approve it or tell me what to change."
        if open_issues:
            text += (f" The Reviewer still has {len(open_issues)} open comment"
                     f"{'s' if len(open_issues) != 1 else ''} on the wording, shown next to the preview.")
        return {"pdf_path": rendered["pdf_path"], "status": "awaiting_approval", "messages": [msg(text)]}

    def approval_node(state: RCAState) -> dict:
        decision = interrupt({"kind": "approval", "pdf_path": state.get("pdf_path", "")})
        decision = dict(decision or {})
        action = decision.get("action", "cancel")
        if action == "approve":
            said = "Approved." + (" Publish to Confluence." if decision.get("publish") else "")
        elif action == "changes":
            said = f"Please change: {decision.get('feedback', '')}"
        else:
            said = "Cancel this RCA."
        update = {"decision": decision, "messages": [msg(said, "user")]}
        if action == "approve":
            update["approved_by"] = decision.get("by") or state["user"]["name"]
        return update

    def feedback_node(state: RCAState) -> dict:
        text = (state.get("decision") or {}).get("feedback", "").strip()
        inputs = list(state.get("raw_inputs", []))
        # The request may contain corrected facts, so it goes through the Extractor too,
        # with override: what the user says replaces what the notes said.
        inputs.append({"text": text, "source": "change request", "origin": "user", "parts": [],
                       "override": True, "extracted": False, "added_at": iso()})
        return {"raw_inputs": inputs, "feedback": text, "revisions": 0, "dirty": list(ALL_SECTIONS),
                "reply_by": (state.get("decision") or {}).get("by", ""),
                "status": "revising", "messages": [msg("Updating the draft with your changes.")]}

    def finalize_node(state: RCAState) -> dict:
        rendered = deps.render(state, final=True)
        return {"pdf_path": rendered["pdf_path"], "status": "approved"}

    def publish_node(state: RCAState) -> dict:
        decision = state.get("decision") or {}
        confluence = dict(state.get("confluence") or {})
        if not decision.get("publish"):
            return {"status": "approved", "messages": [msg("Approved. The final PDF is saved.")]}
        if deps.publish is None:
            return {"status": "approved",
                    "messages": [msg("Approved. The final PDF is saved. Confluence is not configured, "
                                     "so it was not published.")]}
        try:
            confluence = {**confluence, **deps.publish(state), "error": ""}
        except PublishError as exc:
            confluence["error"] = str(exc)
            return {"confluence": confluence, "status": "approved",
                    "messages": [msg("Approved, and the final PDF is saved, but publishing to Confluence "
                                     f"failed: {exc}")]}
        return {"confluence": confluence, "status": "published",
                "messages": [msg(f"Published to Confluence: {confluence.get('url', '')}")]}

    def memorize_node(state: RCAState) -> dict:
        facts = load(state)
        now, user = iso(), state["user"]["name"]
        confluence = state.get("confluence") or {}
        key = state["incident_key"]
        gaps = declined_gaps(facts, state.get("clarifications_declined", []))
        services = [item.value for item in facts.lists["affected_services"].items]
        deps.memory.save_incident({
            "incident_key": key, "incident_id": facts.value("incident_id"), "title": facts.value("title"),
            "severity": facts.severity, "services": ", ".join(services),
            "started_at": facts.value("started_at"),
            "summary": " ".join((state.get("drafts") or {}).get("summary", [])),
            "root_cause": facts.value("root_cause"), "status": state.get("status", "approved"),
            "version": state.get("version", 1), "facts": state.get("facts", {}),
            "drafts": state.get("drafts", {}),
            "notes": [{"text": item["text"], "source": item.get("source", ""), "origin": item.get("origin", "")}
                      for item in state.get("raw_inputs", [])],
            "extra": {"plan": state.get("plan", {}), "extra_gaps": state.get("extra_gaps", []),
                      "clarifications_declined": state.get("clarifications_declined", [])},
            "pdf_path": state.get("pdf_path", ""), "confluence_page_id": confluence.get("page_id", ""),
            "confluence_url": confluence.get("url", ""), "created_by": user,
        }, gaps, now)
        for service in services:
            deps.memory.remember("service", service, now, source=key)
        commander = facts.value("incident_commander")
        if commander:
            deps.memory.remember("person", commander, now, value="incident commander", source=key)
        deps.memory.log(key, state.get("session_id", ""), now, state.get("approved_by") or user,
                        state.get("status", "approved"),
                        f"version {state.get('version', 1)}; {len(gaps)} information gap(s)")
        return {}

    def cancel_node(state: RCAState) -> dict:
        return {"status": "cancelled", "messages": [msg("Cancelled. Nothing was published.")]}

    # ---- routing ---------------------------------------------------------

    def after_gaps(state: RCAState) -> str:
        return "ask" if gaps_of(state) else "draft"

    def route_start(state: RCAState) -> str:
        return "extract" if pending_inputs(state) else after_gaps(state)

    def after_extract(state: RCAState) -> str:
        if not (state.get("plan") or {}).get("done"):
            return "plan"
        return after_gaps(state)

    def after_absorb(state: RCAState) -> str:
        return "extract" if pending_inputs(state) else after_gaps(state)

    def after_review(state: RCAState) -> str:
        return state.get("review_action", "render")

    def after_approval(state: RCAState) -> str:
        action = (state.get("decision") or {}).get("action")
        if action == "approve":
            return "finalize"
        if action == "changes" and (state.get("decision") or {}).get("feedback", "").strip():
            return "feedback"
        return "cancel"

    # ---- wiring ----------------------------------------------------------

    graph = StateGraph(RCAState)
    for name, node in (("extract", extract_node), ("plan", plan_node), ("ask", ask_node),
                       ("absorb", absorb_node), ("draft", draft_node), ("review", review_node),
                       ("render", render_node), ("approval", approval_node), ("feedback", feedback_node),
                       ("finalize", finalize_node), ("publish", publish_node),
                       ("memorize", memorize_node), ("cancel", cancel_node)):
        graph.add_node(name, node)

    step = {"extract": "extract", "plan": "plan", "ask": "ask", "draft": "draft"}
    graph.add_conditional_edges(START, route_start, {k: step[k] for k in ("extract", "ask", "draft")})
    graph.add_conditional_edges("extract", after_extract, {k: step[k] for k in ("plan", "ask", "draft")})
    graph.add_conditional_edges("plan", after_gaps, {k: step[k] for k in ("ask", "draft")})
    graph.add_edge("ask", "absorb")
    graph.add_conditional_edges("absorb", after_absorb, {k: step[k] for k in ("extract", "ask", "draft")})
    graph.add_edge("draft", "review")
    graph.add_conditional_edges("review", after_review, {"revise": "draft", "ask": "ask", "render": "render"})
    graph.add_edge("render", "approval")
    graph.add_conditional_edges("approval", after_approval,
                                {"finalize": "finalize", "feedback": "feedback", "cancel": "cancel"})
    graph.add_edge("feedback", "extract")
    graph.add_edge("finalize", "publish")
    graph.add_edge("publish", "memorize")
    graph.add_edge("memorize", END)
    graph.add_edge("cancel", END)
    return graph.compile(checkpointer=checkpointer)
