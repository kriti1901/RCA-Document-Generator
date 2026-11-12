"""The application's entry points: start a session, answer questions, approve, reopen.

The chat UI calls this module and nothing else. A Slack bot or a REST API
would call the same methods.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from langgraph.types import Command

from rca.config import Settings, load_settings
from rca.gaps import declined_gaps
from rca.graph.build import ALL_SECTIONS, Deps, PublishError, build_graph
from rca.llm import LLM, LLMError, build_llm
from rca.memory.long_term import LongTermMemory
from rca.memory.short_term import make_checkpointer
from rca.render.pdf import render_document
from rca.schema.facts import FieldValue, IncidentFacts, ListValue
from rca.security.redact import redact

Progress = Callable[[str], None]

STEP_LABELS = {
    "extract": "Extractor: reading the notes",
    "plan": "Planner: checking the facts against the RCA layout",
    "absorb": "Clarifier: reading your answers",
    "draft": "Drafter: writing the sections",
    "review": "Reviewer: checking the draft against the facts",
    "render": "Building the PDF",
    "feedback": "Applying your changes",
    "finalize": "Building the final PDF",
    "publish": "Publishing to Confluence",
    "memorize": "Saving to long-term memory",
}
FINISHED = {"published", "approved", "cancelled"}
_WAITING = {"questions": "clarifying", "approval": "awaiting_approval"}


@dataclass
class SessionView:
    """Everything the UI needs to draw one session."""

    session_id: str
    incident_key: str = ""
    status: str = ""
    title: str = ""
    author: str = ""  # who started the session
    version: int = 1
    messages: list[dict] = field(default_factory=list)
    pending: dict | None = None  # {"kind": "questions" | "approval", ...} when waiting for the user
    pdf_path: str = ""
    review_issues: list[dict] = field(default_factory=list)
    gaps: list[dict] = field(default_factory=list)  # details declined so far
    rejected: list[dict] = field(default_factory=list)  # extracted items dropped by the evidence check
    confluence: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    fact_count: int = 0
    error: str = ""
    can_retry: bool = False  # a step failed and can be run again

    @property
    def finished(self) -> bool:
        return self.status in FINISHED


class RCAService:
    def __init__(self, settings: Settings | None = None, *, llm: LLM | None = None,
                 memory: LongTermMemory | None = None, checkpointer=None,
                 publish: Callable[[dict], dict] | None = None,
                 now: Callable[[], datetime] = datetime.now) -> None:
        self.settings = settings or load_settings()
        self.now = now
        self.memory = memory or LongTermMemory(self.settings.db_path)
        self.checkpointer = checkpointer or make_checkpointer(self.settings.db_path)
        self._llm = llm
        self._publish = publish
        self._graph = None

    # ---- wiring ----------------------------------------------------------

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = build_llm(self.settings)  # raises LLMConfigError if no key is configured
        return self._llm

    @property
    def publish_fn(self) -> Callable[[dict], dict] | None:
        if self._publish is None and self.settings.confluence is not None:
            from rca.publish.confluence import publish_rca

            cfg = self.settings.confluence
            self._publish = lambda state: publish_rca(state, cfg, now=self.now(),
                                                      model_description=self.llm.description)
        return self._publish

    @property
    def graph(self):
        if self._graph is None:
            deps = Deps(
                settings=self.settings, llm=self.llm, memory=self.memory,
                render=lambda state, final=False: render_document(
                    state, self.settings, final=final, now=self.now(), model_description=self.llm.description),
                publish=self.publish_fn, now=self.now,
            )
            self._graph = build_graph(deps, self.checkpointer)
        return self._graph

    @staticmethod
    def _config(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}, "recursion_limit": 80}

    def _iso(self) -> str:
        return self.now().isoformat(timespec="seconds")

    # ---- running the workflow -------------------------------------------

    def _run(self, session_id: str, payload, on_progress: Progress | None) -> SessionView:
        """Advance the session until it needs the user again, finishes, or a step fails."""
        error = ""
        llm = self.llm
        previous_hook = llm.on_event
        llm.on_event = on_progress
        try:
            for mode, event in self.graph.stream(payload, self._config(session_id),
                                                 stream_mode=["tasks", "updates"]):
                if mode == "tasks" and "input" in event and on_progress:  # a step is starting
                    name = event.get("name", "")
                    label = STEP_LABELS.get(name)
                    if name == "publish":
                        decision = (event.get("input") or {}).get("decision") or {}
                        if not decision.get("publish") or self.publish_fn is None:
                            label = None
                    if label:
                        on_progress(label)
        except (LLMError, PublishError) as exc:
            error = str(exc)
        finally:
            llm.on_event = previous_hook
        view = self.view(session_id)
        view.error = error
        view.can_retry = bool(error)
        self.memory.upsert_session(session_id, view.incident_key, view.title or "Untitled RCA",
                                   "error" if error else view.status,
                                   (self._values(session_id).get("user") or {}).get("name", ""), self._iso())
        return view

    def _values(self, session_id: str) -> dict:
        return self.graph.get_state(self._config(session_id)).values or {}

    def view(self, session_id: str) -> SessionView:
        snapshot = self.graph.get_state(self._config(session_id))
        values = snapshot.values or {}
        if not values:
            raise KeyError(f"No session with id {session_id}")
        pending = None
        for item in snapshot.interrupts:
            pending = dict(item.value)
        messages = list(values.get("messages", []))
        if pending and pending.get("kind") == "questions":
            messages.append({"role": "assistant", "text": pending["text"], "at": ""})
        facts = IncidentFacts.load(values.get("facts"))
        return SessionView(
            session_id=session_id,
            incident_key=values.get("incident_key", ""),
            status=_WAITING.get((pending or {}).get("kind", ""), values.get("status", "")),
            title=facts.value("title"),
            author=(values.get("user") or {}).get("name", ""),
            version=int(values.get("version", 1)),
            messages=messages,
            pending=pending,
            pdf_path=values.get("pdf_path", ""),
            review_issues=list((values.get("review") or {}).get("issues", [])),
            gaps=declined_gaps(facts, values.get("clarifications_declined", [])),
            rejected=list(values.get("rejected", [])),
            confluence=dict(values.get("confluence") or {}),
            usage={agent: dict(stats) for agent, stats in self.llm.usage.items()},
            fact_count=facts.provided_count(),
        )

    # ---- public API ------------------------------------------------------

    def start(self, notes: str | list[dict], user_name: str, user_email: str = "", *,
              on_progress: Progress | None = None) -> SessionView:
        """Begin an RCA from raw notes. `notes` is text, or [{"text", "source"}, ...] for several files."""
        items = [{"text": notes, "source": "pasted notes"}] if isinstance(notes, str) else notes
        now = self._iso()
        inputs, masked, size = [], 0, 0
        for item in items:
            text, count = redact(item["text"])
            if not text.strip():
                continue
            masked += count
            size += len(text)
            inputs.append({"text": text, "source": item.get("source", "notes"), "origin": "notes",
                           "parts": [], "override": False, "extracted": False, "added_at": now})
        if not inputs:
            raise ValueError("The notes are empty.")
        said = f"Submitted incident notes ({size:,} characters from {len(inputs)} source{'s' if len(inputs) != 1 else ''})."
        if masked:
            said += f" {masked} secret{'s were' if masked != 1 else ' was'} masked before anything was stored or sent."
        session_id = uuid.uuid4().hex[:12]
        state = {
            "session_id": session_id, "incident_key": uuid.uuid4().hex[:12], "version": 1,
            "user": {"name": user_name.strip() or "Unknown user", "email": user_email.strip()},
            "raw_inputs": inputs, "facts": IncidentFacts.empty().model_dump(),
            "status": "extracting", "messages": [{"role": "user", "text": said, "at": now}],
        }
        return self._run(session_id, state, on_progress)

    def reply(self, session_id: str, text: str, *, user_name: str = "",
              on_progress: Progress | None = None) -> SessionView:
        """Answer the pending questions. `user_name` is who is answering, if not the session's author."""
        pending = self.view(session_id).pending
        if not pending or pending.get("kind") != "questions":
            raise ValueError("This session is not waiting for answers.")
        text, _ = redact(text)
        return self._run(session_id, Command(resume={"text": text, "by": user_name.strip()}), on_progress)

    def decide(self, session_id: str, action: str, *, publish: bool = False, feedback: str = "",
               user_name: str = "", on_progress: Progress | None = None) -> SessionView:
        """Approve the draft, ask for changes, or cancel. action: "approve" | "changes" | "cancel"."""
        pending = self.view(session_id).pending
        if not pending or pending.get("kind") != "approval":
            raise ValueError("This session is not waiting for approval.")
        if action not in {"approve", "changes", "cancel"}:
            raise ValueError(f"Unknown action: {action}")
        feedback, _ = redact(feedback)
        decision = {"action": action, "publish": bool(publish), "feedback": feedback,
                    "by": user_name.strip()}
        return self._run(session_id, Command(resume=decision), on_progress)

    def retry(self, session_id: str, *, on_progress: Progress | None = None) -> SessionView:
        """Run the step that failed again (for example after a rate limit has passed)."""
        return self._run(session_id, None, on_progress)

    def retry_publish(self, session_id: str) -> SessionView:
        """Publish an approved RCA whose first publish attempt failed."""
        if self.publish_fn is None:
            raise ValueError("Confluence is not configured.")
        values = self._values(session_id)
        try:
            confluence = {**(values.get("confluence") or {}), **self.publish_fn(values), "error": ""}
        except PublishError as exc:
            view = self.view(session_id)
            view.error = str(exc)
            return view
        now = self._iso()
        self.graph.update_state(self._config(session_id), {
            "confluence": confluence, "status": "published",
            "messages": [{"role": "assistant", "text": f"Published to Confluence: {confluence.get('url', '')}",
                          "at": now}],
        }, as_node="memorize")
        record = self.memory.get_incident(values["incident_key"])
        if record:
            record.update(status="published", confluence_page_id=confluence.get("page_id", ""),
                          confluence_url=confluence.get("url", ""))
            self.memory.save_incident(record, [{**gap, "id": gap["gap_id"], "by": gap["declined_by"],
                                                "at": gap["declined_at"]}
                                               for gap in self.memory.open_gaps(values["incident_key"])], now)
        self.memory.log(values["incident_key"], session_id, now, (values.get("user") or {}).get("name", ""),
                        "published", confluence.get("url", ""))
        return self.view(session_id)

    def reopen(self, incident_key: str, user_name: str, user_email: str = "", *,
               on_progress: Progress | None = None) -> SessionView:
        """Start a new version of a saved RCA, asking again only for what was not provided last time."""
        record = self.memory.get_incident(incident_key)
        if record is None:
            raise KeyError(f"No saved incident with key {incident_key}")
        facts = IncidentFacts.load(record["facts"])
        reopened = 0
        for fid, fv in facts.scalars.items():
            if fv.status == "declined":
                facts.scalars[fid] = FieldValue()
                reopened += 1
        for fid, lv in facts.lists.items():
            if lv.status == "declined":
                facts.lists[fid] = ListValue()
                reopened += 1
        if facts.actions_status == "declined":
            facts.actions_status, facts.actions_by, facts.actions_at = "missing", "", ""
            reopened += 1
        for action in facts.actions:
            if action.owner_status == "declined":
                action.owner_status, action.owner_by, action.owner_at = "missing", "", ""
                reopened += 1
        extra = record.get("extra") or {}
        declined_ids = {item["id"] for item in extra.get("clarifications_declined", [])}
        reopened += len(declined_ids)
        # Questions the Planner or Reviewer raised earlier: ask again only those left unanswered.
        gap_log = {gap["id"]: {"asked": 0, "outcome": "answered"}
                   for gap in extra.get("extra_gaps", []) if gap["id"] not in declined_ids}
        now = self._iso()
        version = int(record.get("version") or 1) + 1
        title = record.get("title") or "this incident"
        opening = f'Reopened "{title}" to prepare version {version}.'
        opening += (f" Last time {reopened} detail{'s were' if reopened != 1 else ' was'} not provided, "
                    "so I will ask for those again." if reopened else " Nothing was missing last time.")
        session_id = uuid.uuid4().hex[:12]
        state = {
            "session_id": session_id, "incident_key": incident_key, "version": version,
            "user": {"name": user_name.strip() or "Unknown user", "email": user_email.strip()},
            "raw_inputs": [{**note, "parts": [], "override": False, "extracted": True, "added_at": now}
                           for note in record.get("notes") or []],
            "facts": facts.model_dump(), "plan": {**(extra.get("plan") or {}), "done": True},
            "extra_gaps": extra.get("extra_gaps", []), "gap_log": gap_log,
            "drafts": record.get("drafts") or {},
            # With nothing to ask, redraft everything so the new version is checked afresh.
            "dirty": [] if reopened else list(ALL_SECTIONS),
            "confluence": {"page_id": record.get("confluence_page_id", ""),
                           "url": record.get("confluence_url", "")},
            "status": "clarifying", "messages": [{"role": "assistant", "text": opening, "at": now}],
        }
        self.memory.log(incident_key, session_id, now, state["user"]["name"], "reopened", f"version {version}")
        return self._run(session_id, state, on_progress)

    def incidents(self) -> list[dict]:
        return self.memory.list_incidents()

    def sessions(self) -> list[dict]:
        return self.memory.list_sessions()
