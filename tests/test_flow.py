"""The whole workflow through the service: questions, declines, loops, failures, resume, reopen."""

from __future__ import annotations

import pytest

from rca.llm import LLMError, LLMRateLimitError
from rca.llm_fake import demo_handler
from rca.schema.facts import IncidentFacts
from rca.service import RCAService
from tests.conftest import NOW, ROUND_1, ROUND_2, pdf_text, to_approval


def facts_of(service: RCAService, session_id: str) -> IncidentFacts:
    return IncidentFacts.load(service._values(session_id)["facts"])


# ---- the main path -------------------------------------------------------------

def test_notes_to_approved_pdf(service, demo_notes):
    view = service.start(demo_notes, "Asha Rao", "asha@example.com")
    assert view.status == "clarifying" and view.pending["kind"] == "questions" and view.pending["count"] == 5
    assert view.fact_count == 21 and not view.error
    assert "1. What was the severity" in view.messages[-1]["text"]

    view = service.reply(view.session_id, ROUND_1)
    assert "Marked as not provided: Scale of impact" in view.messages[-2]["text"]
    view = service.reply(view.session_id, ROUND_2)
    assert view.pending["kind"] == "questions", "Sev2 makes the lessons-learned fields required"
    assert view.messages[-1]["text"].startswith("Last questions:")
    view = service.reply(view.session_id, "skip all")

    assert view.status == "awaiting_approval" and view.pdf_path.endswith("RCA-INC-4821-v1-draft.pdf")
    assert {g["label"] for g in view.gaps} == {"Scale of impact", "Business or SLA impact",
                                              "What went well", "What went wrong"}
    draft = pdf_text(view.pdf_path)
    assert "Draft, awaiting approval" in draft

    view = service.decide(view.session_id, "approve")
    assert view.status == "approved" and view.finished and view.pending is None
    assert view.pdf_path.endswith("RCA-INC-4821-v1.pdf")
    text = pdf_text(view.pdf_path)
    # The requirement: a declined detail is marked, with the name of the person who declined.
    assert text.count("Not provided by the user (Asha Rao, 11 Nov 2026)") == 4
    assert "Information gaps" in text and "Scale of impact" in text
    assert "Approved by Asha Rao" in text and "Time to detect: 7 minutes" in text
    assert "2026-11-03 08:35 UTC" in text, "times are converted to UTC by code"
    for heading in ("Incident overview", "Executive summary", "Impact", "Timeline", "Detection",
                    "Response and resolution", "Root cause analysis", "Lessons learned",
                    "Corrective and preventive actions", "Related documents", "Information gaps",
                    "Appendix and revision history"):
        assert heading in text


def test_the_person_who_answers_is_the_one_recorded(service, demo_notes):
    view = service.start(demo_notes, "Asha Rao")
    view = service.reply(view.session_id, "skip", user_name="Vikram Shah")
    declined = {g["label"]: g["by"] for g in view.gaps}
    assert declined["Severity"] == "Vikram Shah", "not the person who started the session"
    view = service.reply(view.session_id, "skip all", user_name="Vikram Shah")
    view = service.decide(view.session_id, "approve", user_name="Meera Iyer")
    text = pdf_text(view.pdf_path)
    assert "Not provided by the user (Vikram Shah, 11 Nov 2026)" in text
    assert "Approved by Meera Iyer" in text and "Prepared by Asha Rao" in text.replace("\n", " ")


def test_skip_all_marks_everything_left_and_goes_straight_to_the_draft(service, demo_notes):
    view = service.start(demo_notes, "Asha Rao")
    view = service.reply(view.session_id, "skip all")
    assert view.pending["kind"] == "approval"
    notes = {g["label"]: g["note"] for g in view.gaps}
    assert notes["Severity"] == "skipped" and notes["Incident commander"] == "skipped"
    assert "Root cause (more detail)" in notes, "the Planner's question is listed as unanswered too"
    assert facts_of(service, view.session_id).value("root_cause") == "bad config push"


def test_secrets_never_reach_the_stored_state(service, demo_notes):
    notes = demo_notes + "\nDebug: db password = hunter2!x and key AKIAABCDEFGHIJKLMNOP\n"
    view = service.start(notes, "Asha Rao")
    assert "2 secrets were masked" in view.messages[0]["text"]
    stored = str(service._values(view.session_id))
    assert "hunter2!x" not in stored and "AKIAABCDEFGHIJKLMNOP" not in stored
    view = service.reply(view.session_id, "1. Sev2 2. UTC 3. the token=abc123secret99 leaked 4. skip 5. skip")
    assert "abc123secret99" not in str(service._values(view.session_id))


def test_empty_notes_are_refused(service):
    with pytest.raises(ValueError, match="empty"):
        service.start("   \n", "Asha Rao")


# ---- the final round and unanswered questions ----------------------------------

def test_questions_left_unanswered_in_the_final_round_are_marked(make_service, demo_notes):
    service = make_service(max_clarify_rounds=1)  # the first round is the final one: everything is asked
    view = service.start(demo_notes, "Asha Rao")
    assert view.pending["count"] == 8 and view.messages[-1]["text"].startswith("To complete the RCA")
    view = service.reply(view.session_id, "1. Sev3")  # answers one, ignores the rest
    assert view.pending["kind"] == "approval", "nothing is asked twice after the final round"
    gaps = {g["label"]: g["note"] for g in view.gaps}
    assert gaps["Timezone"] == "no answer given" and gaps["Incident ID"] == "no answer given"
    assert facts_of(service, view.session_id).severity == "Sev3"


def test_a_field_that_becomes_required_late_is_still_asked(make_service, demo_notes):
    service = make_service(max_clarify_rounds=1)
    view = service.start(demo_notes, "Asha Rao")
    view = service.reply(view.session_id, "1. Sev1")  # Sev1 makes five more fields required
    assert view.pending["kind"] == "questions", "newly required details must be asked, not silently marked"
    asked = view.messages[-1]["text"]
    assert "What went well" in asked and "business impact" in asked.lower()
    view = service.reply(view.session_id, "skip")
    assert view.pending["kind"] == "approval"


# ---- review loops --------------------------------------------------------------

def reviewer_that(replies):
    """A handler whose Reviewer answers from a list (the last entry repeats)."""
    replies = list(replies)

    def handler(agent, schema, system, user):
        if schema.__name__ == "ReviewOut":
            return replies.pop(0) if len(replies) > 1 else replies[0]
        return demo_handler(agent, schema, system, user)

    return handler


REVISE = {"verdict": "revise", "missing_info": [],
          "issues": [{"section": "impact", "problem": "It reads as a list.", "fix": "Write sentences."}]}
APPROVE = {"verdict": "approve", "issues": [], "missing_info": []}


def test_a_revision_redrafts_only_the_flagged_section(make_service, demo_notes):
    service = make_service(reviewer_that([REVISE, APPROVE]))
    view = to_approval(service, demo_notes)
    calls = service.llm.calls
    assert calls.count(("reviewer", "ReviewOut")) == 2
    assert calls.count(("drafter", "SectionDraft")) == 4 + 1, "four sections, then impact once more"
    assert calls.count(("drafter", "SummaryDraft")) == 1
    assert view.review_issues == []


def test_revisions_stop_at_the_limit_and_the_comments_are_shown(make_service, demo_notes):
    service = make_service(reviewer_that([REVISE]))  # never satisfied
    view = to_approval(service, demo_notes)
    assert service.llm.calls.count(("reviewer", "ReviewOut")) == service.settings.max_revisions + 1 == 3
    assert [i["problem"] for i in view.review_issues] == ["It reads as a list."]
    assert "1 open comment" in view.messages[-1]["text"]


def test_the_reviewer_can_send_one_question_back_to_the_user(make_service, demo_notes):
    needs = {"verdict": "needs_info", "issues": [],
             "missing_info": [{"field": "monitoring_gaps", "question": "Why did the alert take 7 minutes?"}]}
    service = make_service(reviewer_that([needs, needs, APPROVE]))
    view = service.start(demo_notes, "Asha Rao")
    for answer in (ROUND_1, ROUND_2, "skip all"):
        view = service.reply(view.session_id, answer)
    assert view.pending["kind"] == "questions"
    asked = view.messages[-1]["text"]
    assert asked.startswith("While reviewing the draft") and "Why did the alert take 7 minutes?" in asked
    view = service.reply(view.session_id, "The alert threshold was set too high")
    assert view.pending["kind"] == "approval", "the Reviewer may ask only once"
    assert facts_of(service, view.session_id).value("monitoring_gaps") == "The alert threshold was set too high"
    assert "The alert threshold was set too high" in pdf_text(view.pdf_path)


# ---- failures and resuming -----------------------------------------------------

def test_a_failed_step_can_be_retried_without_losing_the_answer(make_service, demo_notes):
    state = {"fail": True}

    def handler(agent, schema, system, user):
        if schema.__name__ == "ReplyOut" and state["fail"]:
            raise LLMRateLimitError("The rate limit was reached. Try again later.")
        return demo_handler(agent, schema, system, user)

    service = make_service(handler)
    view = service.start(demo_notes, "Asha Rao")
    view = service.reply(view.session_id, ROUND_1)
    assert view.error.startswith("The rate limit was reached") and view.can_retry
    assert service.sessions()[0]["status"] == "error"

    state["fail"] = False
    view = service.retry(view.session_id)
    assert not view.error and view.pending["kind"] == "questions"
    assert facts_of(service, view.session_id).severity == "Sev2", "the answer typed before the failure was kept"
    assert [m["text"] for m in view.messages].count(ROUND_1) == 1, "and it is not duplicated in the chat"


def test_a_session_survives_a_restart(settings, demo_notes):
    from rca.llm_fake import ScriptedLLM

    first = RCAService(settings, llm=ScriptedLLM(demo_handler), now=lambda: NOW)
    view = first.start(demo_notes, "Asha Rao")
    view = first.reply(view.session_id, ROUND_1)
    session_id = view.session_id
    del first

    # A new process: nothing in memory, everything read back from the checkpoint file.
    second = RCAService(settings, llm=ScriptedLLM(demo_handler), now=lambda: NOW)
    resumed = second.view(session_id)
    assert resumed.pending["kind"] == "questions" and resumed.author == "Asha Rao"
    assert resumed.messages[-1]["text"].startswith("Thanks. A few more:")
    assert second.sessions()[0]["session_id"] == session_id
    view = second.reply(session_id, "skip all")
    assert view.pending["kind"] == "approval"


def test_wrong_step_is_refused(service, demo_notes):
    view = service.start(demo_notes, "Asha Rao")
    with pytest.raises(ValueError, match="not waiting for approval"):
        service.decide(view.session_id, "approve")
    view = service.reply(view.session_id, "skip all")
    with pytest.raises(ValueError, match="not waiting for answers"):
        service.reply(view.session_id, "hello")
    with pytest.raises(KeyError):
        service.view("no-such-session")


# ---- change requests and cancelling --------------------------------------------

def test_a_change_request_can_correct_a_fact(service, demo_notes):
    view = to_approval(service, demo_notes)
    before = pdf_text(view.pdf_path)
    assert "Reverted the config change that set the DB pool size to 5" in before
    view = service.decide(view.session_id, "changes",
                          feedback="Fix: Reverted the config change and raised the pool size back to 50")
    assert view.pending["kind"] == "approval" and not view.error
    fix = facts_of(service, view.session_id).scalars["fix_applied"]
    assert fix.value == "Reverted the config change and raised the pool size back to 50"
    assert fix.status == "from_user", "what the user says replaces what the notes said"
    after = pdf_text(view.pdf_path)
    assert "raised the pool size back to 50" in after
    assert "Please change: Fix: Reverted" in [m["text"] for m in view.messages if m["role"] == "user"][-1]


def test_cancel_publishes_and_saves_nothing(service, demo_notes):
    view = to_approval(service, demo_notes)
    view = service.decide(view.session_id, "cancel")
    assert view.status == "cancelled" and view.finished
    assert service.incidents() == []


# ---- long-term memory: reopen and similar incidents ----------------------------

def test_reopening_asks_only_for_what_was_declined(service, demo_notes):
    view = to_approval(service, demo_notes)
    view = service.decide(view.session_id, "approve")
    key = view.incident_key
    assert {g["gap_id"] for g in service.memory.open_gaps(key)} == {
        "field:impact_scale", "field:business_impact", "field:went_well", "field:went_wrong"}
    saved = service.incidents()[0]
    assert (saved["version"], saved["open_gaps"], saved["severity"]) == (1, 4, "Sev2")

    reopened = service.reopen(key, "Vikram Shah")
    assert reopened.version == 2 and reopened.incident_key == key
    assert "4 details were not provided" in reopened.messages[0]["text"]
    asked = reopened.messages[-1]["text"]
    assert reopened.pending["count"] == 4
    for question in ("How many users", "business impact", "What went well", "What went wrong"):
        assert question in asked
    assert "severity" not in asked.lower(), "details already provided are not asked for again"

    reopened = service.reply(reopened.session_id,
                             "1. About 12,000 users\n2. skip\n3. The rollback was quick\n4. skip",
                             user_name="Vikram Shah")
    assert reopened.pending["kind"] == "approval"
    done = service.decide(reopened.session_id, "approve", user_name="Vikram Shah")
    assert done.pdf_path.endswith("RCA-INC-4821-v2.pdf")
    text = pdf_text(done.pdf_path)
    assert "About 12,000 users" in text and "The rollback was quick" in text
    assert "Not provided by the user (Vikram Shah, 11 Nov 2026)" in text
    assert {g["gap_id"] for g in service.memory.open_gaps(key)} == {"field:business_impact", "field:went_wrong"}
    assert service.incidents()[0]["version"] == 2 and len(service.incidents()) == 1
    events = [entry["event"] for entry in service.memory.audit(key)]
    assert events == ["approved", "reopened", "approved"]


def test_similar_past_incidents_are_listed(service, demo_notes):
    first = service.decide(to_approval(service, demo_notes).session_id, "approve")
    second_notes = demo_notes.replace("Payments API outage", "Payments API slow after config change")
    second = to_approval(service, second_notes)
    text = pdf_text(second.pdf_path)
    assert "Payments API outage" in text.split("Related documents")[1].split("Information gaps")[0]
    assert first.incident_key != second.incident_key
    assert service.memory.recall("service") == ["checkout-web", "payments-api"] or \
        set(service.memory.recall("service")) == {"checkout-web", "payments-api"}


def test_model_errors_are_shown_not_raised(make_service, demo_notes):
    def handler(agent, schema, system, user):
        raise LLMError("The model endpoint returned an error (500).")

    view = make_service(handler).start(demo_notes, "Asha Rao")
    assert view.error == "The model endpoint returned an error (500)." and view.can_retry
    assert view.pending is None and view.fact_count == 0
