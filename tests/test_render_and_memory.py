"""The document view, the PDF, and long-term memory."""

from __future__ import annotations

from datetime import datetime

from rca.memory.long_term import LongTermMemory
from rca.render.document import build_view, not_provided
from rca.render.pdf import pdf_filename, render_document, render_html, slugify
from rca.schema.facts import ActionItem, FieldValue, IncidentFacts, ListItem, ListValue, TimelineEvent
from tests.conftest import NOW, pdf_text

AT = "2026-11-11T10:00:00"


def state_with(facts: IncidentFacts, **extra) -> dict:
    return {"incident_key": "abc123", "version": 1, "user": {"name": "Asha Rao"},
            "facts": facts.model_dump(), "drafts": {}, **extra}


def test_declined_marker_names_the_person_and_date():
    assert not_provided("Asha Rao", AT) == "Not provided by the user (Asha Rao, 11 Nov 2026)"
    assert not_provided("", "") == "Not provided by the user"


def test_view_shows_required_gaps_and_hides_unasked_optional_fields():
    facts = IncidentFacts.empty()
    facts.scalars["severity"] = FieldValue(value="Sev3", status="from_user")
    facts.scalars["customer_impact"] = FieldValue(status="declined", by="Asha Rao", at=AT)
    facts.scalars["monitoring_gaps"] = FieldValue(value="No alert on pool saturation", status="from_notes")
    view = build_view(state_with(facts), now=NOW, final=False)

    impact = {row["id"]: row for row in view["impact"]["rows"]}
    assert impact["customer_impact"]["state"] == "declined"
    assert impact["customer_impact"]["text"] == "Not provided by the user (Asha Rao, 11 Nov 2026)"
    assert impact["impact_scale"]["state"] == "missing", "required but never answered: shown, not hidden"
    assert "business_impact" not in impact, "only required for Sev1-2, and this is Sev3"
    detection = {row["id"] for row in view["detection"]["rows"]}
    assert detection == {"detection_method", "monitoring_gaps"}, "an optional field appears once it has content"
    assert view["lessons"] == [] and view["actions"]["state"] == "missing"
    assert [gap["label"] for gap in view["gaps"]] == ["Customer impact"]


def test_view_orders_events_and_handles_midnight():
    facts = IncidentFacts.empty()
    facts.scalars["started_at"] = FieldValue(value="2026-11-03 23:50", status="from_notes")
    facts.scalars["resolved_at"] = FieldValue(value="2026-11-04 00:40", status="from_notes")
    facts.scalars["timezone"] = FieldValue(value="UTC", status="from_notes")
    facts.timeline = [TimelineEvent(time="00:10", event="Rollback finished"),
                      TimelineEvent(time="23:45", event="Deploy started"),
                      TimelineEvent(time="23:58", event="Alert fired")]
    timeline = build_view(state_with(facts), now=NOW, final=False)["timeline"]
    assert [e["event"] for e in timeline["events"]] == ["Deploy started", "Alert fired", "Rollback finished"]
    assert timeline["events"][-1]["time"] == "2026-11-04 00:10", "a time after midnight lands on the next day"
    assert timeline["durations"] == [{"label": "Time to resolve", "text": "50 minutes"}]
    assert timeline["caption"] == "Times were recorded in UTC."


def test_a_date_without_a_year_is_shown_without_one():
    facts = IncidentFacts.empty()
    facts.scalars["started_at"] = FieldValue(value="5 Nov 09:40", status="from_notes")
    facts.scalars["resolved_at"] = FieldValue(value="10:50", status="from_notes")
    facts.scalars["timezone"] = FieldValue(value="UTC", status="from_notes")
    view = build_view(state_with(facts), now=NOW, final=False)
    header = {row["label"]: row for row in view["header_rows"]}
    assert header["Incident date"] == {"label": "Incident date", "state": "ok", "text": "5 Nov"}
    assert [row["local"] for row in view["timeline"]["rows"]][0] == "5 Nov 09:40"
    assert "2026" not in str(view["timeline"]["rows"]), "the current year is never filled in"

    facts.scalars["incident_date"] = FieldValue(value="5 November 2025", status="from_user", by="Asha Rao")
    view = build_view(state_with(facts), now=NOW, final=False)
    header = {row["label"]: row for row in view["header_rows"]}
    assert header["Incident date"]["text"] == "5 Nov 2025"
    assert view["timeline"]["rows"][0]["local"] == "2025-11-05 09:40"
    assert view["timeline"]["rows"][3]["utc"] == "2025-11-05 10:50 UTC"

    nothing = build_view(state_with(IncidentFacts.empty()), now=NOW, final=False)
    assert {row["label"]: row for row in nothing["header_rows"]}["Incident date"]["state"] == "missing"


def test_text_from_the_notes_cannot_inject_markup():
    facts = IncidentFacts.empty()
    facts.scalars["title"] = FieldValue(value='<script>alert("x")</script> outage', status="from_notes")
    facts.scalars["root_cause"] = FieldValue(value="<img src=x onerror=alert(1)>", status="from_notes")
    facts.lists["affected_services"] = ListValue(items=[ListItem(value="<b>api</b>")], status="from_notes")
    state = state_with(facts, drafts={"summary": ["</p><h1>Injected</h1>"]})
    html = render_html(build_view(state, now=NOW, final=False))
    assert "<script>" not in html and "<img src=x" not in html and "<h1>Injected" not in html
    assert "&lt;script&gt;" in html and "&lt;b&gt;api&lt;/b&gt;" in html


def test_pdf_renders_even_when_nothing_was_provided(settings):
    facts = IncidentFacts.empty()
    for fid in facts.scalars:
        facts.scalars[fid] = FieldValue(status="declined", by="Asha Rao", at=AT)
    for fid in facts.lists:
        facts.lists[fid] = ListValue(status="declined", by="Asha Rao", at=AT)
    facts.actions_status, facts.actions_by, facts.actions_at = "declined", "Asha Rao", AT
    out = render_document(state_with(facts), settings, final=False, now=NOW)
    text = pdf_text(out["pdf_path"])
    assert out["pdf_path"].endswith("RCA-abc123-v1-draft.pdf")
    assert "Incident report" in text and "DRAFT" in text
    assert text.count("Not provided by the user (Asha Rao, 11 Nov 2026)") >= 20
    assert "The facts available were not enough to write a summary." in text


def test_action_owner_declined_is_marked_in_the_table(settings):
    facts = IncidentFacts.empty()
    facts.actions = [ActionItem(action="Add an alert", owner_status="declined", owner_by="Asha Rao", owner_at=AT),
                     ActionItem(action="Fix CI", owner="Meera", owner_status="from_notes", due_date="2026-11-20")]
    view = build_view(state_with(facts), now=NOW, final=True)
    rows = view["actions"]["rows"]
    assert rows[0]["owner"] == {"state": "declined", "text": "Not provided by the user (Asha Rao, 11 Nov 2026)"}
    assert rows[1]["owner"]["text"] == "Meera" and rows[1]["due"] == "2026-11-20"
    assert [g["label"] for g in view["gaps"]] == ["Owner of action: Add an alert"]


def test_filenames():
    assert slugify("INC 4821 / payments") == "INC-4821-payments" and slugify("///") == "incident"
    view = {"incident_ref": "INC-4821", "version": 2, "final": True}
    assert pdf_filename(view, "abc") == "RCA-INC-4821-v2.pdf"
    assert pdf_filename({**view, "incident_ref": "", "final": False}, "abc") == "RCA-abc-v2-draft.pdf"


# ---- long-term memory ----------------------------------------------------------

def record(key: str, title: str, root_cause: str, **extra) -> dict:
    return {"incident_key": key, "incident_id": f"INC-{key}", "title": title, "severity": "Sev2",
            "services": "payments-api", "summary": title, "root_cause": root_cause, "status": "approved",
            "version": 1, "facts": {"k": key}, "drafts": {}, "notes": [], "created_by": "Asha Rao"} | extra


def test_incidents_are_saved_updated_and_listed(tmp_path):
    memory = LongTermMemory(tmp_path / "m.db")
    memory.save_incident(record("a", "Payments outage", "Config typo"), [], "2026-11-11T10:00:00")
    memory.save_incident(record("a", "Payments outage", "Config typo", version=2, created_by="Someone Else"),
                         [], "2026-11-12T09:00:00")
    saved = memory.get_incident("a")
    assert saved["version"] == 2 and saved["facts"] == {"k": "a"}
    assert (saved["created_by"], saved["created_at"]) == ("Asha Rao", "2026-11-11T10:00:00"), "kept from v1"
    assert saved["updated_at"] == "2026-11-12T09:00:00"
    assert len(memory.list_incidents()) == 1 and memory.get_incident("missing") is None


def test_declined_gaps_are_remembered_until_resolved(tmp_path):
    memory = LongTermMemory(tmp_path / "m.db")
    gaps = [{"id": "field:severity", "label": "Severity", "section": "header", "by": "Asha Rao", "at": AT},
            {"id": "owner:0", "label": "Owner of action: X", "section": "actions", "by": "Asha Rao", "at": AT}]
    memory.save_incident(record("a", "Payments outage", "Config typo"), gaps, AT)
    assert [g["gap_id"] for g in memory.open_gaps("a")] == ["field:severity", "owner:0"]
    assert memory.list_incidents()[0]["open_gaps"] == 2

    memory.save_incident(record("a", "Payments outage", "Config typo", version=2), gaps[1:], "2026-11-12T09:00:00")
    assert [g["gap_id"] for g in memory.open_gaps("a")] == ["owner:0"]
    memory.save_incident(record("a", "Payments outage", "Config typo", version=3), [], "2026-11-13T09:00:00")
    assert memory.open_gaps("a") == [] and memory.list_incidents()[0]["open_gaps"] == 0


def test_similar_incidents_are_found_by_shared_words(tmp_path):
    memory = LongTermMemory(tmp_path / "m.db")
    memory.save_incident(record("a", "Payments API outage", "Config push lowered the DB pool size"), [], AT)
    memory.save_incident(record("b", "Search latency", "Elasticsearch shard relocation",
                                services="search-api"), [], AT)
    memory.save_incident(record("c", "Checkout errors", "DB pool exhausted after a config change"), [], AT)
    found = [r["incident_key"] for r in memory.find_similar("config change reduced DB pool size on payments")]
    assert found[:2] in (["a", "c"], ["c", "a"]) and "b" not in found
    assert [r["incident_key"] for r in memory.find_similar("config pool", exclude_key="a")] == ["c"]
    assert memory.find_similar("") == [] and memory.find_similar("the and for") == []
    assert memory.find_similar('what about "quotes" AND (operators) *') == [], "search text is never run as a query"


def test_knowledge_audit_and_sessions():
    memory = LongTermMemory(":memory:")
    memory.remember("service", "payments-api", "2026-11-11T10:00:00", source="a")
    memory.remember("service", "payments-api", "2026-11-12T10:00:00", source="b")
    memory.remember("service", "  ", "2026-11-12T10:00:00")
    memory.remember("person", "Ravi Kumar", "2026-11-12T10:00:00", value="incident commander")
    assert memory.recall("service") == ["payments-api"] and memory.recall("person") == ["Ravi Kumar"]
    memory.log("a", "s1", AT, "Asha Rao", "approved", "version 1")
    assert memory.audit("a") == [{"at": AT, "actor": "Asha Rao", "event": "approved", "detail": "version 1"}]
    memory.upsert_session("s1", "a", "Payments outage", "clarifying", "Asha Rao", "2026-11-11T10:00:00")
    memory.upsert_session("s1", "a", "Payments outage", "approved", "Asha Rao", "2026-11-11T11:00:00")
    session = memory.list_sessions()[0]
    assert (session["status"], session["created_at"], session["updated_at"]) == \
        ("approved", "2026-11-11T10:00:00", "2026-11-11T11:00:00")
    assert isinstance(datetime.fromisoformat(session["updated_at"]), datetime)
