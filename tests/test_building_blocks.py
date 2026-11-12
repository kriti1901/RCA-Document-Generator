"""The field registry, gap detection, secret masking, evidence check and time handling."""

from __future__ import annotations

import pytest

from rca.gaps import MAX_ASKS_PER_GAP, declined_gaps, open_gaps, select_batch
from rca.schema import field_specs as fs
from rca.schema.facts import ActionItem, FieldValue, IncidentFacts, ListItem, ListValue
from rca.schema.llm_io import facts_schema_for
from rca.security.evidence import mentions, supported
from rca.security.redact import redact
from rca.timeutil import clean_time_value, compute_milestones, describe_date, resolve_timezone


# ---- registry ------------------------------------------------------------------

def test_registry_is_consistent():
    ids = [f.id for f in fs.FIELDS]
    assert len(ids) == len(set(ids))
    sections = set(fs.SECTION_TITLES)
    for field in fs.FIELDS:
        assert field.section in sections
        assert field.kind in {"scalar", "list"}
        assert field.required in {"always", "sev1_2", "optional", "generated"}
        assert field.group in {"overview", "analysis"}
        assert field.ask.endswith("?") or field.ask.endswith(")")
    facts = IncidentFacts.empty()
    assert set(facts.scalars) == set(fs.SCALAR_IDS) and set(facts.lists) == set(fs.LIST_IDS)


@pytest.mark.parametrize("text,expected", [
    ("sev 2", "Sev2"), ("SEV-1", "Sev1"), ("P3", "Sev3"), ("severity: 4", "Sev4"), ("2", "Sev2"),
    ("high", "Sev2"), ("critical outage", "Sev1"), ("P0", "Sev1"), ("banana", ""), ("", ""),
])
def test_severity_normalisation(text, expected):
    assert fs.normalize_severity(text) == expected


def test_extraction_schema_rejects_unknown_fields():
    schema = facts_schema_for(("severity", "title"))
    schema.model_validate({"facts": [{"field": "severity", "value": "Sev2", "evidence": "sev 2"}]})
    with pytest.raises(ValueError):
        schema.model_validate({"facts": [{"field": "made_up", "value": "x", "evidence": "y"}]})
    # strict structured-output modes need every object closed and every key required
    assert schema.model_json_schema()["additionalProperties"] is False


# ---- gap detection -------------------------------------------------------------

def test_gaps_depend_on_severity():
    facts = IncidentFacts.empty()
    ids = {g["id"] for g in open_gaps(facts, [], {})}
    assert "field:severity" in ids and "field:root_cause" in ids and "actions" in ids
    assert "field:five_whys" not in ids, "Sev1-2 fields are not asked for until the severity is known"
    assert "field:title" not in ids, "the title is generated, never asked for"
    assert "field:monitoring_gaps" not in ids, "optional fields are never asked for"

    facts.scalars["severity"] = FieldValue(value="Sev1", status="from_user")
    ids = {g["id"] for g in open_gaps(facts, [], {})}
    assert {"field:five_whys", "field:went_well", "field:business_impact"} <= ids

    facts.scalars["severity"] = FieldValue(value="Sev4", status="from_user")
    assert "field:five_whys" not in {g["id"] for g in open_gaps(facts, [], {})}


def test_declined_fields_are_not_asked_again_and_are_reported():
    facts = IncidentFacts.empty()
    facts.scalars["severity"] = FieldValue(status="declined", by="Asha", at="2026-11-11T10:00:00")
    facts.lists["affected_services"] = ListValue(status="declined", by="Asha", at="2026-11-11T10:00:00")
    ids = {g["id"] for g in open_gaps(facts, [], {})}
    assert "field:severity" not in ids and "field:affected_services" not in ids
    reported = {g["id"]: g for g in declined_gaps(facts, [])}
    assert reported["field:severity"]["by"] == "Asha"
    assert reported["field:affected_services"]["label"] == "Affected services"


def test_incident_date_is_asked_for_and_a_year_is_never_assumed():
    def ids(facts, log=None):
        return [g["id"] for g in open_gaps(facts, [], log or {})]

    facts = IncidentFacts.empty()
    assert "field:incident_date" in ids(facts), "no date anywhere: ask for it"

    facts.scalars["started_at"] = FieldValue(value="2026-11-03 14:05", status="from_notes")
    assert "field:incident_date" not in ids(facts) and "quality:incident_date" not in ids(facts), \
        "the start time already gives the full date"

    facts.scalars["started_at"] = FieldValue(value="5 Nov 09:40", status="from_notes")
    assert "field:incident_date" not in ids(facts)
    year_gap = next(g for g in open_gaps(facts, [], {}) if g["id"] == "quality:incident_date")
    assert 'the notes say: "5 Nov"' in year_gap["question"] and "year" in year_gap["question"]
    assert "quality:incident_date" not in ids(facts, {"quality:incident_date": {"asked": 1, "outcome": "declined"}})

    facts.scalars["incident_date"] = FieldValue(value="5 November 2026", status="from_user")
    assert "quality:incident_date" not in ids(facts), "answered: the year is now known"


def test_action_owner_gaps():
    facts = IncidentFacts.empty()
    facts.actions = [ActionItem(action="Add an alert"), ActionItem(action="Fix CI", owner="Meera", owner_status="from_notes")]
    facts.actions_status = "from_notes"
    gaps = {g["id"]: g for g in open_gaps(facts, [], {})}
    assert "actions" not in gaps
    assert "owner:0" in gaps and "owner:1" not in gaps
    assert "Add an alert" in gaps["owner:0"]["question"]


def test_gaps_are_ordered_and_the_final_round_asks_everything():
    gaps = open_gaps(IncidentFacts.empty(), [], {})
    assert gaps[0]["id"] == "field:severity"
    assert [g["priority"] for g in gaps] == sorted(g["priority"] for g in gaps)
    assert len(select_batch(gaps, 0, 5, 3)) == 5
    assert len(select_batch(gaps, 1, 5, 3)) == 5
    assert select_batch(gaps, 2, 5, 3) == gaps, "the last round must ask every remaining question"
    assert MAX_ASKS_PER_GAP == 2


# ---- secret masking ------------------------------------------------------------

def test_redaction_masks_secrets_and_leaves_prose():
    text = ("db password = hunter2!x and token: expired yesterday\n"
            "aws AKIAABCDEFGHIJKLMNOP auth: Bearer abcdefghijklmnopqrstuvwxyz012345\n"
            "postgres://app:s3cr3tpw@db.internal:5432/prod\n"
            "root cause: token validation bug")
    masked, count = redact(text)
    assert count == 4
    for secret in ("hunter2!x", "AKIAABCDEFGHIJKLMNOP", "abcdefghijklmnopqrstuvwxyz012345", "s3cr3tpw"):
        assert secret not in masked
    assert "token: expired yesterday" in masked and "root cause: token validation bug" in masked
    assert "postgres://app:[REDACTED]@db.internal" in masked
    assert redact(masked) == (masked, 0), "masking twice changes nothing"


# ---- evidence check ------------------------------------------------------------

NOTES = "14:05 payments API started returning 500s. Alert fired at 14:12 — on-call (Ravi) rolled back the deploy at 14:40."


@pytest.mark.parametrize("quote", [
    "payments API started returning 500s",
    "Payments api  started returning 500s.",  # case, spacing, trailing punctuation
    "alert fired at 14:12",
    "payments API started returning 500s ... rolled back the deploy",  # stitched with an ellipsis
    "on-call Ravi rolled back the deploy at 14:40",  # a small copying slip
    "14:05",
])
def test_real_quotes_are_accepted(quote):
    assert supported(quote, NOTES)


@pytest.mark.parametrize("quote", [
    "the database crashed at 14:05",
    "payments API started returning 500s ... and 2,000 customers complained",
    "", "   ", "x",
])
def test_invented_quotes_are_rejected(quote):
    assert not supported(quote, NOTES)


def test_mentions():
    assert mentions("Ravi", NOTES) and not mentions("Priya", NOTES)


# ---- time handling -------------------------------------------------------------

def test_durations_and_utc_conversion():
    ms = compute_milestones({"started_at": "2026-11-03 14:05", "detected_at": "14:12",
                             "mitigated_at": "14:40", "resolved_at": "15:10"}, "IST", default_year=2026)
    assert ms.durations == {"time_to_detect": "7 minutes", "time_to_mitigate": "35 minutes",
                            "time_to_resolve": "1 hour 5 minutes"}
    assert ms.points["detected_at"].local_text == "2026-11-03 14:12", "a time inherits the incident date"
    assert ms.points["started_at"].utc_text == "2026-11-03 08:35 UTC"
    assert "Asia/Kolkata" in ms.zone.label, "the document says which zone an abbreviation was taken as"


def test_only_stated_parts_of_a_date_are_shown():
    values = {"started_at": "5 Nov 09:40", "detected_at": "10:05", "mitigated_at": "", "resolved_at": "10:50"}
    ms = compute_milestones(values, "UTC", default_year=2026)
    assert ms.points["started_at"].local_text == "5 Nov 09:40", "no year was stated, so none is printed"
    assert ms.points["resolved_at"].utc_text == "5 Nov 10:50 UTC"
    assert ms.durations["time_to_resolve"] == "1 hour 10 minutes"

    ms = compute_milestones(values, "UTC", default_year=2026, base_date="5 November 2025")
    assert ms.points["started_at"].local_text == "2025-11-05 09:40", "the year comes from the incident date"
    assert ms.points["detected_at"].local_text == "2025-11-05 10:05"

    notes = "thurs 5 nov, ~09:40 UTC"
    assert clean_time_value("2023-11-05 09:40", notes) == "5 Nov 09:40"
    assert clean_time_value("2026-11-12 09:40", notes) == "09:40"
    assert clean_time_value("2023-11-05", "nothing dated here") == ""
    assert clean_time_value("2026-11-03 14:05", "export of 3 Nov 2026") == "2026-11-03 14:05"
    assert describe_date("2026-11-03 14:05") == ("3 Nov 2026", True)
    assert describe_date("5 Nov 09:40") == ("5 Nov", False) and describe_date("09:40") == ("", False)


def test_times_without_a_date_and_past_midnight():
    ms = compute_milestones({"started_at": "23:50", "detected_at": "23:58", "mitigated_at": "",
                             "resolved_at": "00:40"}, "", default_year=2026)
    assert ms.durations == {"time_to_detect": "8 minutes", "time_to_resolve": "50 minutes"}
    assert ms.points["started_at"].local_text == "23:50" and not ms.converted


def test_unparseable_times_are_kept_as_written():
    ms = compute_milestones({"started_at": "14:05", "detected_at": "shortly after", "mitigated_at": "",
                             "resolved_at": "15:10"}, "UTC", default_year=2026)
    assert ms.points["detected_at"].local_text == "shortly after"
    assert "time_to_detect" not in ms.durations and ms.durations["time_to_resolve"] == "1 hour 5 minutes"
    assert ms.points["started_at"].utc_text == "14:05 UTC", "a fixed offset converts even without a date"


@pytest.mark.parametrize("text,label,known", [
    ("UTC", "UTC", True), ("utc+5:30", "UTC+05:30", True), ("GMT-8", "UTC-08:00", True),
    ("Europe/Berlin", "Europe/Berlin", True), ("local time", "local time", False), ("", "", False),
])
def test_timezones(text, label, known):
    zone = resolve_timezone(text)
    assert zone.label == label and (zone.tz is not None) == known


def test_list_helpers_round_trip():
    facts = IncidentFacts.empty()
    facts.lists["affected_services"] = ListValue(items=[ListItem(value="payments-api")], status="from_notes")
    again = IncidentFacts.load(facts.model_dump())
    assert again == facts and again.provided_count() == 1
