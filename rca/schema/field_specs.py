"""The RCA layout as data.

This registry is the "output document schema": which fields the RCA has, which
section each belongs to, when it is required, and how to ask the user for it.
To match a different company template, edit this file and the HTML template;
the agents read everything they need from here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Section id -> heading, in document order.
SECTIONS: tuple[tuple[str, str], ...] = (
    ("summary", "Executive summary"),
    ("impact", "Impact"),
    ("timeline", "Timeline"),
    ("detection", "Detection"),
    ("response", "Response and resolution"),
    ("root_cause", "Root cause analysis"),
    ("lessons", "Lessons learned"),
    ("actions", "Corrective and preventive actions"),
    ("related", "Related documents"),
    ("gaps", "Information gaps"),
    ("revision_history", "Revision history"),
)
SECTION_TITLES = dict(SECTIONS) | {"header": "Incident overview"}

# Sections that get narrative text from the Drafter agent.
NARRATIVE_SECTIONS = ("impact", "detection", "response", "root_cause")


@dataclass(frozen=True)
class FieldSpec:
    id: str
    label: str
    section: str
    kind: str  # "scalar" or "list"
    required: str  # "always", "sev1_2", "optional" or "generated"
    group: str  # extraction call this field belongs to: "overview" or "analysis"
    priority: float  # asking order; lower is asked first
    ask: str  # question put to the user when the field is missing
    hint: str  # what the Extractor should look for
    validator: str = ""  # "severity", "datetime", "date" or ""


FIELDS: tuple[FieldSpec, ...] = (
    # ---- header ----------------------------------------------------------
    FieldSpec(
        "title", "Incident title", "header", "scalar", "generated", "overview", 99,
        ask="What short title should this incident have?",
        hint=("A title only if the notes explicitly label one, for example with 'Title:' or 'Subject:'. "
              "Do not use a channel name, a file name or a heading such as 'notes'."),
    ),
    FieldSpec(
        "incident_id", "Incident ID", "header", "scalar", "always", "overview", 14,
        ask="What is the incident ID or ticket number (for example INC-1234)?",
        hint="Incident or ticket identifier such as INC-1234.",
    ),
    FieldSpec(
        "severity", "Severity", "header", "scalar", "always", "overview", 1,
        ask="What was the severity (Sev1, Sev2, Sev3 or Sev4)?",
        hint="Severity or priority level as stated (Sev1-Sev4, P1-P4, critical/high/medium/low).",
        validator="severity",
    ),
    FieldSpec(
        "incident_date", "Incident date", "header", "scalar", "always", "overview", 2.5,
        ask="What was the date of the incident (day, month and year)?",
        hint=("The calendar date of the incident, only if the notes state it. Write YYYY-MM-DD when the "
              "year is stated, or the day and month (for example 5 Nov) when it is not."),
        validator="date",
    ),
    FieldSpec(
        "incident_commander", "Incident commander", "header", "scalar", "always", "overview", 13,
        ask="Who was the incident commander (the person leading the response)?",
        hint="Name of the person who led or coordinated the incident response.",
    ),
    FieldSpec(
        "affected_services", "Affected services", "header", "list", "always", "overview", 7,
        ask="Which services or systems were affected?",
        hint="Each service, system or component that was affected. One fact per service.",
    ),
    # ---- impact ----------------------------------------------------------
    FieldSpec(
        "customer_impact", "Customer impact", "impact", "scalar", "always", "overview", 6,
        ask="What did customers or users experience during the incident?",
        hint="What users or customers experienced (errors, slowness, feature unavailable).",
    ),
    FieldSpec(
        "impact_scale", "Scale of impact", "impact", "scalar", "always", "overview", 11,
        ask="How many users, requests or regions were affected (a number or an estimate)?",
        hint="How many users, requests, orders or regions were affected, as a number or share.",
    ),
    FieldSpec(
        "business_impact", "Business or SLA impact", "impact", "scalar", "sev1_2", "overview", 19,
        ask="Was there business impact, such as an SLA breach, lost revenue or support tickets?",
        hint="SLA or SLO breach, lost revenue, refunds, support tickets or reputational impact.",
    ),
    FieldSpec(
        "data_loss", "Data loss or exposure", "impact", "scalar", "always", "overview", 17,
        ask="Was any data lost, corrupted or exposed? (Answer 'none' if not.)",
        hint="Whether data was lost, corrupted or exposed, including an explicit statement that none was.",
    ),
    # ---- timeline milestones ---------------------------------------------
    FieldSpec(
        "started_at", "Incident start", "timeline", "scalar", "always", "overview", 2,
        ask="When did the incident start (date and time)?",
        hint="When the impact began for users or systems.",
        validator="datetime",
    ),
    FieldSpec(
        "detected_at", "Detected", "timeline", "scalar", "always", "overview", 3,
        ask="When was the incident detected?",
        hint=("When the responding team first became aware of the problem: the first alert, report or "
              "observation that reached them. Not when customers first experienced it."),
        validator="datetime",
    ),
    FieldSpec(
        "mitigated_at", "Mitigated", "timeline", "scalar", "always", "overview", 16,
        ask="When was the impact mitigated (when did things start working again)?",
        hint="When the impact was first reduced or stopped (rollback done, failover complete).",
        validator="datetime",
    ),
    FieldSpec(
        "resolved_at", "Resolved", "timeline", "scalar", "always", "overview", 4,
        ask="When was the incident fully resolved?",
        hint="When the incident was fully resolved or closed.",
        validator="datetime",
    ),
    FieldSpec(
        "timezone", "Timezone", "timeline", "scalar", "always", "overview", 5,
        ask="Which timezone are the times in (for example UTC, IST or Europe/Berlin)?",
        hint="The timezone the notes use for their times, only if the notes state one.",
    ),
    # ---- detection -------------------------------------------------------
    FieldSpec(
        "detection_method", "How it was detected", "detection", "scalar", "always", "overview", 12,
        ask="How was the incident detected (an alert, a customer report, someone noticed)?",
        hint=("What first made the team aware: which alert, monitor, customer report or person. It must "
              "describe the same event as detected_at, not a later alert."),
    ),
    FieldSpec(
        "monitoring_gaps", "Monitoring gaps", "detection", "scalar", "optional", "analysis", 30,
        ask="Were there monitoring or alerting gaps that delayed detection?",
        hint="Missing, late or noisy alerts or monitoring that delayed detection.",
    ),
    # ---- response --------------------------------------------------------
    FieldSpec(
        "mitigation_steps", "Mitigation steps", "response", "list", "always", "analysis", 15,
        ask="What steps were taken to mitigate the incident?",
        hint="Each action taken to reduce or stop the impact. One fact per step.",
    ),
    FieldSpec(
        "fix_applied", "Fix applied", "response", "scalar", "always", "analysis", 10,
        ask="What finally fixed the problem?",
        hint="The change that resolved the incident (rollback, patch, config fix, restart).",
    ),
    FieldSpec(
        "responders", "People and teams involved", "response", "list", "optional", "analysis", 31,
        ask="Which people or teams took part in the response?",
        hint="Each person or team named as taking part in the response. One fact per name.",
    ),
    # ---- root cause ------------------------------------------------------
    FieldSpec(
        "trigger", "Trigger", "root_cause", "scalar", "always", "analysis", 9,
        ask="What event triggered the incident (a deploy, config change, traffic spike, hardware failure)?",
        hint="The event that set the incident off (a deploy, config change, traffic spike, failure).",
    ),
    FieldSpec(
        "root_cause", "Root cause", "root_cause", "scalar", "always", "analysis", 8,
        ask="What was the underlying root cause?",
        hint="The underlying cause as stated in the notes, not just the symptom.",
    ),
    FieldSpec(
        "contributing_factors", "Contributing factors", "root_cause", "list", "optional", "analysis", 32,
        ask="Were there contributing factors that made the incident more likely or worse?",
        hint="Each factor that made the incident more likely or worse. One fact per factor.",
    ),
    FieldSpec(
        "five_whys", "5 Whys", "root_cause", "list", "sev1_2", "analysis", 20,
        ask="Walk through the 'why' chain: why did the root cause happen, and why that, up to five levels?",
        hint="Each step of a stated why-chain or causal chain, in order. One fact per step.",
    ),
    # ---- lessons ---------------------------------------------------------
    FieldSpec(
        "went_well", "What went well", "lessons", "list", "sev1_2", "analysis", 21,
        ask="What went well during the response?",
        hint=("Only things the notes explicitly describe as having gone well. Do not derive them from "
              "the timeline or from the outcome. One fact per item."),
    ),
    FieldSpec(
        "went_wrong", "What went wrong", "lessons", "list", "sev1_2", "analysis", 22,
        ask="What went wrong or was harder than it should have been?",
        hint=("Only things the notes explicitly describe as having gone badly or been harder than they "
              "should be. Do not derive them from the timeline. One fact per item."),
    ),
    FieldSpec(
        "got_lucky", "Where we got lucky", "lessons", "list", "optional", "analysis", 33,
        ask="Was there anywhere you got lucky?",
        hint="Each thing the notes describe as luck or a near miss. One fact per item.",
    ),
)

BY_ID: dict[str, FieldSpec] = {f.id: f for f in FIELDS}
SCALAR_IDS: tuple[str, ...] = tuple(f.id for f in FIELDS if f.kind == "scalar")
LIST_IDS: tuple[str, ...] = tuple(f.id for f in FIELDS if f.kind == "list")
MILESTONE_IDS: tuple[str, ...] = ("started_at", "detected_at", "mitigated_at", "resolved_at")

# Gaps that are not plain fields.
ACTIONS_GAP_PRIORITY = 10
ACTION_OWNER_PRIORITY = 18
ACTIONS_QUESTION = (
    "What follow-up actions were agreed? For each one, give the action, its owner and a due date."
)

# Fields the Planner may challenge when the stated value is too vague to be useful.
QUALITY_CHECK_IDS: tuple[str, ...] = (
    "root_cause", "trigger", "customer_impact", "impact_scale", "fix_applied", "detection_method",
)


def spec(field_id: str) -> FieldSpec:
    return BY_ID[field_id]


def fields_in_group(group: str) -> tuple[FieldSpec, ...]:
    return tuple(f for f in FIELDS if f.group == group)


def fields_in_section(section: str) -> tuple[FieldSpec, ...]:
    return tuple(f for f in FIELDS if f.section == section)


_SEVERITY_WORDS = {
    "critical": "Sev1", "blocker": "Sev1", "highest": "Sev1",
    "high": "Sev2", "major": "Sev2",
    "medium": "Sev3", "moderate": "Sev3", "minor": "Sev3",
    "low": "Sev4", "lowest": "Sev4", "trivial": "Sev4",
}


def normalize_severity(text: str) -> str:
    """Map 'sev 2', 'SEV-2', 'P2', 'high' and similar to 'Sev1'..'Sev4'. Returns '' if unclear."""
    lowered = text.strip().lower()
    match = re.search(r"\b(?:sev(?:erity)?|p|priority)\s*[-:#]?\s*([0-4])\b", lowered)
    if match is None:
        match = re.fullmatch(r"\s*([1-4])\s*", lowered)
    if match:
        level = max(1, int(match.group(1)))  # Sev0 / P0 is treated as the top level
        return f"Sev{level}"
    for word, level in _SEVERITY_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            return level
    return ""


def is_required(field: FieldSpec, severity: str) -> bool:
    """Whether a field must be filled (or explicitly declined) for this severity."""
    if field.required == "always":
        return True
    if field.required == "sev1_2":
        return severity in {"Sev1", "Sev2"}
    return False
