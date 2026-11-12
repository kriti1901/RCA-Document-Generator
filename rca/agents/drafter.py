"""Drafter agent: writes the narrative sections from confirmed facts, one small call per section."""

from __future__ import annotations

from rca.agents.common import DATA_NOT_INSTRUCTIONS, as_json, block, section_payload
from rca.llm import LLM
from rca.schema.facts import IncidentFacts
from rca.schema.field_specs import NARRATIVE_SECTIONS, SECTION_TITLES
from rca.schema.llm_io import SectionDraft, SummaryDraft

RULES = f"""{DATA_NOT_INSTRUCTIONS}

Rules:
1. Use only the facts given. Do not add numbers, times, names, causes or consequences that are not in them.
2. Do not mention what is unknown or missing; the document lists gaps separately.
3. Be blameless: describe systems, processes and decisions. Never fault or blame a person.
4. Plain, direct prose in the past tense. No headings, bullet points or markdown.
5. If the facts are too thin to say anything, return an empty list of paragraphs."""

SYSTEM_SECTION = f"""You write one section of a Root Cause Analysis (RCA) document from a set of confirmed facts. Write one to three short paragraphs.

{RULES}"""

SYSTEM_SUMMARY = f"""You write the executive summary of a Root Cause Analysis (RCA) document from a set of confirmed facts.

Write one paragraph of 3 to 5 sentences covering, where the facts allow: what happened, the impact, the cause, how it was resolved, and the main follow-up.
Also give `title`: a short factual title for the incident, under 12 words, built only from the facts.

{RULES}"""


def _task(section: str, payload: dict, focus: str, issues: list[str], feedback: str) -> str:
    lines = [f"Section: {SECTION_TITLES[section]}"]
    if focus:
        lines.append(f"What to emphasise: {focus}")
    if issues:
        lines.append("Fix these problems found in the previous draft:")
        lines += [f"- {issue}" for issue in issues]
    if feedback:
        lines.append("The reader asked for these changes. Apply them where they concern this section, "
                     "but still use only the facts below:")
        lines.append(block("requested_changes", feedback))
    lines.append(block("facts", as_json(payload)))
    return "\n".join(lines)


def draft(llm: LLM, facts: IncidentFacts, durations: dict[str, str], *, sections: tuple[str, ...],
          previous: dict[str, list[str]], notes: dict[str, str], issues: dict[str, list[str]],
          feedback: str = "") -> dict:
    """Write the requested sections, keeping earlier drafts of the others.

    Returns {"sections": {section: [paragraphs]}, "title": str}.
    """
    drafts = {name: list(paragraphs) for name, paragraphs in previous.items()}
    for section in NARRATIVE_SECTIONS:
        if section not in sections:
            continue
        payload = section_payload(facts, section, durations)
        if not payload:
            drafts[section] = []  # nothing known for this section; the facts table shows the gaps
            continue
        out = llm.structured("drafter", SectionDraft, SYSTEM_SECTION,
                             _task(section, payload, notes.get(section, ""), issues.get(section, []), feedback))
        drafts[section] = [p.strip() for p in out.paragraphs if p.strip()]

    title = ""
    if "summary" in sections or "summary" not in drafts:
        payload = section_payload(facts, "summary", durations)
        if payload:
            out_s = llm.structured("drafter", SummaryDraft, SYSTEM_SUMMARY,
                                   _task("summary", payload, notes.get("summary", ""),
                                         issues.get("summary", []), feedback))
            drafts["summary"] = [p.strip() for p in out_s.paragraphs if p.strip()]
            title = out_s.title.strip()
        else:
            drafts["summary"] = []
    return {"sections": drafts, "title": title}
