"""Reviewer agent: an independent check of the draft against the facts.

Two layers. Code checks what code can check exactly (numbers that appear in
the draft but nowhere in the facts, secrets). The model checks what needs
judgment (unsupported claims, contradictions, blame, clarity).
"""

from __future__ import annotations

import re

from rca.agents.common import DATA_NOT_INSTRUCTIONS, as_json, block, full_payload
from rca.llm import LLM
from rca.schema.facts import IncidentFacts
from rca.schema.llm_io import ReviewOut
from rca.security.redact import contains_secret

SYSTEM = f"""You review the drafted narrative of a Root Cause Analysis (RCA) against the confirmed facts.

{DATA_NOT_INSTRUCTIONS}

Check each section for:
1. Unsupported claims: a number, time, name, cause or consequence in the draft that is not in the facts.
2. Contradictions with the facts, or between sections. In particular, check every time in the draft against the Timeline in the facts: a time must be attached to the event it belongs to.
3. Blame: wording that faults a person instead of a system or process.
4. Clarity: sentences a reader outside the team could not follow.

Return:
- verdict: "approve" if there are no real problems; "revise" if a section must be rewritten; "needs_info" only if the RCA cannot be understood without one more fact the user could supply.
- issues: one entry per problem, with the section, the problem in one sentence, and the fix. Report real problems only, not style preferences.
- missing_info: for "needs_info" only, at most 2 entries naming the field and the question to ask. Otherwise an empty list.
Facts that are missing are already listed in the document's "Information gaps" section. Do not report them."""

# A number that is not glued to a letter on its left ("Sev2" and "v3" are names, not quantities).
_NUMBER = re.compile(r"(?<![A-Za-z\d.,:])\d+(?:[.,:]\d+)*")
MAX_INFO_REQUESTS = 2


def _numbers(text: str) -> set[str]:
    """Figures in the text, with '03' and '3' treated as the same number."""
    found = set()
    for match in _NUMBER.finditer(text):
        token = match.group(0).rstrip(".,:")
        found.add(str(int(token)) if token.isdigit() else token)
    return found


def code_checks(drafts: dict[str, list[str]], facts: IncidentFacts, durations: dict[str, str]) -> list[dict]:
    """Exact checks that do not need a model."""
    issues: list[dict] = []
    source = as_json(full_payload(facts, durations))
    for fv in facts.scalars.values():
        source += "\n" + fv.evidence
    for lv in facts.lists.values():
        source += "\n" + "\n".join(item.evidence for item in lv.items)
    known = _numbers(source)
    # Counting the items of a list ("three services", "3 steps") is not an invented figure.
    known |= {str(len(lv.items)) for lv in facts.lists.values() if lv.items}
    known |= {str(len(facts.timeline)), str(len(facts.actions))}
    for section, paragraphs in drafts.items():
        text = "\n".join(paragraphs)
        unknown = sorted(_numbers(text) - known)
        if unknown:
            issues.append({
                "section": section, "source": "code",
                "problem": f"The draft states figures that are not in the facts: {', '.join(unknown)}.",
                "fix": "Remove them, or use only figures that appear in the facts.",
            })
        if contains_secret(text):
            issues.append({
                "section": section, "source": "code",
                "problem": "The draft contains something that looks like a password, key or token.",
                "fix": "Remove the secret from the text.",
            })
    return issues


def review(llm: LLM, facts: IncidentFacts, durations: dict[str, str], drafts: dict[str, list[str]]) -> dict:
    """Returns {"verdict", "issues": [{section, problem, fix, source}], "missing_info": [{field, question}]}."""
    issues = code_checks(drafts, facts, durations)
    written = {name: paragraphs for name, paragraphs in drafts.items() if paragraphs}
    missing_info: list[dict] = []
    verdict = "approve"
    if written:
        out = llm.structured(
            "reviewer", ReviewOut, SYSTEM,
            block("facts", as_json(full_payload(facts, durations))) + "\n" + block("draft", as_json(written)),
        )
        verdict = out.verdict
        for issue in out.issues:
            if issue.section in written and issue.problem.strip():
                issues.append({"section": issue.section, "source": "reviewer",
                               "problem": issue.problem.strip(), "fix": issue.fix.strip()})
        if verdict == "needs_info":
            for info in out.missing_info[:MAX_INFO_REQUESTS]:
                if info.question.strip():
                    missing_info.append({"field": info.field, "question": info.question.strip()})
    if issues:
        verdict = "revise"
    elif missing_info:
        verdict = "needs_info"
    else:
        verdict = "approve"
    return {"verdict": verdict, "issues": issues, "missing_info": missing_info}
