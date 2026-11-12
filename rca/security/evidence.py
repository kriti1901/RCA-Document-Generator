"""Check that a quote an agent gives as evidence really appears in the source text.

This is the main guard against invented facts: a fact whose quote cannot be
found in the notes is dropped and treated as missing, so the user gets asked
about it instead of the document containing something nobody wrote.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_TRANSLATE = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
})
_ELLIPSIS = re.compile(r"\s*(?:\.{3,}|…)\s*")
_EDGE = " \t\n\r\"'`.,;:!?()[]{}<>*_-"


def normalize(text: str) -> str:
    text = text.translate(_TRANSLATE).lower()
    return re.sub(r"\s+", " ", text).strip()


def _fragment_supported(fragment: str, source: str, source_tokens: list[str], threshold: float) -> bool:
    fragment = fragment.strip(_EDGE)
    if len(fragment) < 2:
        return False
    if fragment in source:
        return True
    tokens = fragment.split()
    if len(tokens) < 3:
        return False  # short quotes must match exactly
    wanted = set(tokens)
    size = len(tokens)
    for width in (size, size - 1, size + 1):
        if width < 1 or width > len(source_tokens):
            continue
        for start in range(len(source_tokens) - width + 1):
            window = source_tokens[start:start + width]
            if len(wanted.intersection(window)) * 2 < len(wanted):
                continue
            if SequenceMatcher(None, " ".join(window), fragment).ratio() >= threshold:
                return True
    return False


def supported(quote: str, source: str, threshold: float = 0.9) -> bool:
    """True if `quote` appears in `source`, allowing small copying differences.

    Case, whitespace and typographic quotes are ignored. A quote the model
    stitched together with "..." passes only if every piece is found.
    """
    quote_n, source_n = normalize(quote), normalize(source)
    if not quote_n or not source_n:
        return False
    fragments = [f for f in _ELLIPSIS.split(quote_n) if f.strip(_EDGE)]
    if not fragments:
        return False
    source_tokens = source_n.split()
    return all(_fragment_supported(f, source_n, source_tokens, threshold) for f in fragments)


def mentions(value: str, source: str) -> bool:
    """True if `value` (a name, for example) literally occurs in `source`."""
    value_n = normalize(value).strip(_EDGE)
    return bool(value_n) and value_n in normalize(source)
