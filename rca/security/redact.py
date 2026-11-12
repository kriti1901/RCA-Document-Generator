"""Mask secrets in pasted text before it is stored, sent to the model or published.

This is a best-effort pattern match, not a guarantee: it catches common key and
token formats and `password=...` style assignments. It cannot recognise a
secret that looks like an ordinary word.
"""

from __future__ import annotations

import re

MASK = "[REDACTED]"

_WHOLE: tuple[re.Pattern[str], ...] = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),  # GitHub token
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),  # Slack token
    re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b"),  # Groq key
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),  # OpenAI-style key
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),  # JWT
)

# "Bearer <token>": keep the word, mask the token.
_BEARER = re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9\-._~+/]{16,}=*)")

# scheme://user:password@host : keep the user, mask the password.
_URL_CREDENTIALS = re.compile(r"(?<=://)([^\s:/@]+):([^\s@/]+)(?=@)")

# password = hunter2, api_key: "abc123...", token=...  The value must look like a
# secret (quoted, or 6+ characters containing a digit or symbol) so that ordinary
# prose such as "token: expired" is left alone.
_ASSIGNMENT = re.compile(
    r"(?i)\b((?:pass(?:word|wd)?|pwd|secret|api[_-]?key|access[_-]?key|private[_-]?key"
    r"|auth[_-]?token|access[_-]?token|client[_-]?secret|token)\s*[:=]\s*)"
    r"(\"[^\"\n]+\"|'[^'\n]+'|(?=\S*[\d_\-+/=@#$%^&*!])\S{6,})"
)


def redact(text: str) -> tuple[str, int]:
    """Return the text with secrets masked, and how many were masked."""
    count = 0

    def whole(_: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return MASK

    def keep_prefix(match: re.Match[str]) -> str:
        nonlocal count
        if match.group(2) == MASK:
            return match.group(0)
        count += 1
        return match.group(1) + MASK

    def url(match: re.Match[str]) -> str:
        nonlocal count
        if match.group(2) == MASK:  # already masked on an earlier pass
            return match.group(0)
        count += 1
        return f"{match.group(1)}:{MASK}"

    for pattern in _WHOLE:
        text = pattern.sub(whole, text)
    text = _BEARER.sub(keep_prefix, text)
    text = _URL_CREDENTIALS.sub(url, text)
    text = _ASSIGNMENT.sub(keep_prefix, text)
    return text, count


def contains_secret(text: str) -> bool:
    return redact(text)[1] > 0
