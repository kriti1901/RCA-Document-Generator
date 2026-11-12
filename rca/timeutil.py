"""Time handling done in code, never by the model: parsing, durations, UTC conversion."""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import parser as dtparser

# Common abbreviations. Several are ambiguous worldwide (IST, CST, BST); the
# document states which zone was assumed whenever an abbreviation is used.
_ABBREVIATIONS = {
    "UTC": "UTC", "GMT": "UTC", "Z": "UTC", "ZULU": "UTC",
    "IST": "Asia/Kolkata",
    "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles", "PT": "America/Los_Angeles",
    "MST": "America/Denver", "MDT": "America/Denver", "MT": "America/Denver",
    "CST": "America/Chicago", "CDT": "America/Chicago", "CT": "America/Chicago",
    "EST": "America/New_York", "EDT": "America/New_York", "ET": "America/New_York",
    "BST": "Europe/London", "WET": "Europe/Lisbon", "WEST": "Europe/Lisbon",
    "CET": "Europe/Berlin", "CEST": "Europe/Berlin",
    "EET": "Europe/Helsinki", "EEST": "Europe/Helsinki",
    "GST": "Asia/Dubai", "PKT": "Asia/Karachi", "SGT": "Asia/Singapore", "HKT": "Asia/Hong_Kong",
    "JST": "Asia/Tokyo", "KST": "Asia/Seoul",
    "AEST": "Australia/Sydney", "AEDT": "Australia/Sydney", "AWST": "Australia/Perth",
    "NZST": "Pacific/Auckland", "NZDT": "Pacific/Auckland",
}
_OFFSET = re.compile(r"(?:UTC|GMT)?\s*([+-])\s*(\d{1,2})(?::?(\d{2}))?\b", re.I)
_TIME_OF_DAY = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}(?:\.\d{2})?\s*(?:am|pm)\b", re.I)
_PLACEHOLDER = datetime(2000, 1, 1)


@dataclass(frozen=True)
class ResolvedZone:
    tz: tzinfo | None
    label: str  # how to describe it in the document
    fixed: bool = False  # True when the UTC offset does not depend on the date


def resolve_timezone(text: str) -> ResolvedZone:
    """Turn 'UTC', 'IST', 'UTC+05:30' or 'Europe/Berlin' into a tzinfo. tz is None if unclear."""
    raw = (text or "").strip()
    if not raw:
        return ResolvedZone(None, "")
    for token in re.findall(r"[A-Za-z_]+/[A-Za-z_+\-]+(?:/[A-Za-z_]+)?", raw):
        try:
            return ResolvedZone(ZoneInfo(token), token)
        except (ZoneInfoNotFoundError, ValueError):
            continue
    for token in re.findall(r"[A-Za-z]+", raw):
        zone = _ABBREVIATIONS.get(token.upper())
        if zone == "UTC":
            offset = _OFFSET.search(raw)
            if offset and (offset.group(2) != "0" or offset.group(3)):
                break  # "UTC+5:30" is handled as an offset below
            return ResolvedZone(timezone.utc, "UTC", fixed=True)
        if zone:
            return ResolvedZone(ZoneInfo(zone), f"{token.upper()}, taken as {zone}")
    offset = _OFFSET.search(raw)
    if offset:
        hours, minutes = int(offset.group(2)), int(offset.group(3) or 0)
        if hours <= 14 and minutes < 60:
            delta = timedelta(hours=hours, minutes=minutes)
            sign = offset.group(1)
            tz = timezone(delta if sign == "+" else -delta)
            return ResolvedZone(tz, f"UTC{sign}{hours:02d}:{minutes:02d}", fixed=True)
    return ResolvedZone(None, raw)


@dataclass(frozen=True)
class ParsedTime:
    raw: str
    hour: int | None = None
    minute: int | None = None
    month: int | None = None
    day: int | None = None
    year: int | None = None

    @property
    def has_time(self) -> bool:
        return self.hour is not None

    @property
    def has_day(self) -> bool:
        return self.month is not None


def parse_time(text: str, *, dayfirst: bool = False) -> ParsedTime:
    """Parse what the text states and nothing more: a missing date or year stays None."""
    raw = (text or "").strip()
    if not raw or not re.search(r"\d", raw):
        return ParsedTime(raw)
    try:
        a = dtparser.parse(raw, default=datetime(2000, 1, 1), fuzzy=True, ignoretz=True, dayfirst=dayfirst)
        b = dtparser.parse(raw, default=datetime(2001, 2, 2), fuzzy=True, ignoretz=True, dayfirst=dayfirst)
    except (ValueError, OverflowError, TypeError):
        return ParsedTime(raw)
    has_time = bool(_TIME_OF_DAY.search(raw))
    has_day = (a.month, a.day) == (b.month, b.day)
    has_year = a.year == b.year
    return ParsedTime(
        raw=raw,
        hour=a.hour if has_time else None,
        minute=a.minute if has_time else None,
        month=a.month if has_day else None,
        day=a.day if has_day else None,
        year=a.year if (has_day and has_year) else None,
    )


@dataclass
class TimePoint:
    raw: str
    local: datetime | None = None  # naive, in the notes' timezone
    date_known: bool = False  # the day and month were stated
    year_known: bool = False  # the year was stated too
    utc: datetime | None = None

    def _format(self, moment: datetime) -> str:
        """Show only what was stated: a year or date nobody wrote is never printed."""
        if self.date_known and self.year_known:
            return moment.strftime("%Y-%m-%d %H:%M")
        if self.date_known:
            return f"{moment.day} {moment.strftime('%b %H:%M')}"
        return moment.strftime("%H:%M")

    @property
    def local_text(self) -> str:
        return self.raw if self.local is None else self._format(self.local)

    @property
    def utc_text(self) -> str:
        return "" if self.utc is None else self._format(self.utc) + " UTC"


@dataclass
class Sequence:
    """A chronological run of times that share a base date (milestones, timeline events)."""

    zone: ResolvedZone
    default_year: int
    dayfirst: bool = False
    _previous: datetime | None = field(default=None, repr=False)
    _date_known: bool = field(default=False, repr=False)
    _year_known: bool = field(default=False, repr=False)

    def seed(self, texts: list[str]) -> None:
        """Take the base date from the texts: one that states the year if any does, else the first date."""
        parsed_texts = [parse_time(text, dayfirst=self.dayfirst) for text in texts]
        dated = [p for p in parsed_texts if p.has_day]
        if not dated:
            return
        best = next((p for p in dated if p.year is not None), dated[0])
        try:
            self._previous = datetime(best.year or self.default_year, best.month, best.day)  # type: ignore[arg-type]
        except ValueError:  # a date that does not exist, such as 30 Feb
            return
        self._date_known = True
        self._year_known = best.year is not None

    def add(self, text: str, *, advance: bool = True) -> TimePoint:
        parsed = parse_time(text, dayfirst=self.dayfirst)
        point = TimePoint(raw=(text or "").strip())
        if not parsed.has_time:
            return point
        year_known = self._year_known
        if parsed.has_day:
            # A day and month without a year belong to the incident's year when that is known.
            inherited = self._previous.year if (self._year_known and self._previous) else self.default_year
            try:
                base = datetime(parsed.year or inherited, parsed.month, parsed.day)  # type: ignore[arg-type]
            except ValueError:
                return point
            local, known = base.replace(hour=parsed.hour, minute=parsed.minute), True
            year_known = parsed.year is not None or self._year_known
        else:
            anchor = self._previous or _PLACEHOLDER
            local = anchor.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
            # A time earlier than the one before it means the clock passed midnight.
            if advance and self._previous is not None and local < self._previous:
                local += timedelta(days=1)
            known = self._date_known
        point.local, point.date_known, point.year_known = local, known, known and year_known
        if advance:
            self._previous = local
        if self.zone.tz is not None and (known or self.zone.fixed):
            point.utc = local.replace(tzinfo=self.zone.tz).astimezone(timezone.utc)
        return point


def humanize(delta: timedelta) -> str:
    minutes = int(round(delta.total_seconds() / 60))
    if minutes < 1:
        return "under 1 minute"
    days, rest = divmod(minutes, 1440)
    hours, mins = divmod(rest, 60)
    parts = []
    if days:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if mins and not days:
        parts.append(f"{mins} minute{'s' if mins != 1 else ''}")
    return " ".join(parts)


@dataclass
class Milestones:
    points: dict[str, TimePoint]
    zone: ResolvedZone
    durations: dict[str, str]  # "time_to_detect" -> "7 minutes"

    @property
    def converted(self) -> bool:
        return any(p.utc is not None for p in self.points.values())


_DURATIONS = (
    ("time_to_detect", "started_at", "detected_at"),
    ("time_to_mitigate", "started_at", "mitigated_at"),
    ("time_to_resolve", "started_at", "resolved_at"),
)
DURATION_LABELS = {
    "time_to_detect": "Time to detect",
    "time_to_mitigate": "Time to mitigate",
    "time_to_resolve": "Time to resolve",
}


def compute_milestones(values: dict[str, str], timezone_text: str, *, default_year: int,
                       dayfirst: bool = False, base_date: str = "") -> Milestones:
    """Parse the four milestones in order and work out the durations between them.

    `base_date` is the incident date when it is known separately from the times.
    `default_year` is used only for arithmetic when no year was stated; it is never displayed.
    """
    zone = resolve_timezone(timezone_text)
    order = ("started_at", "detected_at", "mitigated_at", "resolved_at")
    sequence = Sequence(zone, default_year, dayfirst)
    sequence.seed([base_date] + [values.get(key, "") for key in order])
    points = {key: sequence.add(values.get(key, "")) for key in order}
    durations: dict[str, str] = {}
    for name, start_key, end_key in _DURATIONS:
        start, end = points[start_key].local, points[end_key].local
        if start is not None and end is not None and end >= start:
            durations[name] = humanize(end - start)
    return Milestones(points, zone, durations)


def describe_date(text: str) -> tuple[str, bool]:
    """('3 Nov 2026', True) or ('5 Nov', False) for a text that states a date; ('', False) otherwise."""
    parsed = parse_time(text)
    if not parsed.has_day:
        return "", False
    label = f"{parsed.day} {calendar.month_abbr[parsed.month]}"
    return (f"{label} {parsed.year}", True) if parsed.year is not None else (label, False)


def clean_time_value(value: str, source: str) -> str:
    """Remove a year, or a whole date, that a model wrote but the source text does not contain.

    Models tend to complete "5 Nov 09:40" to a full date with a year from their
    training data. The quote check cannot catch that, because the quote is real.
    Returns '' when nothing stated is left.
    """
    parsed = parse_time(value)
    if not parsed.has_day:
        return value.strip()
    lowered = source.lower()
    names = (calendar.month_name[parsed.month].lower(), calendar.month_abbr[parsed.month].lower())
    day_stated = re.search(rf"(?<!\d)0?{parsed.day}(?!\d)", source) is not None
    month_stated = any(name in lowered for name in names) or \
        re.search(rf"(?<!\d)0?{parsed.month}(?!\d)", source) is not None
    clock = f"{parsed.hour:02d}:{parsed.minute:02d}" if parsed.has_time else ""
    if not (day_stated and month_stated):
        return clock  # the date is not in the source at all
    year_stated = parsed.year is not None and (
        str(parsed.year) in source or re.search(rf"(?<!\d){parsed.year % 100:02d}(?!\d)", source) is not None)
    if parsed.year is not None and not year_stated:
        return f"{parsed.day} {calendar.month_abbr[parsed.month]} {clock}".strip()
    return value.strip()
