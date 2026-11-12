"""The incident facts held in the session state.

Every field carries a status, so "missing" and "declined by the user" are
explicit rather than implied by an empty value.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rca.schema.field_specs import LIST_IDS, SCALAR_IDS

Status = Literal["from_notes", "from_user", "derived", "missing", "declined", "not_applicable"]

PROVIDED: frozenset[str] = frozenset({"from_notes", "from_user", "derived"})


class FieldValue(BaseModel):
    value: str = ""
    status: Status = "missing"
    evidence: str = ""  # verbatim quote from the notes or from the user's answer
    by: str = ""  # who provided or declined it
    at: str = ""  # ISO timestamp of that
    note: str = ""  # e.g. "no clear answer given"

    @property
    def provided(self) -> bool:
        return self.status in PROVIDED and bool(self.value)


class ListItem(BaseModel):
    value: str
    evidence: str = ""
    status: Status = "from_notes"
    by: str = ""


class ListValue(BaseModel):
    items: list[ListItem] = Field(default_factory=list)
    status: Status = "missing"
    by: str = ""
    at: str = ""
    note: str = ""

    @property
    def provided(self) -> bool:
        return self.status in PROVIDED and bool(self.items)


class TimelineEvent(BaseModel):
    time: str  # "YYYY-MM-DD HH:MM" or "HH:MM", local time as written in the notes
    event: str
    source: str = ""
    evidence: str = ""
    status: Status = "from_notes"


class ActionItem(BaseModel):
    action: str
    type: str = ""  # corrective | preventive | detective | ""
    owner: str = ""
    owner_status: Status = "missing"
    owner_by: str = ""
    owner_at: str = ""
    due_date: str = ""
    priority: str = ""
    ticket: str = ""
    evidence: str = ""
    status: Status = "from_notes"


class IncidentFacts(BaseModel):
    scalars: dict[str, FieldValue] = Field(default_factory=dict)
    lists: dict[str, ListValue] = Field(default_factory=dict)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    actions: list[ActionItem] = Field(default_factory=list)
    actions_status: Status = "missing"  # status of the action list as a whole
    actions_by: str = ""
    actions_at: str = ""

    @classmethod
    def empty(cls) -> "IncidentFacts":
        return cls(
            scalars={fid: FieldValue() for fid in SCALAR_IDS},
            lists={fid: ListValue() for fid in LIST_IDS},
        )

    @classmethod
    def load(cls, data: dict | None) -> "IncidentFacts":
        """Rebuild from the state dict, adding any field the registry gained since it was saved."""
        facts = cls.model_validate(data) if data else cls()
        for fid in SCALAR_IDS:
            facts.scalars.setdefault(fid, FieldValue())
        for fid in LIST_IDS:
            facts.lists.setdefault(fid, ListValue())
        return facts

    def scalar(self, field_id: str) -> FieldValue:
        return self.scalars[field_id]

    def value(self, field_id: str) -> str:
        """The value of a scalar field if it was provided, else ''."""
        fv = self.scalars.get(field_id)
        return fv.value if fv and fv.provided else ""

    @property
    def severity(self) -> str:
        return self.value("severity")

    def provided_count(self) -> int:
        count = sum(1 for fv in self.scalars.values() if fv.provided)
        count += sum(len(lv.items) for lv in self.lists.values() if lv.provided)
        return count + len(self.timeline) + len(self.actions)
