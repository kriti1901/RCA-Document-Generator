"""Publishing, against an in-memory stand-in for the Confluence Cloud REST endpoints.

These tests check our side of the conversation: which endpoints are called,
with what, and how failures are reported. They cannot prove the endpoint
shapes match the real Confluence; that needs a run against a real site.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace

import httpx
import pytest

from rca.config import ConfluenceSettings, normalize_confluence_url
from rca.graph.build import PublishError
from rca.publish.confluence import TOOL_LABEL, ConfluenceClient, label_safe, publish_rca
from tests.conftest import NOW, pdf_text, to_approval

CFG = ConfluenceSettings(base_url="https://example.atlassian.net/wiki", email="bot@example.com",
                         api_token="token", space_key="OPS", parent_path="Knowledge Base/RCAs")


class FakeConfluence:
    def __init__(self) -> None:
        self.spaces = {"OPS": "1001"}
        self.pages: dict[str, dict] = {}
        self.attachments: dict[str, dict[str, list[bytes]]] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail_with: int | None = None
        self._next = 5000
        self.add_page("Knowledge Base", None)
        self.add_page("RCAs", self.id_of("Knowledge Base"))

    # -- helpers used by the tests --
    def add_page(self, title: str, parent_id: str | None, labels: tuple[str, ...] = ()) -> str:
        self._next += 1
        page_id = str(self._next)
        self.pages[page_id] = {"id": page_id, "title": title, "spaceId": "1001", "parentId": parent_id,
                               "version": {"number": 1}, "body": "", "labels": set(labels),
                               "_links": {"webui": f"/spaces/OPS/pages/{page_id}"}}
        return page_id

    def id_of(self, title: str) -> str:
        return next(p["id"] for p in self.pages.values() if p["title"] == title)

    def titled(self, prefix: str) -> list[dict]:
        return [p for p in self.pages.values() if p["title"].startswith(prefix)]

    def public(self, page: dict) -> dict:
        return {k: v for k, v in page.items() if k not in ("labels", "body")}

    # -- the fake server --
    def handler(self, request: httpx.Request) -> httpx.Response:
        path, method = request.url.path.removeprefix("/wiki"), request.method
        self.calls.append((method, path))
        if self.fail_with:
            return httpx.Response(self.fail_with, json={"message": "nope"})
        if not request.headers.get("authorization", "").startswith("Basic "):
            return httpx.Response(401, json={"message": "no credentials"})
        params = request.url.params

        if (method, path) == ("GET", "/api/v2/spaces"):
            key = params.get("keys")
            return httpx.Response(200, json={"results": [{"id": self.spaces[key], "key": key}]
                                                        if key in self.spaces else []})
        if (method, path) == ("GET", "/api/v2/pages"):
            found = [self.public(p) for p in self.pages.values()
                     if p["spaceId"] == params.get("space-id") and p["title"] == params.get("title")]
            return httpx.Response(200, json={"results": found})
        if (method, path) == ("POST", "/api/v2/pages"):
            data = json.loads(request.content)
            if any(p["title"] == data["title"] for p in self.pages.values()):
                return httpx.Response(400, json={"message": "A page with this title already exists"})
            page_id = self.add_page(data["title"], data.get("parentId"))
            self.pages[page_id]["body"] = data["body"]["value"]
            assert data["body"]["representation"] == "storage" and data["status"] == "current"
            return httpx.Response(200, json=self.public(self.pages[page_id]))

        match = re.fullmatch(r"/api/v2/pages/(\d+)(/labels)?", path)
        if match and match.group(1) in self.pages:
            page = self.pages[match.group(1)]
            if match.group(2):
                return httpx.Response(200, json={"results": [{"name": n} for n in sorted(page["labels"])]})
            if method == "GET":
                return httpx.Response(200, json=self.public(page))
            if method == "PUT":
                data = json.loads(request.content)
                if data["version"]["number"] != page["version"]["number"] + 1:
                    return httpx.Response(409, json={"message": "version conflict"})
                page.update(title=data["title"], body=data["body"]["value"], version={"number": data["version"]["number"]})
                return httpx.Response(200, json=self.public(page))

        match = re.fullmatch(r"/rest/api/content/(\d+)/child/attachment", path)
        if match and method == "PUT":
            if request.headers.get("x-atlassian-token") != "nocheck":
                return httpx.Response(403, json={"message": "XSRF check failed"})
            body = request.content
            filename = re.search(rb'name="file"; filename="([^"]+)"', body).group(1).decode()
            assert b"%PDF-" in body and b'name="comment"' in body
            self.attachments.setdefault(match.group(1), {}).setdefault(filename, []).append(body)
            return httpx.Response(200, json={"results": [{"title": filename}]})

        match = re.fullmatch(r"/rest/api/content/(\d+)/label", path)
        if match and method == "POST":
            self.pages[match.group(1)]["labels"].update(item["name"] for item in json.loads(request.content))
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404, json={"message": f"no route for {method} {path}"})

    def client(self, cfg: ConfluenceSettings = CFG) -> ConfluenceClient:
        return ConfluenceClient(cfg, transport=httpx.MockTransport(self.handler))


@pytest.fixture
def fake():
    return FakeConfluence()


@pytest.fixture
def approved(service, demo_notes):
    """The saved state of an approved RCA, with its final PDF on disk."""
    view = service.decide(to_approval(service, demo_notes).session_id, "approve")
    return service._values(view.session_id)


def publish(state, fake, cfg=CFG):
    return publish_rca(state, cfg, now=NOW, client=fake.client(cfg))


# ---- the publisher -------------------------------------------------------------

def test_first_publish_creates_the_page_under_the_path(fake, approved):
    result = publish(approved, fake)
    title = "RCA – INC-4821 – Payments API outage"
    page = fake.pages[result["page_id"]]
    assert result["title"] == page["title"] == title
    assert page["parentId"] == fake.id_of("RCAs"), "created under Knowledge Base / RCAs"
    assert result["url"] == f"https://example.atlassian.net/wiki/spaces/OPS/pages/{result['page_id']}"
    assert page["labels"] == {TOOL_LABEL, "rca", "incident", "sev2", "payments-api", "checkout-web"}

    body = page["body"]
    assert "<h2>Executive summary</h2>" in body and "<th>Incident ID</th><td>INC-4821</td>" in body
    assert '<ri:attachment ri:filename="RCA-inc-4821.pdf" />' in body
    assert "Scale of impact: <em>Not provided by the user (Test User, 11 Nov 2026)</em>" in body
    files = fake.attachments[result["page_id"]]
    assert list(files) == ["RCA-inc-4821.pdf"] and len(files["RCA-inc-4821.pdf"]) == 1


def test_publishing_again_updates_the_same_page(fake, approved):
    first = publish(approved, fake)
    again = publish({**approved, "version": 2, "confluence": first}, fake)
    assert again["page_id"] == first["page_id"] and again["version"] == 2
    assert len(fake.titled("RCA \u2013")) == 1, "no duplicate page"
    assert len(fake.attachments[first["page_id"]]["RCA-inc-4821.pdf"]) == 2, "a new attachment version"
    assert ("POST", "/api/v2/pages") not in fake.calls[-6:]


def test_a_page_this_tool_made_is_found_again_by_its_label(fake, approved):
    first = publish(approved, fake)
    again = publish(approved, fake)  # the page id was lost, e.g. published from another machine
    assert again["page_id"] == first["page_id"] and len(fake.titled("RCA \u2013")) == 1


def test_someone_elses_page_is_never_overwritten(fake, approved):
    theirs = fake.add_page("RCA – INC-4821 – Payments API outage", fake.id_of("RCAs"))
    fake.pages[theirs]["body"] = "written by a person"
    with pytest.raises(PublishError, match="was not created by this tool"):
        publish(approved, fake)
    assert fake.pages[theirs]["body"] == "written by a person" and theirs not in fake.attachments


def test_missing_parent_path_is_explained_or_created(fake, approved):
    cfg = replace(CFG, parent_path="Knowledge Base/RCAs/2026")
    with pytest.raises(PublishError, match="'Knowledge Base / RCAs / 2026' does not exist"):
        publish(approved, fake, cfg)
    assert fake.titled("RCA –") == [], "nothing is created when the path is wrong"

    result = publish(approved, fake, replace(cfg, create_missing_parents=True))
    year = fake.pages[fake.id_of("2026")]
    assert year["parentId"] == fake.id_of("RCAs")
    assert fake.pages[result["page_id"]]["parentId"] == year["id"]


def test_a_path_that_does_not_nest_is_refused(fake, approved):
    fake.add_page("Runbooks", None)
    with pytest.raises(PublishError, match="but not under 'Runbooks'"):
        publish(approved, fake, replace(CFG, parent_path="Runbooks/RCAs"))


def test_empty_path_publishes_at_the_space_root(fake, approved):
    result = publish(approved, fake, replace(CFG, parent_path=""))
    assert fake.pages[result["page_id"]]["parentId"] is None


@pytest.mark.parametrize("status,message", [(401, "rejected the request"), (403, "rejected the request"),
                                            (500, "Confluence returned 500")])
def test_failures_are_reported_in_plain_words(fake, approved, status, message):
    fake.fail_with = status
    with pytest.raises(PublishError, match=message):
        publish(approved, fake)


def test_unknown_space_and_missing_pdf(fake, approved):
    with pytest.raises(PublishError, match="space 'NOPE' was not found"):
        publish(approved, fake, replace(CFG, space_key="NOPE"))
    with pytest.raises(PublishError, match="PDF to publish was not found"):
        publish({**approved, "pdf_path": "/nonexistent/rca.pdf"}, fake)


def test_small_helpers():
    assert normalize_confluence_url("https://x.atlassian.net/") == "https://x.atlassian.net/wiki"
    assert normalize_confluence_url("https://x.atlassian.net/wiki/") == "https://x.atlassian.net/wiki"
    assert label_safe("Payments API (EU)") == "payments-api-eu" and label_safe("Sev2") == "sev2"


# ---- publishing through the workflow -------------------------------------------

def test_approve_and_publish_through_the_service(make_service, fake, demo_notes):
    service = make_service(publish=lambda state: publish_rca(state, CFG, now=NOW, client=fake.client()))
    view = service.decide(to_approval(service, demo_notes).session_id, "approve", publish=True)
    assert view.status == "published" and view.confluence["error"] == ""
    assert view.messages[-1]["text"] == f"Published to Confluence: {view.confluence['url']}"
    saved = service.incidents()[0]
    assert saved["confluence_url"] == view.confluence["url"] and saved["status"] == "published"
    uploaded = fake.attachments[view.confluence["page_id"]]["RCA-inc-4821.pdf"][0]
    assert b"%PDF-" in uploaded
    assert "Approved by Test User" in pdf_text(view.pdf_path), "the final PDF is what gets uploaded"

    # A later version of the same incident updates that page.
    reopened = service.reopen(view.incident_key, "Test User")
    reopened = service.reply(reopened.session_id, "skip all")
    again = service.decide(reopened.session_id, "approve", publish=True)
    assert again.confluence["page_id"] == view.confluence["page_id"] and len(fake.titled("RCA –")) == 1
    assert len(fake.attachments[view.confluence["page_id"]]["RCA-inc-4821.pdf"]) == 2


def test_a_failed_publish_keeps_the_approval_and_can_be_retried(make_service, fake, demo_notes):
    service = make_service(publish=lambda state: publish_rca(state, CFG, now=NOW, client=fake.client()))
    fake.fail_with = 500
    view = service.decide(to_approval(service, demo_notes).session_id, "approve", publish=True)
    assert view.status == "approved", "the approval and the PDF are not lost"
    assert "Confluence returned 500" in view.confluence["error"]
    assert "publishing to Confluence failed" in view.messages[-1]["text"]
    assert service.incidents()[0]["confluence_url"] == ""

    assert "Confluence returned 500" in service.retry_publish(view.session_id).error
    fake.fail_with = None
    retried = service.retry_publish(view.session_id)
    assert retried.status == "published" and retried.confluence["error"] == "" and not retried.error
    assert service.incidents()[0]["confluence_url"] == retried.confluence["url"]
    assert [e["event"] for e in service.memory.audit(view.incident_key)] == ["approved", "published"]


def test_approving_without_publishing_touches_nothing(make_service, fake, demo_notes):
    service = make_service(publish=lambda state: publish_rca(state, CFG, now=NOW, client=fake.client()))
    view = service.decide(to_approval(service, demo_notes).session_id, "approve", publish=False)
    assert view.status == "approved" and fake.calls == [] and view.confluence == {}
