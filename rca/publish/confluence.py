"""Publishes a finished RCA to Confluence Cloud.

The RCA becomes a page under the configured path (space key + parent page
titles), with the PDF attached. Publishing the same incident again updates
that page and adds a new version of the attachment instead of creating a copy.

Endpoints used (Confluence Cloud REST):
  GET  /wiki/api/v2/spaces?keys=KEY                      space key -> space id
  GET  /wiki/api/v2/pages?space-id=..&title=..           find a page by title
  GET  /wiki/api/v2/pages/{id}                           page and its version
  GET  /wiki/api/v2/pages/{id}/labels                    labels on a page
  POST /wiki/api/v2/pages                                create a page
  PUT  /wiki/api/v2/pages/{id}                           update a page
  PUT  /wiki/rest/api/content/{id}/child/attachment      create or update an attachment
  POST /wiki/rest/api/content/{id}/label                 add labels
"""

from __future__ import annotations

import re
from datetime import datetime
from html import escape
from pathlib import Path

import httpx

from rca.config import ConfluenceSettings
from rca.graph.build import PublishError
from rca.render.document import build_view

TOOL_LABEL = "rca-assistant"  # marks pages this tool created, so it never overwrites anyone else's


def label_safe(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-")[:60]


class ConfluenceClient:
    def __init__(self, cfg: ConfluenceSettings, *, transport: httpx.BaseTransport | None = None,
                 timeout: float = 30.0) -> None:
        self.cfg = cfg
        self.http = httpx.Client(base_url=cfg.base_url, auth=(cfg.email, cfg.api_token), timeout=timeout,
                                 transport=transport, headers={"Accept": "application/json"})

    def close(self) -> None:
        self.http.close()

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise PublishError(f"Could not reach Confluence at {self.cfg.base_url}: {exc}") from exc
        if response.status_code in (401, 403):
            raise PublishError(
                "Confluence rejected the request. Check CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN, and that "
                f"this account may create pages in space {self.cfg.space_key}."
            )
        if response.status_code >= 400:
            raise PublishError(f"Confluence returned {response.status_code} for {method} {path}: "
                               f"{response.text[:300]}")
        return response

    # ---- lookups ---------------------------------------------------------

    def space_id(self, key: str) -> str:
        results = self._request("GET", "/api/v2/spaces", params={"keys": key}).json().get("results", [])
        if not results:
            raise PublishError(f"Confluence space '{key}' was not found, or this account cannot see it.")
        return str(results[0]["id"])

    def find_page(self, space_id: str, title: str) -> dict | None:
        results = self._request("GET", "/api/v2/pages",
                                params={"space-id": space_id, "title": title, "status": "current",
                                        "limit": 10}).json().get("results", [])
        for page in results:
            if page.get("title", "").strip().lower() == title.strip().lower():
                return page
        return None

    def get_page(self, page_id: str) -> dict:
        return self._request("GET", f"/api/v2/pages/{page_id}").json()

    def page_labels(self, page_id: str) -> set[str]:
        results = self._request("GET", f"/api/v2/pages/{page_id}/labels").json().get("results", [])
        return {label.get("name", "") for label in results}

    def resolve_parent(self, space_id: str, path: str) -> str | None:
        """Walk "Knowledge Base/RCAs" by page title and return the id of the last page."""
        parent_id: str | None = None
        walked: list[str] = []
        for title in (segment.strip() for segment in path.split("/") if segment.strip()):
            page = self.find_page(space_id, title)
            walked.append(title)
            if page is None:
                if not self.cfg.create_missing_parents:
                    raise PublishError(
                        f"The Confluence page '{' / '.join(walked)}' does not exist in space "
                        f"{self.cfg.space_key}. Create it, fix CONFLUENCE_PARENT_PATH, or set "
                        "CONFLUENCE_CREATE_MISSING_PARENTS=true."
                    )
                page = self.create_page(space_id, title, "<p>RCA documents.</p>", parent_id)
            elif parent_id is not None and str(page.get("parentId") or "") != parent_id:
                raise PublishError(
                    f"A page titled '{title}' exists in space {self.cfg.space_key}, but not under "
                    f"'{' / '.join(walked[:-1])}'. Check CONFLUENCE_PARENT_PATH."
                )
            parent_id = str(page["id"])
        return parent_id

    # ---- writes ----------------------------------------------------------

    def create_page(self, space_id: str, title: str, body: str, parent_id: str | None) -> dict:
        payload = {"spaceId": space_id, "status": "current", "title": title,
                   "body": {"representation": "storage", "value": body}}
        if parent_id:
            payload["parentId"] = parent_id
        return self._request("POST", "/api/v2/pages", json=payload).json()

    def update_page(self, page_id: str, title: str, body: str, message: str) -> dict:
        current = self.get_page(page_id)
        payload = {"id": str(page_id), "status": "current", "title": title,
                   "body": {"representation": "storage", "value": body},
                   "version": {"number": int(current["version"]["number"]) + 1, "message": message}}
        return self._request("PUT", f"/api/v2/pages/{page_id}", json=payload).json()

    def upload_attachment(self, page_id: str, filename: str, data: bytes, comment: str) -> dict:
        return self._request(
            "PUT", f"/rest/api/content/{page_id}/child/attachment",
            files={"file": (filename, data, "application/pdf")},
            data={"comment": comment, "minorEdit": "true"},
            headers={"X-Atlassian-Token": "nocheck"},
        ).json()

    def add_labels(self, page_id: str, labels: list[str]) -> None:
        names = list(dict.fromkeys(name for name in (label_safe(label) for label in labels) if name))
        if names:
            self._request("POST", f"/rest/api/content/{page_id}/label",
                          json=[{"prefix": "global", "name": name} for name in names])

    def page_url(self, page: dict) -> str:
        webui = (page.get("_links") or {}).get("webui", "")
        if webui:
            return f"{self.cfg.base_url}{webui}"
        return f"{self.cfg.base_url}/pages/viewpage.action?pageId={page['id']}"


# ---- page content ------------------------------------------------------------

def _cell(row: dict) -> str:
    text = escape(row.get("text", ""))
    return f"<em>{text}</em>" if row.get("state") == "declined" else text


def page_body(view: dict, attachment_name: str) -> str:
    """The Confluence page in storage format: summary, key facts, a link to the PDF, and the gaps."""
    parts = [
        f"<p><strong>Severity:</strong> {escape(view['severity'] or 'Not recorded')} &nbsp; "
        f"<strong>Status:</strong> {'Final' if view['final'] else 'Draft'}, version {view['version']}</p>",
        "<h2>Executive summary</h2>",
    ]
    parts += [f"<p>{escape(p)}</p>" for p in view["summary"]] or ["<p>No summary was written.</p>"]
    parts.append("<h2>Key facts</h2><table><tbody>")
    parts += [f"<tr><th>{escape(row['label'])}</th><td>{_cell(row)}</td></tr>" for row in view["header_rows"]]
    for item in view["timeline"]["durations"]:
        parts.append(f"<tr><th>{escape(item['label'])}</th><td>{escape(item['text'])}</td></tr>")
    parts.append("</tbody></table>")
    parts.append(
        "<h2>Full document</h2><p>The complete RCA is attached as a PDF: "
        f'<ac:link><ri:attachment ri:filename="{escape(attachment_name)}" />'
        f"<ac:plain-text-link-body><![CDATA[{attachment_name}]]></ac:plain-text-link-body></ac:link></p>"
    )
    if view["gaps"]:
        parts.append("<h2>Information gaps</h2><ul>")
        for gap in view["gaps"]:
            who = ", ".join(p for p in (gap["by"], gap["date"]) if p)
            parts.append(f"<li>{escape(gap['label'])}: <em>{escape(view['not_provided'])}"
                         f"{escape(' (' + who + ')') if who else ''}</em></li>")
        parts.append("</ul>")
    return "".join(parts)


def page_title(view: dict) -> str:
    pieces = ["RCA"] + [p for p in (view["incident_ref"], view["title"]) if p]
    return " – ".join(pieces)[:250]


def publish_rca(state: dict, cfg: ConfluenceSettings, *, now: datetime, model_description: str = "",
                client: ConfluenceClient | None = None) -> dict:
    """Create or update the RCA page, attach the PDF, add labels. Returns page id, URL, title, version."""
    owns_client = client is None
    client = client or ConfluenceClient(cfg)
    try:
        view = build_view(state, now=now, final=True, model_description=model_description)
        pdf_path = Path(state.get("pdf_path", ""))
        if not pdf_path.is_file():
            raise PublishError("The PDF to publish was not found on disk.")
        attachment_name = f"RCA-{label_safe(view['incident_ref'] or state['incident_key'])}.pdf"
        title = page_title(view)
        body = page_body(view, attachment_name)
        message = f"RCA version {view['version']} published by {state.get('user', {}).get('name', 'RCA Assistant')}"

        space_id = client.space_id(cfg.space_key)
        known_id = (state.get("confluence") or {}).get("page_id", "")
        if known_id:
            page = client.update_page(known_id, title, body, message)
        else:
            existing = client.find_page(space_id, title)
            if existing is None:
                parent_id = client.resolve_parent(space_id, cfg.parent_path)
                page = client.create_page(space_id, title, body, parent_id)
            elif TOOL_LABEL in client.page_labels(str(existing["id"])):
                page = client.update_page(str(existing["id"]), title, body, message)
            else:
                raise PublishError(
                    f"A page titled '{title}' already exists in space {cfg.space_key} and was not created "
                    "by this tool, so it was left untouched. Rename that page or change the incident title."
                )
        page_id = str(page["id"])
        client.upload_attachment(page_id, attachment_name, pdf_path.read_bytes(), message)
        severity = view["severity"]
        services = [item for row in view["header_rows"] if row["label"] == "Affected services"
                    and row["state"] == "ok" for item in row["text"].split(", ")]
        client.add_labels(page_id, [TOOL_LABEL, "rca", "incident"] + ([severity] if severity else []) + services)
        return {"page_id": page_id, "url": client.page_url(page), "title": title,
                "version": int((page.get("version") or {}).get("number", 1)), "attachment": attachment_name}
    finally:
        if owns_client:
            client.close()
