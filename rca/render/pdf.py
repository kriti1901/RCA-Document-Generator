"""Renders the RCA to PDF: a Jinja2 HTML template printed by WeasyPrint.

The template is escaped automatically, so text from the notes can never inject
markup into the document.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from rca.config import Settings
from rca.render.document import build_view

TEMPLATES = Path(__file__).parent / "templates"
# WeasyPrint warns once per font when the optional HarfBuzz-Subset library is absent.
logging.getLogger("weasyprint").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)
_env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=True,
                   trim_blocks=True, lstrip_blocks=True)


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")[:60] or "incident"


def _shorten(text: str, limit: int = 42) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render_html(view: dict) -> str:
    footer_text = view["incident_ref"] or _shorten(view["title"])
    return _env.get_template("rca.html.j2").render(**view, footer_text=footer_text)


def pdf_filename(view: dict, incident_key: str) -> str:
    reference = view["incident_ref"] or incident_key
    suffix = "" if view["final"] else "-draft"
    return f"RCA-{slugify(reference)}-v{view['version']}{suffix}.pdf"


def render_document(state: dict, settings: Settings, *, final: bool = False,
                    now: datetime | None = None, model_description: str = "") -> dict:
    """Write the PDF for the session and return its path, the HTML and the view."""
    from weasyprint import HTML  # imported here because it is slow to load

    now = now or datetime.now()
    view = build_view(state, now=now, final=final, model_description=model_description)
    html = render_html(view)
    out_dir = settings.output_dir / state["incident_key"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / pdf_filename(view, state["incident_key"])
    HTML(string=html).write_pdf(str(path))
    return {"pdf_path": str(path), "html": html, "view": view}
