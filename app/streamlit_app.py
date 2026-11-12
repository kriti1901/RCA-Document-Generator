"""RCA Assistant: the web chat UI.

Run with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from rca.config import load_settings  # noqa: E402
from rca.llm import LLMError  # noqa: E402
from rca.service import RCAService, SessionView  # noqa: E402

st.set_page_config(page_title="RCA Assistant", layout="wide")

STATUS_TEXT = {
    "extracting": "Reading the notes", "planning": "Planning", "clarifying": "Waiting for your answers",
    "drafting": "Drafting", "reviewing": "Reviewing", "revising": "Revising",
    "awaiting_approval": "Waiting for your approval", "approved": "Approved",
    "published": "Published to Confluence", "cancelled": "Cancelled", "error": "Stopped on an error",
}


@st.cache_resource
def get_service() -> RCAService:
    return RCAService(load_settings())


service = get_service()
settings = service.settings


# ---- helpers -------------------------------------------------------------------

def open_session(session_id: str | None) -> None:
    st.session_state["session_id"] = session_id
    st.session_state.pop("view", None)
    if session_id:
        st.query_params["session"] = session_id
    elif "session" in st.query_params:
        del st.query_params["session"]


def run(action, label: str) -> None:
    """Run one workflow step with a live list of what the agents are doing."""
    with st.status(label, expanded=True) as status:
        try:
            view: SessionView = action(status.write)
        except (LLMError, ValueError, KeyError) as exc:
            status.update(label="Could not continue", state="error")
            st.session_state["flash"] = str(exc)
            return
        status.update(label="Stopped on an error" if view.error else "Done",
                      state="error" if view.error else "complete", expanded=False)
    st.session_state["session_id"] = view.session_id
    st.session_state["view"] = view
    st.query_params["session"] = view.session_id
    st.rerun()


def current_user() -> str:
    return st.session_state.get("user_name", "").strip()


def current_view() -> SessionView | None:
    session_id = st.session_state.get("session_id") or st.query_params.get("session")
    if not session_id:
        return None
    view = st.session_state.get("view")
    if view is None or view.session_id != session_id:
        try:
            view = service.view(session_id)
        except (KeyError, LLMError) as exc:
            st.session_state["flash"] = f"That session could not be opened: {exc}"
            open_session(None)
            return None
        st.session_state["session_id"] = session_id
        st.session_state["view"] = view
    return view


view = current_view()
if view is not None and not st.session_state.get("user_name"):
    # Resuming from a link or after a reload: assume the same person, who can still change the name.
    st.session_state["user_name"] = view.author

# ---- sidebar -------------------------------------------------------------------

with st.sidebar:
    st.header("RCA Assistant")
    st.text_input("Your name", key="user_name", placeholder="Shown as the author, and on anything you skip")
    st.text_input("Your email (optional)", key="user_email")
    if st.button("Start a new RCA", use_container_width=True):
        open_session(None)
        st.rerun()

    st.divider()
    st.caption("Model")
    if settings.llm_backend == "demo":
        st.warning("Offline demo mode: a rule-based stand-in, not an AI model. It only understands notes "
                   "written as 'Label: value' lines.")
    elif settings.llm_ready:
        for model in sorted(set(settings.models.values())):
            roles = [role.replace("_", " ") for role, name in settings.models.items() if name == model]
            st.write(f"`{model}`  \n{', '.join(roles)}")
        st.caption(f"Endpoint: {settings.llm_base_url}")
    else:
        st.error("No LLM API key. Set LLM_API_KEY in .env.")
    st.caption("Confluence")
    if settings.confluence:
        path = settings.confluence.parent_path or "(space root)"
        st.write(f"Space `{settings.confluence.space_key}` / {path}")
    else:
        st.write("Not configured. PDFs are saved on this computer only.")

    saved = service.incidents()
    if saved:
        st.divider()
        st.caption("Saved RCAs")
        for incident in saved[:10]:
            label = incident["title"] or incident["incident_id"] or incident["incident_key"]
            gaps = incident["open_gaps"]
            st.write(f"**{label}**  \nv{incident['version']} · {incident['severity'] or 'no severity'} · "
                     f"{gaps} detail{'s' if gaps != 1 else ''} not provided")
            if st.button("Reopen" if gaps else "New version", key=f"reopen-{incident['incident_key']}"):
                if not current_user():
                    st.session_state["flash"] = "Enter your name first."
                else:
                    key = incident["incident_key"]
                    run(lambda progress: service.reopen(key, current_user(),
                                                        st.session_state.get("user_email", ""),
                                                        on_progress=progress), "Reopening")

# ---- main ----------------------------------------------------------------------

flash = st.session_state.pop("flash", None)
if flash:
    st.error(flash)

if view is None:
    st.title("Turn incident notes into an RCA")
    st.write("Paste the raw notes from the incident. The assistant extracts the facts, asks you for anything "
             "the RCA needs that the notes do not say, drafts the document, and waits for your approval "
             "before publishing.")
    with st.form("start"):
        notes = st.text_area("Incident notes", height=320,
                             placeholder="Paste chat logs, timestamps, bullet points: whatever you have.")
        files = st.file_uploader("Or add text files", type=["txt", "md", "log"], accept_multiple_files=True)
        submitted = st.form_submit_button("Start", type="primary")
    if submitted:
        items = [{"text": notes, "source": "pasted notes"}] if notes.strip() else []
        for upload in files or []:
            items.append({"text": upload.getvalue().decode("utf-8", errors="replace"), "source": upload.name})
        if not current_user():
            st.error("Enter your name in the sidebar first. It is recorded as the author.")
        elif not items:
            st.error("Paste some notes or add a file.")
        elif not settings.llm_ready:
            st.error("No LLM API key is configured. Set LLM_API_KEY in .env and restart.")
        else:
            run(lambda progress: service.start(items, current_user(), st.session_state.get("user_email", ""),
                                               on_progress=progress), "Reading the notes")
    st.stop()

# ---- an open session -----------------------------------------------------------

st.title(view.title or "New RCA")
st.caption(f"{STATUS_TEXT.get(view.status, view.status)} · version {view.version} · "
           f"{view.fact_count} facts · session {view.session_id}")

for message in view.messages:
    with st.chat_message("user" if message["role"] == "user" else "assistant"):
        st.markdown(message["text"])

if view.error:
    st.error(view.error)
    if st.button("Try that step again"):
        session_id = view.session_id
        run(lambda progress: service.retry(session_id, on_progress=progress), "Retrying")

kind = (view.pending or {}).get("kind")

if kind == "questions" and not view.error:
    left, right, _ = st.columns([1, 1.4, 3])
    skip = left.button("Skip these")
    skip_all = right.button("Skip all and draft now")
    typed = st.chat_input("Answer by number, for example: 1. Sev2   2. skip")
    answer = "skip" if skip else "skip all" if skip_all else typed
    if answer:
        if not current_user():
            st.error("Enter your name in the sidebar first. It is recorded against anything you skip.")
        else:
            session_id = view.session_id
            run(lambda progress: service.reply(session_id, answer, user_name=current_user(),
                                               on_progress=progress), "Reading your answers")

elif kind == "approval" and not view.error:
    preview, panel = st.columns([3, 2])
    with preview:
        pdf = Path(view.pdf_path)
        if pdf.is_file():
            st.pdf(str(pdf), height=760)
            st.download_button("Download the draft PDF", pdf.read_bytes(), file_name=pdf.name,
                               mime="application/pdf")
        else:
            st.warning("The draft PDF is missing from disk.")
    with panel:
        st.subheader("Review")
        if view.review_issues:
            st.write("**Reviewer comments still open**")
            for issue in view.review_issues:
                st.write(f"- {issue['section'].replace('_', ' ').title()}: {issue['problem']}")
        if view.gaps:
            with st.expander(f"{len(view.gaps)} detail{'s' if len(view.gaps) != 1 else ''} not provided"):
                for gap in view.gaps:
                    st.write(f"- {gap['label']} ({gap['by']})")
        else:
            st.write("Every required detail was provided.")
        if view.rejected:
            with st.expander(f"{len(view.rejected)} extracted item(s) dropped for lack of a supporting quote"):
                for item in view.rejected:
                    text = item.get("value") or item.get("event") or item.get("action") or item.get("owner", "")
                    st.write(f"- {item['kind']}: {text} ({item['reason']})")

        st.divider()
        session_id = view.session_id
        if not current_user():
            st.info("Enter your name in the sidebar to approve.")
        else:
            if settings.confluence and st.button("Approve and publish to Confluence", type="primary",
                                                 use_container_width=True):
                run(lambda progress: service.decide(session_id, "approve", publish=True,
                                                    user_name=current_user(), on_progress=progress), "Publishing")
            if st.button("Approve and save the PDF" + (" only" if settings.confluence else ""),
                         type="secondary" if settings.confluence else "primary", use_container_width=True):
                run(lambda progress: service.decide(session_id, "approve", publish=False,
                                                    user_name=current_user(), on_progress=progress), "Finalising")
            with st.form("changes"):
                feedback = st.text_area("What should change?", height=110,
                                        placeholder="For example: the root cause was the missing index, "
                                                    "not the deploy. Shorten the summary.")
                if st.form_submit_button("Request changes") and feedback.strip():
                    run(lambda progress: service.decide(session_id, "changes", feedback=feedback,
                                                        user_name=current_user(), on_progress=progress),
                        "Revising the draft")
            if st.button("Cancel this RCA"):
                run(lambda progress: service.decide(session_id, "cancel", user_name=current_user(),
                                                    on_progress=progress), "Cancelling")

elif view.finished:
    pdf = Path(view.pdf_path)
    url = view.confluence.get("url", "")
    if view.status == "published" and url:
        st.success(f"Published to Confluence: {url}")
    elif view.status == "approved":
        st.success("Approved. The final PDF is saved on this computer.")
        if view.confluence.get("error") and settings.confluence:
            st.warning(f"Publishing to Confluence failed: {view.confluence['error']}")
            if st.button("Try publishing again"):
                with st.spinner("Publishing"):
                    retried = service.retry_publish(view.session_id)
                if retried.error:
                    st.session_state["flash"] = retried.error
                st.session_state["view"] = retried
                st.rerun()
    if pdf.is_file() and view.status != "cancelled":
        st.download_button("Download the final PDF", pdf.read_bytes(), file_name=pdf.name,
                           mime="application/pdf")
        st.pdf(str(pdf), height=760)

with st.expander("Session details"):
    st.write(f"Session `{view.session_id}` · incident `{view.incident_key}`")
    st.caption("This session is saved after every step. Reload the page or come back later; the address "
               "bar keeps the link to it.")
    if view.usage:
        st.table([{"agent": agent, **stats} for agent, stats in view.usage.items()])
