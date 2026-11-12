"""Run one RCA from the command line, without the web UI.

    python scripts/run_notes.py notes.txt --name "Asha Rao"
    python scripts/run_notes.py notes.txt --name "Asha Rao" --answers answers.txt --approve
    python scripts/run_notes.py notes.txt --name "Asha Rao" --skip-all --approve --publish

In a terminal it asks the questions and waits for you to type. With --answers
it takes each round's answer from a file (rounds separated by a line holding
only ---), and once the file runs out it answers "skip all". With --skip-all
every question is skipped. Without --approve it stops at the draft PDF.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rca.config import load_settings  # noqa: E402
from rca.llm import LLMError  # noqa: E402
from rca.service import RCAService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notes", type=Path, help="text file with the raw incident notes")
    parser.add_argument("--name", required=True, help="your name: the author, and recorded on anything skipped")
    parser.add_argument("--email", default="")
    parser.add_argument("--answers", type=Path, help="file with one answer block per question round")
    parser.add_argument("--skip-all", action="store_true", help="skip every question")
    parser.add_argument("--approve", action="store_true", help="approve the draft and write the final PDF")
    parser.add_argument("--publish", action="store_true", help="with --approve: also publish to Confluence")
    args = parser.parse_args()

    scripted = None
    if args.answers:
        scripted = [block.strip() for block in args.answers.read_text().split("\n---\n") if block.strip()]
    elif args.skip_all or not sys.stdin.isatty():
        scripted = []

    try:
        service = RCAService(load_settings())
        shown = 0

        def show(view) -> int:
            for message in view.messages[shown:]:
                who = "you" if message["role"] == "user" else "assistant"
                print(f"\n[{who}] {message['text']}")
            return len(view.messages)

        progress = lambda text: print(f"  .. {text}", flush=True)  # noqa: E731
        view = service.start(args.notes.read_text(), args.name, args.email, on_progress=progress)
        shown = show(view)
        while view.pending and view.pending["kind"] == "questions" and not view.error:
            if scripted is None:
                print("\n(type your answer; finish with an empty line)")
                lines = []
                while (line := input("> ")).strip():
                    lines.append(line)
                answer = "\n".join(lines) or "skip"
            else:
                answer = scripted.pop(0) if scripted else "skip all"
            view = service.reply(view.session_id, answer, user_name=args.name, on_progress=progress)
            shown = show(view)
        if view.error:
            print(f"\nStopped: {view.error}\nSession {view.session_id} is saved; open it in the web UI to retry.")
            return 1
        if view.pending and view.pending["kind"] == "approval":
            print(f"\nDraft PDF: {view.pdf_path}")
            for issue in view.review_issues:
                print(f"  reviewer comment ({issue['section']}): {issue['problem']}")
            if args.approve:
                view = service.decide(view.session_id, "approve", publish=args.publish, user_name=args.name,
                                      on_progress=progress)
                show(view)
                print(f"\nFinal PDF: {view.pdf_path}")
                if view.confluence.get("url"):
                    print(f"Confluence: {view.confluence['url']}")
                if view.confluence.get("error"):
                    print(f"Confluence publish failed: {view.confluence['error']}")
            else:
                print(f"Not approved yet. Session {view.session_id} is saved; open it in the web UI, "
                      "or run again with --approve.")
        if view.gaps:
            print("\nNot provided:")
            for gap in view.gaps:
                print(f"  - {gap['label']} ({gap['by']}{', ' + gap['note'] if gap.get('note') else ''})")
        if view.usage:
            print("\nModel usage:")
            for agent, stats in view.usage.items():
                print(f"  {agent:<10} " + "  ".join(f"{name}={count}" for name, count in stats.items()))
    except (LLMError, ValueError) as exc:
        print(f"Could not run: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
