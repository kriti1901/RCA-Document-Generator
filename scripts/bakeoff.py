"""Compare candidate models on the sample incident notes, to choose which one the agents use.

    python scripts/bakeoff.py --list                      models your key can call
    python scripts/bakeoff.py                             score the default candidates on extraction
    python scripts/bakeoff.py --models a,b --full         also run the whole workflow once per model

For each model it runs the Extractor over three sample notes (complete, partial,
sparse) and reports:
  found        expected facts it extracted (out of those the notes actually state)
  unsupported  facts it reported for fields the notes do not mention (lower is better)
  dropped      items the evidence check threw away because the quote was not in the notes
  repairs      replies that did not match the schema and had to be sent back
With --full it also drafts and reviews one RCA and reports figures the Drafter
wrote that are not in the facts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rca.agents import extractor  # noqa: E402
from rca.config import MODEL_ROLES, load_settings  # noqa: E402
from rca.llm import LLMError, OpenAICompatLLM  # noqa: E402
from rca.service import RCAService  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
DEFAULT_CANDIDATES = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "qwen/qwen3.6-27b",
                      "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
SAMPLES = ["notes_complete.txt", "notes_partial.txt", "notes_sparse.txt"]


def score(result: extractor.Extraction, key: dict) -> dict:
    values = {fid: fv.value for fid, fv in result.scalars.items()}
    for fid, items in result.list_items.items():
        values[fid] = " | ".join(item.value for item in items)
    found = [fid for fid, fragments in key["expected"].items()
             if any(fragment.lower() in values.get(fid, "").lower() for fragment in fragments)]
    return {
        "found": len(found), "expected": len(key["expected"]),
        "missed": sorted(set(key["expected"]) - set(found)),
        "unsupported": sorted(fid for fid in key["absent"] if values.get(fid)),
        "dropped": len(result.rejected),
        "actions": len(result.actions), "actions_expected": key["actions"],
        "events": len(result.events), "events_min": key["events_min"],
        "values": values,
    }


def totals(llm: OpenAICompatLLM) -> dict:
    out: dict[str, int] = {}
    for stats in llm.usage.values():
        for name, count in stats.items():
            out[name] = out.get(name, 0) + count
    return out


def run_model(model: str, base, keys: dict, full: bool, verbose: bool) -> dict:
    settings = replace(base, models={role: model for role in MODEL_ROLES})
    llm = OpenAICompatLLM(settings, on_event=(lambda m: print(f"    .. {m}", flush=True)) if verbose else None)
    report: dict = {"model": model, "samples": {}, "error": ""}
    started = time.time()
    try:
        for name in SAMPLES:
            t0 = time.time()
            result = extractor.extract(llm, (FIXTURES / name).read_text(), max_chars=settings.chunk_chars)
            report["samples"][name] = {**score(result, keys[name]), "seconds": round(time.time() - t0, 1)}
            s = report["samples"][name]
            print(f"  {name:<20} found {s['found']}/{s['expected']}  unsupported {len(s['unsupported'])}  "
                  f"dropped {s['dropped']}  actions {s['actions']}/{s['actions_expected']}  {s['seconds']}s",
                  flush=True)
        report["extraction_usage"] = totals(llm)
        if full:
            data_dir = base.data_dir / "bakeoff" / model.replace("/", "_")
            service = RCAService(replace(settings, data_dir=data_dir), llm=llm)
            view = service.start((FIXTURES / "notes_partial.txt").read_text(), "Bake-off")
            while view.pending and view.pending["kind"] == "questions" and not view.error:
                view = service.reply(view.session_id, "skip all")
            report["full"] = {
                "reached_approval": bool(view.pending and view.pending["kind"] == "approval"),
                "error": view.error, "pdf": view.pdf_path,
                "open_review_comments": [f"{i['section']}: {i['problem']}" for i in view.review_issues],
                "figures_not_in_facts": sum(1 for i in view.review_issues if i.get("source") == "code"),
            }
            print(f"  full workflow: {'reached approval' if report['full']['reached_approval'] else 'stopped'}"
                  f"  open review comments {len(view.review_issues)}  {view.error}", flush=True)
    except LLMError as exc:
        report["error"] = str(exc)
        print(f"  FAILED: {exc}", flush=True)
    report["usage"] = totals(llm)
    report["seconds"] = round(time.time() - started, 1)
    return report


def table(reports: list[dict]) -> str:
    lines = ["| Model | Facts found | Unsupported | Dropped by evidence check | Actions | Schema repairs | "
             "Rate-limit waits | Seconds | Result |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in reports:
        samples = r["samples"].values()
        usage = r.get("usage", {})
        found = f"{sum(s['found'] for s in samples)}/{sum(s['expected'] for s in samples)}" if samples else "-"
        actions = (f"{sum(s['actions'] for s in samples)}/{sum(s['actions_expected'] for s in samples)}"
                   if samples else "-")
        if r["error"]:
            result = "failed: " + r["error"][:70]
        elif "full" in r:
            full = r["full"]
            result = ("workflow ok" if full["reached_approval"] else "workflow stopped") + \
                     f", {full['figures_not_in_facts']} unsupported figure(s) in the draft"
        else:
            result = "ok"
        lines.append(
            f"| `{r['model']}` | {found} | {sum(len(s['unsupported']) for s in samples)} | "
            f"{sum(s['dropped'] for s in samples)} | {actions} | "
            f"{usage.get('repairs', 0) + usage.get('failed_generations', 0)} | "
            f"{usage.get('rate_limit_waits', 0)} | {r['seconds']} | {result} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", help="comma-separated model ids (default: the candidates your key can call)")
    parser.add_argument("--list", action="store_true", help="list the models your key can call and exit")
    parser.add_argument("--full", action="store_true", help="also run the whole workflow once per model")
    parser.add_argument("--quiet", action="store_true", help="do not print rate-limit waits")
    args = parser.parse_args()

    base = load_settings()
    if not base.llm_api_key:
        print("No LLM_API_KEY is set. Put a key in .env first.")
        return 2
    try:
        available = OpenAICompatLLM(base).list_models()
    except Exception as exc:  # any failure here is a setup problem worth showing as-is
        print(f"Could not list models at {base.llm_base_url}: {exc}")
        return 2
    if args.list:
        print("\n".join(available))
        return 0

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = [m for m in DEFAULT_CANDIDATES if m in available]
        skipped = [m for m in DEFAULT_CANDIDATES if m not in available]
        if skipped:
            print("Not available to this key, skipped:", ", ".join(skipped))
    if not models:
        print("None of the default candidates is available. Run with --list and pass --models.")
        return 2

    keys = json.loads((FIXTURES / "expected.json").read_text())
    reports = []
    for model in models:
        print(f"\n== {model}", flush=True)
        reports.append(run_model(model, base, keys, args.full, not args.quiet))

    out_dir = base.data_dir / "bakeoff"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"bakeoff-{stamp}.json").write_text(json.dumps(reports, indent=1))
    markdown = table(reports)
    (out_dir / f"bakeoff-{stamp}.md").write_text(markdown + "\n")
    print("\n" + markdown)
    print(f"\nDetails: {out_dir / f'bakeoff-{stamp}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
