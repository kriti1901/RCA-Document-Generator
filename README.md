# RCA Assistant

Turns raw incident notes into a structured Root Cause Analysis (RCA) PDF and stores it in Confluence. When the notes lack something the RCA needs, it asks you. If you skip a question, the document says "Not provided by the user" with your name and the date.

Five AI agents do the reading, questioning, writing and checking. Ordinary code does everything that must be exact: deciding what is missing, calculating durations, laying out the PDF, and publishing.

```
Raw notes
   │
   ▼
Extractor ──▶ Planner ──▶ gaps? ──yes──▶ Clarifier ◀──▶ YOU (answer or skip)
                            │ no             │
                            ▼                │ facts updated
                         Drafter ◀───────────┘
                            │        ▲
                            ▼        │ revise (at most 2)
                         Reviewer ───┘
                            │ approve
                            ▼
                       PDF renderer ──▶ YOU approve ──▶ Confluence ──▶ long-term memory
```

## Quick start

You need Python 3.11 or newer, and the system libraries WeasyPrint uses to draw PDFs (Pango). On Debian or Ubuntu: `sudo apt install libpango-1.0-0 libpangoft2-1.0-0`. For macOS and Windows, see the [WeasyPrint installation guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html). This project was built and tested on Linux only.

```bash
python -m venv .venv
```

```bash
. .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Create a free API key at [console.groq.com](https://console.groq.com) and put it in `.env` as `LLM_API_KEY`. The model names in `.env.example` are the setup this project was tested with. Then start the app:

```bash
streamlit run app/streamlit_app.py
```

To look around without a key, set `LLM_BACKEND=demo` in `.env`. That uses a rule-based stand-in, not an AI model, and it only understands notes written as `Label: value` lines such as `tests/fixtures/demo_notes.txt`.

## Using it

1. Enter your name in the sidebar. It is recorded as the author and against anything you skip.
2. Paste the incident notes, or upload text files, and press **Start**.
3. Answer the questions by number, for example `1. Sev2  2. skip  3. Ravi, by 30 Nov`. Say `skip` for anything you do not have, or `skip all` to go ahead.
4. Review the draft PDF. Approve it, or type what should change.
5. On approval the final PDF is saved under `data/output/`, and published to Confluence if that is configured.

The session is saved after every step. Reload the page or come back days later; the link in the address bar reopens it. A saved RCA can be reopened from the sidebar: the assistant asks only for what was skipped last time, produces the next version, and updates the same Confluence page.

There is also a command-line runner:

```bash
python scripts/run_notes.py tests/fixtures/notes_partial.txt --name "Your Name"
```

## The agents

| Agent | Job |
|---|---|
| Extractor | Reads the notes and pulls out facts. Each fact must carry the exact quote it came from. |
| Planner | Flags facts too vague to be useful and sets the focus of each section. |
| Clarifier | Reads your free-text answers and sorts them into answers, skips and extra information. |
| Drafter | Writes the narrative sections from confirmed facts only. |
| Reviewer | Checks the draft against the facts and sends it back for a rewrite if needed. |

What keeps the output trustworthy, especially with open-source models:

- **Evidence check.** Code looks for every quote in your notes. A fact whose quote is not there is dropped, so you get asked instead of the document inventing something.
- **Validate and repair.** Every model reply is checked against a schema. A reply that does not fit is sent back with the error, at most twice.
- **No invented dates.** Models like to complete "5 Nov" with a year from their training data. Code removes any year or date that is not in your notes, and the app asks you for the year instead.
- **Figures check.** Code flags any number in the draft that is not in the facts.
- **Small steps.** Each call has one small job and a short schema.
- **Your approval** before anything is published.

The models still make judgment slips that code cannot catch, such as pairing a time with the wrong event or an awkward sentence. The Reviewer catches some. Read the draft before you approve it.

## How missing details are handled

The RCA layout lives in `rca/schema/field_specs.py`: every field, its section, when it is required, and the question to ask for it. Code compares the facts with that list.

- Questions come five at a time, most important first. The last round asks everything that is left, so nothing is marked "not provided" without having been asked.
- An unclear answer is asked once more, then closed.
- "Don't know", "nobody measured it", "will check later" and similar count as a skip, not as an answer.
- A date without a year gets one follow-up question. A year is never assumed.
- Some fields are required only for Sev1 and Sev2, so they are asked once the severity is known.
- A skipped detail appears in the PDF as *Not provided by the user (name, date)*, in place and again in the "Information gaps" section.

To match your company's RCA template, edit `field_specs.py` and `rca/render/templates/rca.html.j2`. The agents read the field list from there.

## Memory

| | What it holds | Where |
|---|---|---|
| Short-term | One session's notes, facts, questions, answers, drafts and review comments | `data/rca.db`, saved after every step |
| Long-term | Finished RCAs, what was skipped and by whom, known service names, an audit trail, and a search index for similar past incidents | `data/rca.db` |
| Published documents | The RCA page and PDF | Confluence |
| Files | Generated PDFs | `data/output/` |

For production, both move to Postgres (LangGraph's Postgres checkpointer, and pgvector for semantic search).

## Configuration

All settings are in `.env`.

| Setting | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | Any OpenAI-compatible endpoint |
| `LLM_API_KEY` | | Key for that endpoint |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Model for all agents |
| `LLM_MODEL_EXTRACTOR`, `_PLANNER`, `_CLARIFIER`, `_DRAFTER`, `_REVIEWER` | `LLM_MODEL` | Per-agent override |
| `LLM_MODEL_EXTRACTOR_LISTS` | the Extractor's model | Model for the Extractor's timeline and action-item passes |
| `LLM_BACKEND` | `openai` | `demo` for the offline stand-in |
| `LLM_JSON_MODE` | `auto` | `auto` tries strict JSON schema first and steps down if the endpoint refuses |
| `LLM_REASONING_EFFORT` | | Passed through for models that accept it |
| `LLM_MAX_OUTPUT_TOKENS` | `4000` | First limit for a reply; doubled once automatically if a reply is cut off |
| `LLM_MAX_RATE_WAIT_S` | `300` | Longest total wait for a rate limit before giving up |
| `RCA_QUESTIONS_PER_ROUND` | `5` | |
| `RCA_MAX_CLARIFY_ROUNDS` | `3` | The last round asks everything left |
| `RCA_MAX_REVISIONS` | `2` | Rewrites the Reviewer may ask for |
| `RCA_CHUNK_CHARS` | `8000` | Longer notes are split into pieces of this size for extraction |
| `RCA_DATA_DIR` | `./data` | Database and PDFs |
| `CONFLUENCE_*` | | See below |

### Choosing a model

```bash
python scripts/bakeoff.py --list
```

```bash
python scripts/bakeoff.py --full
```

The first lists the models your key can call. The second runs each candidate over the three sample notes in `tests/fixtures/` and prints a table: facts found, unsupported facts, items dropped by the evidence check, schema repairs, and rate-limit waits.

Results on 12 November 2026 with a Groq free-tier key (three sample notes, one run each, so treat small differences as noise):

| Model | Expected facts found | Facts the notes do not support | Dropped by the evidence check | Follow-up actions found | Schema repairs |
|---|---|---|---|---|---|
| `openai/gpt-oss-120b` | 35 of 38 | 0 | 1 | 5 of 5 | 0 |
| `openai/gpt-oss-20b` | 37 of 38 | 0 | 0 | 5 of 5 | 0 |
| `qwen/qwen3.8-27b` | 33 of 38 | 0 | 0 | 5 of 5 | 0 |

Some "missed" facts are a different but defensible reading of the notes (for example, taking the moment an engineer noticed the errors as the detection time, where the scoring key expects the alert six minutes later). A missed fact is the safe kind of error: the app asks you for it.

All three returned valid JSON on every call. The tested setup uses each model where it did best and spreads the load:

| Role | Model | Why |
|---|---|---|
| Extractor (facts), Planner, Reviewer | `openai/gpt-oss-120b` | The largest of the three, for the steps that need the most judgment |
| Extractor (timeline and actions) | `openai/gpt-oss-20b` | Found every event and action; uses a separate rate-limit budget |
| Clarifier, Drafter | `qwen/qwen3.8-27b` | Read awkward replies correctly (18 of 18 checks) and writes fluent prose with the fewest tokens |

### Free-tier limits

Measured on that key: each model allows 1,000 requests a day and 8,000 tokens a minute. One RCA took 12 to 16 model calls and 17,000 to 23,000 tokens, and finished in about a minute with a few short waits. The daily request limit therefore allows on the order of a hundred RCAs a day; the per-minute token limit is the one you notice. Your limits are shown in the Groq console and may differ.

### Running a model on your own machine

Install [Ollama](https://ollama.com), pull a model, and point `.env` at it. No key is needed and the notes never leave the machine.

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen3.6:27b
LLM_MAX_TOKENS_PARAM=max_tokens
```

## Confluence

```
CONFLUENCE_BASE_URL=https://your-site.atlassian.net/wiki
CONFLUENCE_EMAIL=you@example.com
CONFLUENCE_API_TOKEN=...
CONFLUENCE_SPACE_KEY=OPS
CONFLUENCE_PARENT_PATH=Knowledge Base/RCAs
```

Create the API token at [id.atlassian.com/manage-profile/security/api-tokens](https://id.atlassian.com/manage-profile/security/api-tokens). The account needs permission to add pages and attachments in the space.

`CONFLUENCE_PARENT_PATH` is the chain of page titles the RCA goes under. On approval the app:

1. finds the space and walks the path to the parent page,
2. creates the page `RCA – <incident ID> – <title>` with the summary, key facts, a link to the PDF and the information gaps,
3. attaches the PDF,
4. adds the labels `rca`, `incident`, the severity and the service names.

Publishing a later version updates the same page and adds a new version of the attachment. The app labels its pages `rca-assistant` and will not overwrite a page with the same title that it did not create. If the path does not exist, publishing stops with a message; set `CONFLUENCE_CREATE_MISSING_PARENTS=true` to have it created.

## Tests

```bash
python -m pytest
```

The suite runs offline. It uses scripted model replies and an in-memory stand-in for Confluence, and covers gap detection, the skip path, loop limits, validate-and-repair, rate-limit handling, the evidence check, retry after a failed step, restart, reopening, PDF content and publishing.

## Project layout

```
app/streamlit_app.py        the chat UI
rca/config.py               settings from .env
rca/llm.py                  model client: JSON schema, validate and repair, rate-limit waits
rca/llm_fake.py             scripted stand-in for tests and demo mode
rca/schema/                 field_specs.py (the RCA layout), facts.py, llm_io.py
rca/agents/                 extractor, planner, clarifier, drafter, reviewer
rca/gaps.py                 what is missing, in what order to ask
rca/graph/                  the workflow: steps, routing, pauses for the user
rca/memory/                 short_term.py (session checkpoints), long_term.py
rca/render/                 document view, HTML template, PDF
rca/publish/confluence.py   Confluence Cloud publisher
rca/security/               redact.py (masks secrets), evidence.py (quote check)
rca/timeutil.py             time parsing, durations, UTC conversion
rca/service.py              start / reply / decide / reopen, used by the UI
scripts/                    bakeoff.py, run_notes.py
samples/                    two RCAs produced from the sample notes, and two screenshots
tests/                      test suite and sample notes
```

## Limits to know about

- **Your notes are sent to the model provider.** With the Groq free tier that means Groq's servers. Check your company's data policy before using real incidents, or run a model locally.
- **Secret masking is best-effort.** Common key and token formats and `password=...` style text are masked before anything is stored or sent. A secret that looks like an ordinary word is not recognised.
- **Timezone abbreviations are ambiguous.** `IST` is read as India, `CST` as US Central, and so on. The PDF states which zone was assumed.
- **No login.** The name typed in the sidebar is trusted. This is a single-user local app; put it behind single sign-on before sharing it.
- **Free-tier rate limits.** When the provider says to slow down the app waits and retries, up to `LLM_MAX_RATE_WAIT_S`. If a step still fails, the session keeps your answers and offers a retry.
- **Confluence publishing is tested against a stand-in only.** The publisher's requests are checked by the test suite against an in-memory copy of the Confluence endpoints. It has not yet been run against a real Confluence site, so expect to adjust it on first use.
- **Related documents** lists similar past RCAs made with this tool. It does not yet search your existing SOPs and runbooks in Confluence.
