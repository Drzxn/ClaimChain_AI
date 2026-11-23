# ClaimChain AI

Evidence-grounded claim verification and evidence mapping.

A complex statement is decomposed into atomic subclaims, each subclaim is
checked against retrieved source passages, and every relationship between a
passage and a subclaim is stored with its source metadata. The final verdict is
computed deterministically from those stored relationships, so it can always be
traced back to the evidence that produced it.

**"Insufficient evidence" is a legitimate outcome.** The system does not force a
verdict when the evidence does not support one.

---

## Status

**Core implementation complete.** The repository contains the full six-stage
claim verification pipeline, React frontend with evidence graph and inspector UI,
PostgreSQL persistence, deterministic verification, explainable reports, and a
deterministic evaluation harness.

| Area | State |
| --- | --- |
| Backend skeleton + `GET /api/health` | done |
| Stage 1 + claims API (`POST`/`GET`) | done |
| Frontend skeleton + backend connectivity | done |
| Docker PostgreSQL | done |
| Database schema (8 tables) + migration | done |
| Stage 1 — decomposition | done |
| Stage 2 — evidence collection | done |
| Stage 3 — evidence analysis | done |
| Stage 5 — source assessment | done |
| Stage 4 — verification | done |
| Stage 6 — explainable report | done |
| Evidence graph + inspector UI | done |
| Evaluation harness | done |

---

## Architecture

```
React + TypeScript + React Flow  â”€â”€HTTP/JSONâ”€â”€â–¶  FastAPI (single monolith)
                                                    â”œâ”€â”€ pipeline/   six stages
                                                    â”œâ”€â”€ services/   llm, search, fetch, extract, chunk, cache
                                                    â””â”€â”€ PostgreSQL  single source of truth
```

The evidence graph is **derived on read** from PostgreSQL rows. It is never
stored as a separate artifact, so the graph and the data cannot drift apart.

### Approved v1 decisions

- **Search:** Exa
- **LLM:** Google Gemini API, model `gemini-3.5-flash-lite`
- **Structured output:** JSON only, `temperature=0`, schema validation, one
  repair retry, then hard failure. Never a lenient parse.
- **Aggregation:** deterministic, version `agg-1.0.0`. The LLM never produces
  the overall verdict.
- **Subclaim weights:** fixed at `1.0`.
- **Deferred:** embeddings/RAG, vector database, NetworkX in the core pipeline.
- **Transport:** HTTP polling. No SSE, no WebSockets.
- **State:** TanStack Query and local state. No Redux, no Zustand.
- **Auth:** none. Local academic demo only.
- **Deployment:** local.

### Stage execution order

```
Stage 1 â†’ Stage 2 â†’ Stage 3 â†’ Stage 5 â†’ Stage 4 â†’ Stage 6
```

**Stage numbers are conceptual/rubric labels, not runtime execution order.**
Stage 5 (source assessment) runs before Stage 4 (verification) because the
deterministic aggregation in Stage 4 consumes the source weights that Stage 5
produces. Running them in numeric order would give every link a weight of zero.

---

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker Desktop

### 1. Environment

From the repository root:

```powershell
Copy-Item .env.example .env

### 2. PostgreSQL

```powershell
docker compose up -d
docker compose ps
```

The database is published on host port **5433**, not 5432, so it never collides
with a PostgreSQL installation on the host. Data persists in the
`claimchain_pgdata` volume.

```powershell
docker compose down      # stop, keep data
docker compose down -v   # stop, delete data
```

### 3. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

- Health: <http://127.0.0.1:8000/api/health>
- API docs: <http://127.0.0.1:8000/docs>

Expected response:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "environment": "development",
  "database": "ok",
  "llm_configured": true,
  "search_configured": true
}

`database` reads `ok` when the Docker container is reachable and `unavailable`
when it is not. The endpoint always returns HTTP 200 while the process is alive,
so a stopped container looks like a database problem rather than a crashed app.

### 4. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The page reports live backend status.

### 5. Using the application

1. Start PostgreSQL with Docker.
2. Start the FastAPI backend on port `8000`.
3. Start the React frontend on port `5173`.
4. Open `http://localhost:5173`.
5. Enter a claim to verify.
6. Submit the claim and wait for the verification pipeline to complete.
7. Review the overall verdict, subclaim verdicts, supporting and contradicting
   evidence, source assessments, and the interactive evidence graph.
8. Select graph nodes to inspect the corresponding claim, subclaim, evidence
   passage, or source.

The frontend communicates with the backend through the API base URL configured
by `VITE_API_BASE_URL`.

### 6. Migrations

```powershell
cd backend
.\.venv\Scripts\python.exe -m alembic upgrade head    # apply
.\.venv\Scripts\python.exe -m alembic current         # show revision
.\.venv\Scripts\python.exe -m alembic check           # detect model drift
.\.venv\Scripts\python.exe -m alembic downgrade base  # drop everything
```

The database URL is read from `DATABASE_URL` by `alembic/env.py`. No credential
is stored in `alembic.ini`.

### 7. Round-trip check

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.db.seed_roundtrip
```

Inserts a claim and its full relationship tree, reads it back, verifies the
cascade, then deletes it. Exits non-zero if the schema or ORM is broken, and
leaves no data behind.

### 8. Tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
```

Database tests run against an isolated `claimchain_test` database on the same
container, with the real migration applied and every test rolled back. The
development database is never touched.

The live API smoke tests are opt-in and excluded from the default run, so the
suite never spends quota:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/smoke -q -m smoke
```

`test_gemini_smoke.py` needs `GEMINI_API_KEY`; `test_exa_smoke.py` needs
`EXA_API_KEY`. Each skips cleanly when its key is absent.

### 9. Stage 2 — evidence collection

```powershell
# after POST /api/claims has decomposed a claim
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/claims/<id>/retrieve
```

Runs search, dedupe, SSRF-checked fetching, extraction, chunking and the
relevance prefilter, then persists `sources` and `passages`. It creates **no**
evidence links and **no** verdicts: classification is Stage 3.

**Fixture mode** runs the whole stage offline with no `EXA_API_KEY` and no
network access, using the committed dataset in `app/services/fixtures.py`.
Enable it with `USE_FIXTURES=true` in `.env`. Fixture runs are recorded with
`runs.cache_hit = true` so their provenance is never ambiguous.

Hard limits (constants in `app/pipeline/stage2_retrieve.py`):

| Limit | Value |
| --- | --- |
| Queries per subclaim | 3 |
| Results per query | 5 |
| Total URLs per claim | 30 |
| Pages fetched per claim | 20 |
| Fetch timeout | 10 s |
| Response size cap | 2 MB |
| Redirects | 5, revalidated each hop |

### 10. Stage 3 — evidence analysis

```powershell
# after Stage 2 has produced sources and passages
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/claims/<id>/analyze
```

Classifies each candidate passage against its subclaim as `SUPPORTS`,
`CONTRADICTS`, `MIXED` or `INSUFFICIENT`, and persists one `evidence_links` row
per (subclaim, passage). It produces **no** verdicts: aggregation is Stage 4.

One LLM call per subclaim, carrying that subclaim's passages. Subclaims are never
merged into a single call.

Deterministic checks on every response:

- **Schema** — relationship must be one of the four tokens; scores must be in
  `[0,1]`. `SUPPORTED`/`CONTRADICTED`/`MISLEADING` are rejected by name, because
  they mean the model answered the overall question instead of the link.
- **Quote** — must be a verbatim substring of that passage, tolerating whitespace
  re-wrapping only. A failed quote is stored with `quote_validation = 'failed'`
  and the unverifiable text is discarded, never kept as if it were evidence.
- **Coverage** — a passage the model skipped becomes an `INSUFFICIENT` link with
  zero confidence and a system rationale, so the abstention is visible.
- **Failure rate** — two or more failed quotes *and* more than half failing
  fails the stage, which is what a broken prompt looks like.

Retrieved passages are untrusted data. The prompt states that passage content is
never an instruction, and the schema rejects any attempt to smuggle a verdict
token through it.

### 11. Stage 5 — source assessment

```powershell
# after Stage 2 has produced sources
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/claims/<id>/assess
```

Scores each source for authority and recency, classifies its type, and records
the assessment on the `sources` row. It produces **no** verdicts: aggregation is
Stage 4.

**Authority and recency are always computed by backend logic**, never by the
model. Those two numbers feed Stage 4's weighting, and a verdict must be
reproducible from stored data; a model-produced score would break that.

Authority is a tier lookup (`services/source_tiers.py`) — a data table, not
logic, so it is auditable and unit-testable:

| Tier | Score | Examples |
| --- | --- | --- |
| Government / standards | 1.0 | `.gov`, `.gov.uk`, nist.gov, ietf.org, w3.org |
| Academic / research | 0.9 | `.edu`, `.ac.uk`, arxiv.org, nature.com |
| Established reference | 0.8 | wikipedia.org, britannica.com |
| News / established org | 0.7 | reuters.com, bbc.com, owasp.org, python.org |
| General web (default) | 0.5 | anything unmatched |
| Low-signal | 0.3 | medium.com, `.xyz`, content farms |

Matching is exact host, then suffix (so `docs.python.org` inherits `python.org`),
then TLD, then publisher-name hint.

Recency is piecewise-linear decay from `published_at`, measured against
`retrieved_at` so it is reproducible from stored data:

```
age_days = retrieved_at.date() - published_at
published_at is null      -> 0.5   (neutral, not punished)
age <= window             -> 1.0
age <= 1095               -> 1.0 - 0.4 * (age - window) / (1095 - window)
else                      -> max(0.2, 0.6 - 0.4 * (age - 1095) / 1825)
future dates              -> 1.0 (clamped)
```

`window` is 365 days, or 180 for `technology` claims, because technology ages
faster. The 0.2 floor keeps an old-but-relevant source usable rather than
zeroing it.

`source_type` and `cites_primary_sources` may be enriched by one cached LLM call
per unique `content_hash`. If the provider is unavailable, the stage still
completes: `source_type` falls back to a deterministic guess from the domain
tier, and `cites_primary_sources` stays `NULL` rather than being invented.

Authority is not truth. A government site can publish something wrong; a blog can
publish something right. This stage describes the publisher — Stage 3 already
judged the passage.

### 12. Stage 4 — verification (deterministic)

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/claims/<id>/verify
```

Computes one `subclaim_verdicts` row per subclaim and one `verdicts` row per
claim, from stored evidence links and source scores.

**No model is involved.** `stage4_verify.py` imports no LLM, and a test asserts
that structurally. The verdict must be reproducible from stored rows alone.

Link weight is the five-factor product:

```
w = authority × recency × relevance × evidence_quality × stance_confidence
```

Links with `w < 0.15` are excluded from all sums but left in the database.
`INSUFFICIENT` links never contribute weight. A `NULL` authority or recency
score yields `w = 0`, so an unassessed source cannot move a verdict.

Subclaim verdict order (fixed):

```
T < 0.25                        -> INSUFFICIENT
C == 0 and M == 0               -> SUPPORTED
S == 0 and M == 0               -> CONTRADICTED
M > 0 with S or C               -> MIXED
S > 0 and C > 0                 -> dominance margin decides
otherwise                       -> INSUFFICIENT
```

Overall verdict order (fixed, `agg-1.0.0`):

```
coverage < 0.50                                    -> INSUFFICIENT_EVIDENCE
score > 0.20 and contradicted_ratio >= 0.20        -> MISLEADING
score >= 0.60                                      -> SUPPORTED
score <= -0.60                                     -> CONTRADICTED
otherwise                                          -> MIXED
```

Canonical regression: the HTTPS claim yields SUPPORTED / SUPPORTED /
CONTRADICTED, overall **MISLEADING** with score `0.3333`, coverage `1.0000`,
contradicted ratio `0.3333`.

### 13. Stage 6 — explainable report

```powershell
# build, record the run, and stamp report_generated_at
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/claims/<id>/report

# read the same report without writing anything
Invoke-RestMethod http://127.0.0.1:8000/api/claims/<id>/report

# optional model-written prose (off by default)
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/api/claims/<id>/report?narrative=true"
```

Projects the persisted Stage 1–5 results into a structured, traceable report.
**Stage 4 owns the verdict**; Stage 6 copies it verbatim and never recalculates
it. A test corrupts the stored verdict and asserts the report echoes the
corruption rather than "fixing" it.

Report shape:

```
claim
verdict            <- copied from `verdicts`, never recomputed
subclaims[]        <- each with its persisted subclaim_verdict
  evidence[]       <- evidence_link_id -> passage_id -> source_id
sources[]          <- with Stage 5 authority/recency/type assessment
explanation        <- deterministic statements + optional narrative
limitations[]      <- each with provenance (verdict / evidence / sources)
```

Every evidence item carries the full identity chain, so the frontend can build
the graph without a second query and any displayed quote can be traced to the
passage it came from.

**No schema change.** The report is a projection, not a stored artifact: it is
rebuilt from authoritative rows on every request, so there is nothing to keep in
sync. The only field Stage 6 writes is `verdicts.report_generated_at`, which
exists for exactly that purpose.

**Optional LLM prose.** Off by default; the report is fully deterministic
without it. When enabled, the model writes exactly one field,
`explanation.narrative`, and receives the verdict as a fixed fact. The verdict
fields come from the database, not the model, so prose cannot corrupt them — the
worst case is a bad sentence beside a correct verdict. A narrow guard also
discards a narrative that explicitly asserts a different claim-level verdict,
while still allowing correct prose like "two of three subclaims were supported"
under a MISLEADING verdict.

Re-running is safe: each generation records its own `runs` attempt.

---

## Evaluation

A small deterministic harness measures whether the verification pipeline reaches
the right verdict for a set of known claims.

```powershell
cd backend
.\.venv\Scripts\python.exe -m app.eval.run
```

**Offline and reproducible.** No API keys, no network, no database. The harness
feeds each case's evidence links straight into the real Stage 4 aggregation
functions (`aggregate_subclaim`, `aggregate_overall`), so the verdicts under test
come from the same code that serves the API, not a parallel copy.

Options:

```powershell
--dataset PATH       # use a different dataset file
--output-dir PATH    # write artifacts elsewhere
--no-write           # print the summary only
```

### Dataset

`backend/app/eval/dataset/claims.v1.json` — 12 hand-curated cases covering all
five outcomes: SUPPORTED, CONTRADICTED, MIXED, MISLEADING and
INSUFFICIENT_EVIDENCE.

Each case states the subclaims a decomposition would produce and the
relationship each retrieved passage was judged to have. Expected verdicts were
derived from the frozen `agg-1.0.0` rules, not fitted to the output.

**This is a manually curated demonstration set, not a statistically
representative benchmark.** It exists to show the decision layer behaves
correctly on clear-cut and adversarial cases, and to catch regressions. Twelve
cases cannot support claims about real-world accuracy.

### What the evaluation does and does not measure

Stages 1, 2, 3 and 5 are **not executed**: decomposition, retrieval and source
assessment need live Gemini and Exa, which would make a run non-deterministic and
quota-dependent. The dataset supplies the evidence links those stages would have
produced.

So the harness measures the **deterministic decision layer**: given this
evidence, does the system reach the right verdict? It does not measure retrieval
quality or decomposition quality.

### Metrics

- Accuracy
- Macro precision, recall and F1 (averaged over classes that actually occur)
- Per-class precision, recall, F1 and support
- Confusion matrix with every cell present

Macro averages skip classes with zero support, so an absent label cannot drag the
score down. Zero denominators yield `0.0` rather than an error.

### Results

Written to `backend/app/eval/results/` as `eval_<timestamp>.json` (full report,
including every case) and `eval_<timestamp>.csv` (one row per case). Both are
generated artifacts and git-ignored; only the directory README is tracked.

A case that raises is recorded as a failure with its error message and excluded
from the metrics, so a run cannot look clean while cases vanished. The CLI
returns a non-zero exit code if any case failed.

The current v1.0.0 evaluation run contains 12 cases covering all five verdict
classes.

| Metric | Result |
| --- | ---: |
| Cases evaluated | 12 / 12 |
| Accuracy | 1.0000 |
| Macro precision | 1.0000 |
| Macro recall | 1.0000 |
| Macro F1 | 1.0000 |

---

## Environment variables

| Variable | Purpose |
| --- | --- |
| `GEMINI_API_KEY` | Google Gemini API key. Server-side only. |
| `GEMINI_MODEL` | Model id. Default `gemini-3.5-flash-lite`. |
| `EXA_API_KEY` | Exa search key. Server-side only. |
| `DATABASE_URL` | SQLAlchemy URL. Defaults to the Docker service on 5433. |
| `APP_ENV` | `development` / `test` / `production`. |
| `LOG_LEVEL` | Python log level. |
| `FRONTEND_ORIGIN` | CORS origin. Default `http://localhost:5173`. |
| `LLM_TEMPERATURE` | `0`. |
| `LLM_MAX_REPAIR_RETRIES` | `1`. |
| `MAX_SUBCLAIMS` | `8`. |
| `MAX_CANDIDATES_PER_SUBCLAIM` | `8`. |
| `PER_CLAIM_TOKEN_BUDGET` | `60000`. |
| `USE_FIXTURES` | Replay committed fixture responses instead of live external calls. |
| `VITE_API_BASE_URL` | Frontend â†’ backend base URL. |

---

## Security

**Real API keys must never be committed to this repository.**

- `.env` is git-ignored. Only `.env.example` is tracked, and it contains empty
  placeholders.
- Keys are read server-side via `pydantic-settings`. They are never sent to the
  browser, never logged, and never returned in an API response. The health
  endpoint reports only a boolean `configured` flag.
- The frontend talks only to the FastAPI backend. It holds no credentials.
- Retrieved web content is treated as **untrusted data**, never as
  instructions. Stage 3 validates structured model output and prevents
  retrieved content from being interpreted as instructions.

---

## Repository layout

```
claimchain/
├── backend/
│   ├── app/
│   │   ├── main.py            app factory, CORS, router mount
│   │   ├── config.py          typed settings from env
│   │   ├── db/                engine, session, declarative base
│   │   ├── api/               HTTP API routers
│   │   ├── schemas/            pydantic request/response models
│   │   ├── pipeline/           six stages + deterministic aggregation
│   │   ├── services/           llm, search, fetch, extract, chunk, cache
│   │   ├── prompts/            versioned prompt templates
│   │   └── eval/               dataset, harness, metrics
│   ├── alembic/                migrations
│   ├── tests/                  unit, contract, integration, fixtures
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── api/                typed backend client and API types
│       ├── components/         claim input, report, graph, inspector, verdict UI
│       └── lib/                graph layout and verdict utilities
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## Implementation status

1. Repo foundation — **done**
2. Database — **done**
3. Stage 1 + API skeleton — **done**
4. Retrieval layer — **done**
5. Stage 3 — **done**
6. Stage 5 + Stage 4 — **done**
7. Stage 6 + evidence graph — **done**
8. Frontend — **done**
9. Evaluation harness — **done**
10. Hardening and validation — **done**

The current repository represents the completed v1 implementation. Further work
should focus on documentation, demonstration, evaluation expansion, and future
features rather than changing the frozen verification architecture.

---

## Known advisories

`npm audit` reports 5 high-severity advisories in the frontend **dev**
dependencies (a ReDoS in `picomatch`, reached through Vite's glob tooling). They
do not affect the production bundle. The suggested fix downgrades Vite to 6.x,
which is a breaking change; it is deliberately not applied. Revisit when Vite
ships a patched release.

---

## Team

Prashnath G (2362539) · Darshan R (2361010) · Yazhini C (2362542)



