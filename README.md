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

**Foundation build.** The repository contains project scaffolding, a booting
FastAPI backend with a health endpoint, a booting React frontend, and local
PostgreSQL infrastructure. No pipeline stages, retrieval, LLM calls, or evidence
graph are implemented yet.

| Area | State |
| --- | --- |
| Backend skeleton + `GET /api/health` | done |
| Frontend skeleton + backend connectivity | done |
| Docker PostgreSQL | done |
| Database schema (8 tables) | not started |
| Stage 1 — decomposition | not started |
| Stage 2 — evidence collection | not started |
| Stage 3 — evidence analysis | not started |
| Stage 5 — source assessment | not started |
| Stage 4 — verification | not started |
| Stage 6 — explainable report | not started |
| Evidence graph + inspector UI | not started |
| Evaluation harness | not started |

---

## Architecture

```
React + TypeScript + React Flow  ──HTTP/JSON──▶  FastAPI (single monolith)
                                                    ├── pipeline/   six stages
                                                    ├── services/   llm, search, fetch, extract, chunk, cache
                                                    └── PostgreSQL  single source of truth
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
Stage 1 → Stage 2 → Stage 3 → Stage 5 → Stage 4 → Stage 6
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

```powershell
Copy-Item .env.example .env
```

Fill in `GEMINI_API_KEY` and `EXA_API_KEY`. The foundation runs fine without
them — the health endpoint reports them as `not set` and nothing calls them yet.

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
  "llm_configured": false,
  "search_configured": false
}
```

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

### 5. Tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest tests -q
```

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
| `USE_FIXTURES` | Replay recorded responses instead of live calls. Deferred. |
| `VITE_API_BASE_URL` | Frontend → backend base URL. |

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
  instructions. Prompt-injection defenses land with Stage 3.

---

## Repository layout

```
claimchain/
├── backend/
│   ├── app/
│   │   ├── main.py            app factory, CORS, router mount
│   │   ├── config.py          typed settings from env
│   │   ├── db/                engine, session, declarative base
│   │   ├── api/               routers (health only, for now)
│   │   ├── schemas/           pydantic request/response models
│   │   ├── pipeline/          six stages + deterministic aggregation
│   │   ├── services/          llm, search, fetch, extract, chunk, cache
│   │   ├── prompts/           versioned prompt templates
│   │   └── eval/              dataset, harness, metrics
│   ├── alembic/               migrations
│   ├── tests/                 unit, contract, integration, fixtures
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── api/               typed backend client
│       ├── pages/             Home, Claim, Eval
│       ├── components/        graph, inspector, verdict banner
│       ├── types/             shared API types
│       └── lib/               formatting, layout
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## Implementation order

1. Repo foundation — **done**
2. Database — 8 SQLAlchemy models, Alembic initial migration, seed script
3. Stage 1 + API skeleton — decomposition, `POST /api/claims`, `GET /api/claims/{id}`
4. Retrieval layer — Exa, fetch, extract, chunk, cache, SSRF guard, fixture mode
5. Stage 3 — batched analysis call, quote validation, `evidence_links` persisted
6. Stage 5 + Stage 4 — source scoring and deterministic aggregation, unit suite
7. Stage 6 + graph endpoint — report generation, `{nodes, edges}`
8. Frontend — input, polling, verdict banner, React Flow, inspector
9. Evaluation — annotate, run harness, run baseline A, comparison table
10. Hardening and demo — offline mode, error states, demo script

Backend through step 7 is roughly 60% of the work. Do not start step 8 before
step 7 is green.

---

## Known advisories

`npm audit` reports 5 high-severity advisories in the frontend **dev**
dependencies (a ReDoS in `picomatch`, reached through Vite's glob tooling). They
do not affect the production bundle. The suggested fix downgrades Vite to 6.x,
which is a breaking change; it is deliberately not applied. Revisit when Vite
ships a patched release.

---

## Team

Prashnath G (2362539) · Darshan R (2361010)
