import { useEffect, useState } from 'react';
import { api, ApiError, BASE_URL } from './api/client';
import type { HealthResponse } from './api/client';

type Probe =
  | { state: 'loading' }
  | { state: 'ok'; data: HealthResponse }
  | { state: 'error'; message: string };

/**
 * Foundation shell.
 *
 * Deliberately minimal: it proves the frontend builds, starts, and can reach
 * the backend. The claim input, evidence graph and inspector arrive with the
 * later implementation phases.
 */
function App() {
  const [probe, setProbe] = useState<Probe>({ state: 'loading' });

  useEffect(() => {
    let cancelled = false;

    api
      .health()
      .then((data) => {
        if (!cancelled) setProbe({ state: 'ok', data });
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const message =
          error instanceof ApiError
            ? error.message
            : 'Could not reach the backend.';
        setProbe({ state: 'error', message });
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="shell">
      <header className="shell__header">
        <p className="shell__eyebrow">LLM Semester Project</p>
        <h1 className="shell__title">ClaimChain AI</h1>
        <p className="shell__subtitle">
          Evidence-grounded claim verification and evidence mapping.
        </p>
      </header>

      <section className="card">
        <h2 className="card__title">Backend status</h2>

        {probe.state === 'loading' && <p className="muted">Checking…</p>}

        {probe.state === 'error' && (
          <p className="status status--down">
            {probe.message}
            <span className="status__hint">
              Expected backend at {BASE_URL}. Start it with{' '}
              <code>uvicorn app.main:app</code>.
            </span>
          </p>
        )}

        {probe.state === 'ok' && (
          <dl className="facts">
            <div className="facts__row">
              <dt>API</dt>
              <dd className="status status--up">{probe.data.status}</dd>
            </div>
            <div className="facts__row">
              <dt>Version</dt>
              <dd>{probe.data.version}</dd>
            </div>
            <div className="facts__row">
              <dt>Environment</dt>
              <dd>{probe.data.environment}</dd>
            </div>
            <div className="facts__row">
              <dt>Database</dt>
              <dd
                className={
                  probe.data.database === 'ok'
                    ? 'status status--up'
                    : 'status status--down'
                }
              >
                {probe.data.database}
              </dd>
            </div>
            <div className="facts__row">
              <dt>LLM key</dt>
              <dd>{probe.data.llm_configured ? 'configured' : 'not set'}</dd>
            </div>
            <div className="facts__row">
              <dt>Search key</dt>
              <dd>{probe.data.search_configured ? 'configured' : 'not set'}</dd>
            </div>
          </dl>
        )}
      </section>

      <section className="card card--muted">
        <h2 className="card__title">Implementation status</h2>
        <p className="muted">
          Foundation build. Health endpoint and project scaffolding only — no
          pipeline stages, retrieval, or evidence graph yet.
        </p>
        <p className="muted">
          Runtime stage order when implemented:{' '}
          <code>1 → 2 → 3 → 5 → 4 → 6</code>
        </p>
      </section>
    </main>
  );
}

export default App;
