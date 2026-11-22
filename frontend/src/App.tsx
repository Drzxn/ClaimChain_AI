import { useCallback, useState } from 'react';

import { ApiError, api } from './api/client';
import type { ClaimReport } from './api/types';
import { ClaimInput } from './components/ClaimInput';
import { ReportView } from './components/ReportView';
import { ErrorState, LoadingState } from './components/States';

/**
 * Pipeline stages the frontend drives, in the backend's runtime order.
 *
 * Stage numbers are the backend's conceptual labels; the order here is the
 * execution order (1 -> 2 -> 3 -> 5 -> 4 -> 6), which is why 5 comes before 4.
 */
const PIPELINE_STEPS = [
  { key: 'decompose', label: 'Decomposing the claim into subclaims' },
  { key: 'retrieve', label: 'Collecting evidence from the web' },
  { key: 'analyze', label: 'Judging each passage against its subclaim' },
  { key: 'assess', label: 'Assessing source authority and recency' },
  { key: 'verify', label: 'Computing the deterministic verdict' },
  { key: 'report', label: 'Building the explainable report' },
] as const;

type StepKey = (typeof PIPELINE_STEPS)[number]['key'];

type Phase =
  | { kind: 'idle' }
  | { kind: 'running'; step: StepKey; claimId: string | null }
  | { kind: 'report'; report: ClaimReport }
  | { kind: 'error'; message: string; claimId: string | null; retryable: boolean };

/** Turn any thrown value into a message safe to show a user. */
function describeError(error: unknown): { message: string; retryable: boolean } {
  if (error instanceof ApiError) {
    // A 0 status means the request never reached the backend.
    if (error.status === 0) {
      return {
        message:
          'The backend is not reachable. Start it with `uvicorn app.main:app --port 8001` and try again.',
        retryable: true,
      };
    }
    if (error.status === 409) {
      return {
        message: `The pipeline could not continue: ${error.detail}`,
        retryable: false,
      };
    }
    if (error.status === 503) {
      return {
        message:
          'A required service is not configured on the backend (LLM or search key). Check the backend .env.',
        retryable: false,
      };
    }
    if (error.status === 502) {
      return {
        message: `A pipeline stage failed: ${error.detail}`,
        retryable: true,
      };
    }
    return { message: error.detail, retryable: true };
  }

  return {
    message: 'Something unexpected went wrong. Please try again.',
    retryable: true,
  };
}

export default function App() {
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' });
  const [submitError, setSubmitError] = useState<string | null>(null);

  /**
   * Run the full pipeline for a claim.
   *
   * Every step is a real backend call. The frontend never fabricates a result:
   * if a stage fails, the user sees the failure, not a plausible-looking report.
   */
  const runPipeline = useCallback(async (claimText: string) => {
    setSubmitError(null);
    setPhase({ kind: 'running', step: 'decompose', claimId: null });

    let claimId: string | null = null;

    try {
      // Stage 1: create the claim and decompose it.
      const created = await api.createClaim(claimText);
      claimId = created.id;
      setPhase({ kind: 'running', step: 'retrieve', claimId });

      // Stage 2: evidence collection.
      await api.retrieveEvidence(claimId);
      setPhase({ kind: 'running', step: 'analyze', claimId });

      // Stage 3: evidence analysis.
      await api.analyzeEvidence(claimId);
      setPhase({ kind: 'running', step: 'assess', claimId });

      // Stage 5: source assessment (runs before Stage 4).
      await api.assessSources(claimId);
      setPhase({ kind: 'running', step: 'verify', claimId });

      // Stage 4: deterministic verification.
      await api.verifyClaim(claimId);
      setPhase({ kind: 'running', step: 'report', claimId });

      // Stage 6: the report.
      const report = await api.generateReport(claimId);
      setPhase({ kind: 'report', report });
    } catch (error) {
      const { message, retryable } = describeError(error);
      setPhase({ kind: 'error', message, claimId, retryable });
    }
  }, []);

  const handleReset = useCallback(() => {
    setPhase({ kind: 'idle' });
    setSubmitError(null);
  }, []);

  const busy = phase.kind === 'running';

  return (
    <div className="app">
      <header className="app__header">
        <div className="app__brand">
          <span className="app__mark" aria-hidden="true">
            ⛓
          </span>
          <div>
            <h1 className="app__title">ClaimChain AI</h1>
            <p className="app__tagline">
              Evidence-grounded claim verification
            </p>
          </div>
        </div>
        <p className="app__note">
          Verdicts are computed deterministically from retrieved evidence. The
          model never decides the outcome.
        </p>
      </header>

      <main className="app__main">
        {phase.kind === 'idle' && (
          <ClaimInput
            onSubmit={runPipeline}
            busy={busy}
            error={submitError}
          />
        )}

        {phase.kind === 'running' && (
          <LoadingState
            message="Verifying claim"
            detail={
              PIPELINE_STEPS.find((s) => s.key === phase.step)?.label ?? 'Working…'
            }
          />
        )}

        {phase.kind === 'running' && (
          <ol className="pipeline-steps">
            {PIPELINE_STEPS.map((step) => {
              const currentIndex = PIPELINE_STEPS.findIndex(
                (s) => s.key === (phase as { step: StepKey }).step,
              );
              const stepIndex = PIPELINE_STEPS.findIndex((s) => s.key === step.key);
              const state =
                stepIndex < currentIndex
                  ? 'done'
                  : stepIndex === currentIndex
                    ? 'active'
                    : 'pending';
              return (
                <li key={step.key} className={`pipeline-steps__item pipeline-steps__item--${state}`}>
                  <span className="pipeline-steps__marker" aria-hidden="true">
                    {state === 'done' ? '✓' : state === 'active' ? '•' : '·'}
                  </span>
                  {step.label}
                </li>
              );
            })}
          </ol>
        )}

        {phase.kind === 'error' && (
          <ErrorState
            title="Verification failed"
            message={phase.message}
            onRetry={
              phase.retryable && phase.claimId
                ? () => {
                    // Resume from the report step is not safe: earlier stages
                    // may have partially completed. Start a fresh claim instead.
                    handleReset();
                  }
                : handleReset
            }
            retryLabel={phase.retryable ? 'Start over' : 'Back'}
          />
        )}

        {phase.kind === 'report' && (
          <ReportView report={phase.report} onReset={handleReset} />
        )}
      </main>

      <footer className="app__footer">
        <p>
          ClaimChain AI · LLM Semester Project · Prashnath G, Darshan R
        </p>
      </footer>
    </div>
  );
}
