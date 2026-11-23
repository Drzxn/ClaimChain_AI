/**
 * App orchestration tests.
 *
 * The API client is mocked, so these run the real component tree against
 * scripted responses. No backend, no Gemini, no Exa, no database.
 *
 * What is asserted: the loading state appears, an API failure surfaces as a
 * user-facing error, and a successful pipeline renders the report with the
 * verdict the backend returned.
 */

import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from './App';
import { ApiError, api } from './api/client';
import { canonicalReport } from './test/fixtures';

// Keep the real ApiError class (App branches on it) and stub only the calls.
vi.mock('./api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./api/client')>();
  return {
    ...actual,
    api: {
      health: vi.fn(),
      createClaim: vi.fn(),
      getClaim: vi.fn(),
      retrieveEvidence: vi.fn(),
      analyzeEvidence: vi.fn(),
      assessSources: vi.fn(),
      verifyClaim: vi.fn(),
      generateReport: vi.fn(),
      getReport: vi.fn(),
    },
  };
});

const CLAIM_TEXT = 'Using HTTPS makes a website completely secure.';

/** Wire every pipeline call to succeed, returning the canonical report. */
function mockHappyPath() {
  vi.mocked(api.createClaim).mockResolvedValue({
    id: 'claim-1',
    status: 'queued',
    domain: 'technology',
    subclaim_count: 3,
  });
  vi.mocked(api.retrieveEvidence).mockResolvedValue({});
  vi.mocked(api.analyzeEvidence).mockResolvedValue({});
  vi.mocked(api.assessSources).mockResolvedValue({});
  vi.mocked(api.verifyClaim).mockResolvedValue({});
  vi.mocked(api.generateReport).mockResolvedValue(canonicalReport());
}

async function submitClaim() {
  const user = userEvent.setup();
  render(<App />);
  await user.type(screen.getByRole('textbox'), CLAIM_TEXT);
  await user.click(screen.getByRole('button', { name: /verify claim/i }));
  return user;
}

afterEach(() => {
  vi.clearAllMocks();
});

describe('App', () => {
  it('starts on the claim input screen', () => {
    render(<App />);
    expect(screen.getByRole('heading', { name: /verify a claim/i })).toBeInTheDocument();
  });

  it('shows a loading state while the pipeline runs', async () => {
    // Hold the first call open so the loading state is observable.
    let release: (value: unknown) => void = () => {};
    vi.mocked(api.createClaim).mockImplementation(
      () => new Promise((resolve) => {
        release = resolve as (value: unknown) => void;
      }) as never,
    );

    await submitClaim();

    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(screen.getByText(/verifying claim/i)).toBeInTheDocument();
    // The first pipeline step is named in both the status detail and the
    // step list, so assert on the list item rather than a unique match.
    expect(
      screen.getAllByText(/decomposing the claim into subclaims/i).length,
    ).toBeGreaterThan(0);

    release({ id: 'claim-1', status: 'queued', domain: 'technology', subclaim_count: 3 });
  });

  it('calls the API with the submitted claim text', async () => {
    mockHappyPath();
    await submitClaim();

    await waitFor(() => {
      expect(api.createClaim).toHaveBeenCalledWith(CLAIM_TEXT);
    });
  });

  it('drives the stages in runtime order, Stage 5 before Stage 4', async () => {
    mockHappyPath();
    await submitClaim();

    await waitFor(() => expect(api.generateReport).toHaveBeenCalled());

    const assessOrder = vi.mocked(api.assessSources).mock.invocationCallOrder[0];
    const verifyOrder = vi.mocked(api.verifyClaim).mock.invocationCallOrder[0];
    expect(assessOrder).toBeLessThan(verifyOrder);
  });

  it('shows an error state when the backend is unreachable', async () => {
    vi.mocked(api.createClaim).mockRejectedValue(
      new ApiError(0, 'Could not reach the backend.'),
    );

    await submitClaim();

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeInTheDocument();
    });
    expect(screen.getByText(/verification failed/i)).toBeInTheDocument();
    expect(screen.getByText(/backend is not reachable/i)).toBeInTheDocument();
  });

  it('shows an error state when a pipeline stage fails', async () => {
    mockHappyPath();
    vi.mocked(api.analyzeEvidence).mockRejectedValue(
      new ApiError(502, 'The model did not return a valid analysis.'),
    );

    await submitClaim();

    await waitFor(() => {
      expect(screen.getByText(/verification failed/i)).toBeInTheDocument();
    });
    expect(
      screen.getByText(/the model did not return a valid analysis/i),
    ).toBeInTheDocument();
  });

  it('renders the report after a successful pipeline', async () => {
    mockHappyPath();
    await submitClaim();

    await waitFor(() => {
      expect(screen.getByText(/verification report/i)).toBeInTheDocument();
    });
    // The claim text is echoed back from the backend report.
    expect(screen.getByText(CLAIM_TEXT)).toBeInTheDocument();
  });

  it('does not fabricate a report when a stage fails', async () => {
    mockHappyPath();
    vi.mocked(api.verifyClaim).mockRejectedValue(new ApiError(500, 'boom'));

    await submitClaim();

    await waitFor(() => {
      expect(screen.getByText(/verification failed/i)).toBeInTheDocument();
    });
    expect(screen.queryByText(/verification report/i)).not.toBeInTheDocument();
  });

  it('returns to the claim input from the report', async () => {
    mockHappyPath();
    const user = await submitClaim();

    await waitFor(() => {
      expect(screen.getByText(/verification report/i)).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /verify another claim/i }));

    expect(screen.getByRole('heading', { name: /verify a claim/i })).toBeInTheDocument();
  });
});
