/**
 * Typed API client for the ClaimChain backend.
 *
 * One place knows the base URL and the error shape. Nothing else in the app
 * calls `fetch` directly.
 *
 * The browser only ever talks to the FastAPI backend. It never holds an API key:
 * Gemini and Exa keys live server-side and are never exposed here. No LLM call
 * is made from this application.
 */

import type {
  ClaimReport,
  ClaimResponse,
  CreateClaimResponse,
  HealthResponse,
} from './types';

/**
 * Base URL of the backend.
 *
 * Configurable via `VITE_API_BASE_URL` (see `.env.example`). Defaults to the
 * local development port 8001. Never hard-coded at call sites.
 */
export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  'http://127.0.0.1:8001';

/** Error raised for any non-2xx response, with a message safe to show a user. */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

/** True when the failure looks like the backend is not reachable at all. */
export function isNetworkError(error: unknown): boolean {
  return error instanceof TypeError;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
      ...init,
    });
  } catch (error) {
    // fetch rejects with TypeError on connection failure. Do not leak the
    // underlying error: the user needs to know the backend is unreachable,
    // not the browser's internal message.
    throw new ApiError(
      0,
      `Could not reach the backend at ${API_BASE_URL}. Is it running?`,
    );
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body?.detail === 'string') {
        detail = body.detail;
      } else if (Array.isArray(body?.detail)) {
        // FastAPI validation errors arrive as a list of objects.
        detail = 'The request was rejected as invalid.';
      }
    } catch {
      // Non-JSON error body; keep the status-based message.
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as T;
}

export const api = {
  /** Liveness and dependency reachability. */
  health: (): Promise<HealthResponse> => request<HealthResponse>('/api/health'),

  /** Submit a claim. Runs Stage 1 decomposition on the backend. */
  createClaim: (rawText: string): Promise<CreateClaimResponse> =>
    request<CreateClaimResponse>('/api/claims', {
      method: 'POST',
      body: JSON.stringify({ raw_text: rawText }),
    }),

  /** Read a claim and its subclaims. */
  getClaim: (claimId: string): Promise<ClaimResponse> =>
    request<ClaimResponse>(`/api/claims/${claimId}`),

  /** Run Stage 2 evidence collection. */
  retrieveEvidence: (claimId: string): Promise<unknown> =>
    request<unknown>(`/api/claims/${claimId}/retrieve`, { method: 'POST' }),

  /** Run Stage 3 evidence analysis. */
  analyzeEvidence: (claimId: string): Promise<unknown> =>
    request<unknown>(`/api/claims/${claimId}/analyze`, { method: 'POST' }),

  /** Run Stage 5 source assessment. */
  assessSources: (claimId: string): Promise<unknown> =>
    request<unknown>(`/api/claims/${claimId}/assess`, { method: 'POST' }),

  /** Run Stage 4 deterministic verification. */
  verifyClaim: (claimId: string): Promise<unknown> =>
    request<unknown>(`/api/claims/${claimId}/verify`, { method: 'POST' }),

  /** Build the Stage 6 report, recording the run. */
  generateReport: (claimId: string): Promise<ClaimReport> =>
    request<ClaimReport>(`/api/claims/${claimId}/report`, { method: 'POST' }),

  /** Read the Stage 6 report without recording a run. */
  getReport: (claimId: string): Promise<ClaimReport> =>
    request<ClaimReport>(`/api/claims/${claimId}/report`),
};
