/**
 * API client for the ClaimChain backend.
 *
 * Single place that knows the backend's base URL and error shape. Nothing
 * else in the app should call `fetch` directly.
 *
 * The browser only ever talks to the FastAPI backend. It never holds an API
 * key — Gemini and Exa keys live server-side and are never exposed here.
 */

const BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ??
  'http://127.0.0.1:8000';

/** Shape of the backend health probe. */
export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  database: string;
  llm_configured: boolean;
  search_configured: boolean;
}

/** Error raised for any non-2xx response. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  });

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body?.detail) detail = body.detail;
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
};

export { BASE_URL };
