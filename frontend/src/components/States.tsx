interface LoadingStateProps {
  message?: string;
  detail?: string;
}

/** Shown while a request is in flight. Never a blank screen. */
export function LoadingState({
  message = 'Working…',
  detail,
}: LoadingStateProps) {
  return (
    <div className="state state--loading" role="status" aria-live="polite">
      <div className="spinner" aria-hidden="true" />
      <p className="state__title">{message}</p>
      {detail && <p className="state__detail">{detail}</p>}
    </div>
  );
}

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
  retryLabel?: string;
}

/** A user-facing failure. Never renders a stack trace. */
export function ErrorState({
  title = 'Something went wrong',
  message,
  onRetry,
  retryLabel = 'Try again',
}: ErrorStateProps) {
  return (
    <div className="state state--error" role="alert">
      <p className="state__title">
        <span aria-hidden="true">✕ </span>
        {title}
      </p>
      <p className="state__detail">{message}</p>
      {onRetry && (
        <button type="button" className="button button--secondary" onClick={onRetry}>
          {retryLabel}
        </button>
      )}
    </div>
  );
}

interface EmptyStateProps {
  title: string;
  message?: string;
}

/** A legitimate "nothing here" state, distinct from an error. */
export function EmptyState({ title, message }: EmptyStateProps) {
  return (
    <div className="state state--empty">
      <p className="state__title">{title}</p>
      {message && <p className="state__detail">{message}</p>}
    </div>
  );
}
