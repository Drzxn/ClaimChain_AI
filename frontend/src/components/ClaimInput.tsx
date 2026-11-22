import { useState } from 'react';

const MAX_CLAIM_CHARS = 2000;

const EXAMPLES = [
  'Using HTTPS makes a website completely secure.',
  'Vaccines cause autism.',
  'The Great Wall of China is visible from space.',
];

interface ClaimInputProps {
  onSubmit: (claimText: string) => void;
  /** True while a submission is in flight; the form is disabled. */
  busy: boolean;
  error?: string | null;
}

/**
 * The primary claim entry screen.
 *
 * Client-side validation mirrors the backend's rule (1-2000 characters, not
 * blank) so an obviously invalid claim never costs a round trip. The backend
 * remains the authority: it validates again.
 */
export function ClaimInput({ onSubmit, busy, error }: ClaimInputProps) {
  const [text, setText] = useState('');

  const trimmed = text.trim();
  const tooLong = text.length > MAX_CLAIM_CHARS;
  const blank = trimmed.length === 0;
  const canSubmit = !busy && !blank && !tooLong;

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;
    onSubmit(trimmed);
  }

  return (
    <section className="panel claim-input">
      <header className="panel__header">
        <h1 className="panel__title">Verify a claim</h1>
        <p className="panel__subtitle">
          Enter a statement. ClaimChain decomposes it into atomic subclaims,
          gathers evidence, and reports how that evidence bears on each part.
        </p>
      </header>

      <form onSubmit={handleSubmit} className="claim-input__form">
        <label className="field">
          <span className="field__label">Claim</span>
          <textarea
            className="field__input field__input--textarea"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="e.g. Using HTTPS makes a website completely secure."
            rows={4}
            maxLength={MAX_CLAIM_CHARS + 200}
            disabled={busy}
            aria-describedby="claim-help"
          />
        </label>

        <div className="claim-input__meta" id="claim-help">
          <span className={tooLong ? 'text--danger' : 'text--muted'}>
            {text.length} / {MAX_CLAIM_CHARS} characters
          </span>
          {tooLong && <span className="text--danger">Claim is too long.</span>}
        </div>

        {error && (
          <p className="claim-input__error" role="alert">
            {error}
          </p>
        )}

        <div className="claim-input__actions">
          <button type="submit" className="button button--primary" disabled={!canSubmit}>
            {busy ? 'Verifying…' : 'Verify claim'}
          </button>
          <button
            type="button"
            className="button button--ghost"
            onClick={() => setText('')}
            disabled={busy || blank}
          >
            Clear
          </button>
        </div>
      </form>

      <div className="claim-input__examples">
        <p className="field__label">Try one of these</p>
        <ul className="example-list">
          {EXAMPLES.map((example) => (
            <li key={example}>
              <button
                type="button"
                className="example-list__item"
                onClick={() => setText(example)}
                disabled={busy}
              >
                {example}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
