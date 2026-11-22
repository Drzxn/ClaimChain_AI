import type { ReportEvidence } from '../api/types';
import { formatScore, relationshipStyle } from '../lib/verdicts';
import { VerdictBadge } from './VerdictBadge';

interface EvidenceCardProps {
  evidence: ReportEvidence;
  onSelectSource?: (sourceId: string) => void;
}

/**
 * One evidence link, showing the full chain:
 * evidence link -> passage -> source.
 *
 * The quote is only shown as evidence when the backend marked it validated.
 * A failed quote is surfaced as a limitation rather than displayed as if it
 * were real evidence.
 */
export function EvidenceCard({ evidence, onSelectSource }: EvidenceCardProps) {
  const style = relationshipStyle(evidence.relationship);
  const quoteOk = evidence.quote_validation === 'passed' && evidence.quote;

  return (
    <article className={`evidence-card ${style.className}`}>
      <header className="evidence-card__header">
        <VerdictBadge verdict={evidence.relationship} size="sm" />
        <span className="evidence-card__source-domain">{evidence.source_domain}</span>
      </header>

      {quoteOk ? (
        <blockquote className="evidence-card__quote">{evidence.quote}</blockquote>
      ) : (
        <p className="evidence-card__quote evidence-card__quote--invalid">
          <span aria-hidden="true">⚠ </span>
          Quote could not be verified against the stored passage, so it is not
          shown as evidence.
        </p>
      )}

      {evidence.rationale && (
        <p className="evidence-card__rationale">{evidence.rationale}</p>
      )}

      <details className="evidence-card__details">
        <summary>Passage and scores</summary>
        <p className="evidence-card__passage">{evidence.passage_text}</p>
        <dl className="inline-metrics">
          <div>
            <dt>Relevance</dt>
            <dd>{formatScore(evidence.relevance)}</dd>
          </div>
          <div>
            <dt>Evidence quality</dt>
            <dd>{formatScore(evidence.evidence_quality)}</dd>
          </div>
          <div>
            <dt>Stance confidence</dt>
            <dd>{formatScore(evidence.stance_confidence)}</dd>
          </div>
          <div>
            <dt>Chunk</dt>
            <dd>#{evidence.passage_chunk_index}</dd>
          </div>
        </dl>
      </details>

      <footer className="evidence-card__footer">
        <a
          className="link"
          href={evidence.source_url}
          target="_blank"
          rel="noopener noreferrer"
        >
          {evidence.source_title || evidence.source_url}
        </a>
        {onSelectSource && (
          <button
            type="button"
            className="button button--ghost button--sm"
            onClick={() => onSelectSource(evidence.source_id)}
          >
            Source details
          </button>
        )}
      </footer>
    </article>
  );
}
