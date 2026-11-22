import type { ReportSource } from '../api/types';
import { formatDate, formatDateOnly, formatScore } from '../lib/verdicts';

interface SourceDetailsProps {
  source: ReportSource | undefined;
  onClose: () => void;
}

/**
 * Details for one source, including its Stage 5 assessment.
 *
 * Authority and recency are shown as assessed. A null score is displayed as
 * "not assessed" rather than as zero, because the two mean different things:
 * zero is a judgement, null is the absence of one.
 */
export function SourceDetails({ source, onClose }: SourceDetailsProps) {
  if (!source) {
    return (
      <aside className="source-panel">
        <header className="source-panel__header">
          <h3 className="section-heading">Source</h3>
          <button type="button" className="button button--ghost button--sm" onClick={onClose}>
            Close
          </button>
        </header>
        <p className="text--muted">Select a source to inspect it.</p>
      </aside>
    );
  }

  return (
    <aside className="source-panel">
      <header className="source-panel__header">
        <h3 className="section-heading">Source</h3>
        <button type="button" className="button button--ghost button--sm" onClick={onClose}>
          Close
        </button>
      </header>

      <h4 className="source-panel__title">
        {source.title || source.domain}
      </h4>

      <a
        className="link source-panel__url"
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
      >
        {source.url}
      </a>

      <dl className="detail-list">
        <div>
          <dt>Domain</dt>
          <dd>{source.domain}</dd>
        </div>
        <div>
          <dt>Publisher</dt>
          <dd>{source.publisher || '—'}</dd>
        </div>
        <div>
          <dt>Source type</dt>
          <dd>
            <span className="tag">{source.source_type}</span>
          </dd>
        </div>
        <div>
          <dt>Published</dt>
          <dd>{formatDateOnly(source.published_at)}</dd>
        </div>
        <div>
          <dt>Retrieved</dt>
          <dd>{formatDate(source.retrieved_at)}</dd>
        </div>
        <div>
          <dt>Authority</dt>
          <dd>
            {source.authority_score === null
              ? 'not assessed'
              : formatScore(source.authority_score)}
          </dd>
        </div>
        <div>
          <dt>Recency</dt>
          <dd>
            {source.recency_score === null
              ? 'not assessed'
              : formatScore(source.recency_score)}
          </dd>
        </div>
        <div>
          <dt>Cites primary sources</dt>
          <dd>
            {source.cites_primary_sources === null
              ? 'undetermined'
              : source.cites_primary_sources
                ? 'yes'
                : 'no'}
          </dd>
        </div>
        <div>
          <dt>Evidence links</dt>
          <dd>{source.evidence_count}</dd>
        </div>
      </dl>

      {source.is_duplicate && (
        <p className="source-panel__note">
          <span aria-hidden="true">⧉ </span>
          This source duplicates the content of another source and was not
          counted separately.
        </p>
      )}

      {source.assessment_rationale && (
        <div className="source-panel__rationale">
          <h5 className="section-heading">Assessment rationale</h5>
          <p>{source.assessment_rationale}</p>
        </div>
      )}
    </aside>
  );
}
