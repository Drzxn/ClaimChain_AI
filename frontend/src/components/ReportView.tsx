import { useState } from 'react';

import type { ClaimReport } from '../api/types';
import { formatDate, formatScore } from '../lib/verdicts';
import { EvidenceGraph } from './EvidenceGraph';
import { SourceDetails } from './SourceDetails';
import { SubclaimCard } from './SubclaimCard';
import { EmptyState } from './States';
import { VerdictSummary } from './VerdictSummary';

interface ReportViewProps {
  report: ClaimReport;
  onReset: () => void;
}

type Tab = 'subclaims' | 'graph' | 'sources';

/**
 * The main report screen.
 *
 * Three views over the same report: the subclaim breakdown, the evidence graph,
 * and the source list. All three read the same object, so they cannot disagree.
 */
export function ReportView({ report, onReset }: ReportViewProps) {
  const [tab, setTab] = useState<Tab>('subclaims');
  const [selectedSourceId, setSelectedSourceId] = useState<string | null>(null);

  const selectedSource = selectedSourceId
    ? report.sources.find((s) => s.id === selectedSourceId)
    : undefined;

  function handleSelectSource(sourceId: string) {
    setSelectedSourceId(sourceId);
  }

  return (
    <div className="report">
      <header className="report__toolbar">
        <div>
          <p className="report__eyebrow">Verification report</p>
          <p className="report__meta">
            {report.claim.domain} · generated {formatDate(report.generated_at)} ·{' '}
            <code>{report.report_version}</code>
          </p>
        </div>
        <button type="button" className="button button--secondary" onClick={onReset}>
          Verify another claim
        </button>
      </header>

      <VerdictSummary report={report} />

      <section className="panel">
        <h2 className="panel__title panel__title--section">Explanation</h2>
        <p className="explanation__statement">
          {report.explanation.verdict_statement}
        </p>

        {report.explanation.narrative && (
          <div className="explanation__narrative">
            <p className="explanation__narrative-label">
              {report.explanation.narrative_source === 'model'
                ? 'Narrative'
                : 'Summary'}
            </p>
            <p>{report.explanation.narrative}</p>
          </div>
        )}

        {(report.explanation.supporting_points.length > 0 ||
          report.explanation.contradicting_points.length > 0) && (
          <div className="explanation__points">
            {report.explanation.supporting_points.length > 0 && (
              <div>
                <h3 className="section-heading">Supporting evidence</h3>
                <ul className="quote-list quote-list--support">
                  {report.explanation.supporting_points.map((point, i) => (
                    <li key={`support-${i}`}>{point}</li>
                  ))}
                </ul>
              </div>
            )}
            {report.explanation.contradicting_points.length > 0 && (
              <div>
                <h3 className="section-heading">Contradicting evidence</h3>
                <ul className="quote-list quote-list--contradict">
                  {report.explanation.contradicting_points.map((point, i) => (
                    <li key={`contradict-${i}`}>{point}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </section>

      {report.limitations.length > 0 && (
        <section className="panel panel--muted">
          <h2 className="panel__title panel__title--section">Limitations</h2>
          <ul className="limitation-list">
            {report.limitations.map((limitation) => (
              <li key={limitation.code} className="limitation-list__item">
                <span className="tag tag--muted">{limitation.source}</span>
                <span>{limitation.message}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <nav className="tabs" role="tablist" aria-label="Report views">
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'subclaims'}
          className={`tab ${tab === 'subclaims' ? 'tab--active' : ''}`}
          onClick={() => setTab('subclaims')}
        >
          Subclaims ({report.subclaims.length})
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'graph'}
          className={`tab ${tab === 'graph' ? 'tab--active' : ''}`}
          onClick={() => setTab('graph')}
        >
          Evidence graph
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === 'sources'}
          className={`tab ${tab === 'sources' ? 'tab--active' : ''}`}
          onClick={() => setTab('sources')}
        >
          Sources ({report.sources.length})
        </button>
      </nav>

      {tab === 'subclaims' && (
        <section className="subclaim-list">
          {report.subclaims.length === 0 ? (
            <EmptyState
              title="No subclaims"
              message="This claim was not decomposed into subclaims."
            />
          ) : (
            report.subclaims.map((subclaim) => (
              <SubclaimCard
                key={subclaim.id}
                subclaim={subclaim}
                onSelectSource={handleSelectSource}
              />
            ))
          )}
        </section>
      )}

      {tab === 'graph' && (
        <section className="panel">
          <EvidenceGraph report={report} onSelectSource={handleSelectSource} />
        </section>
      )}

      {tab === 'sources' && (
        <section className="sources-layout">
          <div className="panel">
            {report.sources.length === 0 ? (
              <EmptyState
                title="No sources"
                message="No evidence was retrieved for this claim."
              />
            ) : (
              <ul className="source-list">
                {report.sources.map((source) => (
                  <li key={source.id}>
                    <button
                      type="button"
                      className={`source-list__item ${
                        selectedSourceId === source.id ? 'source-list__item--active' : ''
                      }`}
                      onClick={() => setSelectedSourceId(source.id)}
                    >
                      <span className="source-list__title">
                        {source.title || source.domain}
                      </span>
                      <span className="source-list__meta">
                        {source.domain} · {source.source_type} · authority{' '}
                        {source.authority_score === null
                          ? '—'
                          : formatScore(source.authority_score)}
                        {source.is_duplicate && ' · duplicate'}
                      </span>
                    </button>
                  </li>                ))}
              </ul>
            )}
          </div>
          <SourceDetails
            source={selectedSource}
            onClose={() => setSelectedSourceId(null)}
          />
        </section>
      )}
    </div>
  );
}
