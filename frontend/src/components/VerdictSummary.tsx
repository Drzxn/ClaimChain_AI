import type { ClaimReport } from '../api/types';
import { formatScore, overallVerdictStyle } from '../lib/verdicts';
import { Metric, VerdictBadge } from './VerdictBadge';

interface VerdictSummaryProps {
  report: ClaimReport;
}

/**
 * The overall verdict block.
 *
 * Every number here is displayed exactly as the backend returned it. The
 * frontend never recomputes a score, a coverage figure or a ratio: Stage 4 is
 * the authority and this is a readout.
 */
export function VerdictSummary({ report }: VerdictSummaryProps) {
  const { verdict } = report;
  const style = overallVerdictStyle(verdict.verdict);

  return (
    <section className={`panel verdict-summary ${style.className}`}>
      <div className="verdict-summary__headline">
        <VerdictBadge verdict={verdict.verdict} size="lg" />
        <p className="verdict-summary__gloss">{style.gloss}</p>
      </div>

      <p className="verdict-summary__claim">{report.claim.raw_text}</p>

      <dl className="metric-grid">
        <Metric
          label="Score"
          value={formatScore(verdict.score)}
          hint="Weighted aggregate, −1 to +1"
        />
        <Metric
          label="Coverage"
          value={formatScore(verdict.coverage)}
          hint="Share of subclaim weight with evidence"
        />
        <Metric
          label="Contradicted weight"
          value={formatScore(verdict.contradicted_weight_ratio)}
          hint="Share of evidenced weight that is contradicted"
        />
        <Metric
          label="Subclaims"
          value={report.subclaims.length}
          hint={`${report.evidence_link_count} evidence link(s)`}
        />
      </dl>

      {verdict.summary && (
        <p className="verdict-summary__summary">{verdict.summary}</p>
      )}

      <p className="verdict-summary__provenance">
        Deterministic aggregation <code>{verdict.aggregation_version}</code>,
        computed {new Date(verdict.computed_at).toLocaleString()}.
      </p>
    </section>
  );
}
