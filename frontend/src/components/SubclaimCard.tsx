import type { ReportSubclaim } from '../api/types';
import { formatScore, subclaimVerdictStyle } from '../lib/verdicts';
import { EvidenceCard } from './EvidenceCard';
import { VerdictBadge } from './VerdictBadge';

interface SubclaimCardProps {
  subclaim: ReportSubclaim;
  onSelectSource?: (sourceId: string) => void;
}

/**
 * One subclaim with its persisted verdict and the evidence judged against it.
 *
 * The weights shown are the ones Stage 4 summed. They are displayed, not
 * derived, so a reader can check the arithmetic against the verdict.
 */
export function SubclaimCard({ subclaim, onSelectSource }: SubclaimCardProps) {
  const style = subclaimVerdictStyle(subclaim.verdict.verdict);
  const { verdict } = subclaim;

  return (
    <article className={`subclaim-card ${style.className}`}>
      <header className="subclaim-card__header">
        <span className="subclaim-card__ordinal">{subclaim.ordinal}</span>
        <div className="subclaim-card__heading">
          <p className="subclaim-card__text">{subclaim.text}</p>
          <div className="subclaim-card__tags">
            <span className="tag">{subclaim.claim_type}</span>
            <span className="tag tag--muted">weight {formatScore(subclaim.weight)}</span>
            <span className="tag tag--muted">
              {verdict.link_count} contributing link
              {verdict.link_count === 1 ? '' : 's'}
            </span>
          </div>
        </div>
        <VerdictBadge verdict={verdict.verdict} size="md" />
      </header>

      <dl className="inline-metrics inline-metrics--wide">
        <div>
          <dt>Score</dt>
          <dd>{formatScore(verdict.score)}</dd>
        </div>
        <div>
          <dt>Coverage</dt>
          <dd>{formatScore(verdict.coverage)}</dd>
        </div>
        <div>
          <dt>Support</dt>
          <dd>{formatScore(verdict.support_weight)}</dd>
        </div>
        <div>
          <dt>Contradict</dt>
          <dd>{formatScore(verdict.contradict_weight)}</dd>
        </div>
        <div>
          <dt>Mixed</dt>
          <dd>{formatScore(verdict.mixed_weight)}</dd>
        </div>
      </dl>

      {verdict.rationale && (
        <p className="subclaim-card__rationale">{verdict.rationale}</p>
      )}

      {subclaim.decomposition_note && (
        <p className="subclaim-card__note">
          <span className="text--muted">Decomposition note: </span>
          {subclaim.decomposition_note}
        </p>
      )}

      <div className="subclaim-card__evidence">
        <h4 className="section-heading">
          Evidence ({subclaim.evidence.length})
        </h4>
        {subclaim.evidence.length === 0 ? (
          <p className="text--muted">
            No evidence was judged against this subclaim, so it contributed
            nothing to the aggregate score.
          </p>
        ) : (
          <div className="evidence-list">
            {subclaim.evidence.map((evidence) => (
              <EvidenceCard
                key={evidence.evidence_link_id}
                evidence={evidence}
                onSelectSource={onSelectSource}
              />
            ))}
          </div>
        )}
      </div>
    </article>
  );
}
