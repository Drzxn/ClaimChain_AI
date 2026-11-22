import type { ReactNode } from 'react';

import type {
  OverallVerdict,
  Relationship,
  SubclaimVerdict,
} from '../api/types';
import {
  overallVerdictStyle,
  relationshipStyle,
  subclaimVerdictStyle,
} from '../lib/verdicts';

/**
 * Every token the badge can render.
 *
 * The three vocabularies overlap but are not identical: a relationship is
 * `CONTRADICTS`, a verdict is `CONTRADICTED`, and only the overall verdict has
 * `MISLEADING` and `INSUFFICIENT_EVIDENCE`. Keeping them distinct here is what
 * lets the compiler catch a relationship being passed where a verdict belongs.
 */
export type BadgeToken = OverallVerdict | SubclaimVerdict | Relationship;

const RELATIONSHIP_TOKENS: ReadonlySet<string> = new Set<Relationship>([
  'SUPPORTS',
  'CONTRADICTS',
  'MIXED',
  'INSUFFICIENT',
]);

const OVERALL_ONLY_TOKENS: ReadonlySet<string> = new Set<OverallVerdict>([
  'SUPPORTED',
  'CONTRADICTED',
  'MIXED',
  'INSUFFICIENT_EVIDENCE',
  'MISLEADING',
]);

function styleFor(token: BadgeToken) {
  if (RELATIONSHIP_TOKENS.has(token)) {
    return relationshipStyle(token as Relationship);
  }
  if (OVERALL_ONLY_TOKENS.has(token)) {
    return overallVerdictStyle(token as OverallVerdict);
  }
  return subclaimVerdictStyle(token as SubclaimVerdict);
}

interface VerdictBadgeProps {
  verdict: BadgeToken;
  size?: 'sm' | 'md' | 'lg';
  /** Show the plain-English gloss next to the label. */
  showGloss?: boolean;
}

/**
 * A verdict or relationship pill.
 *
 * Carries a glyph and a text label as well as a colour, so the meaning is
 * readable without relying on colour perception.
 */
export function VerdictBadge({
  verdict,
  size = 'md',
  showGloss = false,
}: VerdictBadgeProps) {
  const style = styleFor(verdict);

  return (
    <span className={`badge badge--${size} ${style.className}`}>
      <span className="badge__glyph" aria-hidden="true">
        {style.glyph}
      </span>
      <span className="badge__label">{style.label}</span>
      {showGloss && <span className="badge__gloss">{style.gloss}</span>}
    </span>
  );
}

interface MetricProps {
  label: string;
  value: ReactNode;
  hint?: string;
}

/** A labelled numeric readout. Values are shown as returned, never recomputed. */
export function Metric({ label, value, hint }: MetricProps) {
  return (
    <div className="metric">
      <dt className="metric__label">{label}</dt>
      <dd className="metric__value">{value}</dd>
      {hint && <p className="metric__hint">{hint}</p>}
    </div>
  );
}
