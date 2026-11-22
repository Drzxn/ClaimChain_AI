/**
 * Presentation metadata for verdicts and relationships.
 *
 * Kept in one place so a verdict looks the same everywhere. Every entry carries
 * a short glyph and a text label as well as a colour, because colour alone is
 * not accessible and this is a system where the distinction matters.
 */

import type {
  OverallVerdict,
  Relationship,
  SubclaimVerdict,
} from '../api/types';

export interface VerdictStyle {
  label: string;
  /** Short glyph shown alongside the label. Never the only signal. */
  glyph: string;
  /** Tailwind-free class names defined in index.css. */
  className: string;
  /** One-line plain-English gloss for the verdict. */
  gloss: string;
}

const OVERALL: Record<OverallVerdict, VerdictStyle> = {
  SUPPORTED: {
    label: 'Supported',
    glyph: '✓',
    className: 'verdict--supported',
    gloss: 'The evidence supports the claim as stated.',
  },
  CONTRADICTED: {
    label: 'Contradicted',
    glyph: '✕',
    className: 'verdict--contradicted',
    gloss: 'The evidence contradicts the claim as stated.',
  },
  MIXED: {
    label: 'Mixed',
    glyph: '◐',
    className: 'verdict--mixed',
    gloss: 'The evidence points both ways and neither side prevails.',
  },
  INSUFFICIENT_EVIDENCE: {
    label: 'Insufficient evidence',
    glyph: '?',
    className: 'verdict--insufficient',
    gloss: 'The available evidence did not settle the claim.',
  },
  MISLEADING: {
    label: 'Misleading',
    glyph: '!',
    className: 'verdict--misleading',
    gloss:
      'Part of the claim holds, but a material part is contradicted.',
  },
};

const SUBCLAIM: Record<SubclaimVerdict, VerdictStyle> = {
  SUPPORTED: {
    label: 'Supported',
    glyph: '✓',
    className: 'verdict--supported',
    gloss: 'Contributing evidence supports this subclaim.',
  },
  CONTRADICTED: {
    label: 'Contradicted',
    glyph: '✕',
    className: 'verdict--contradicted',
    gloss: 'Contributing evidence contradicts this subclaim.',
  },
  MIXED: {
    label: 'Mixed',
    glyph: '◐',
    className: 'verdict--mixed',
    gloss: 'Contributing evidence points both ways.',
  },
  INSUFFICIENT: {
    label: 'Insufficient',
    glyph: '?',
    className: 'verdict--insufficient',
    gloss: 'Not enough contributing evidence to judge this subclaim.',
  },
};

const RELATIONSHIP: Record<Relationship, VerdictStyle> = {
  SUPPORTS: {
    label: 'Supports',
    glyph: '✓',
    className: 'verdict--supported',
    gloss: 'This passage supports the subclaim.',
  },
  CONTRADICTS: {
    label: 'Contradicts',
    glyph: '✕',
    className: 'verdict--contradicted',
    gloss: 'This passage contradicts the subclaim.',
  },
  MIXED: {
    label: 'Mixed',
    glyph: '◐',
    className: 'verdict--mixed',
    gloss: 'This passage both supports and contradicts the subclaim.',
  },
  INSUFFICIENT: {
    label: 'Insufficient',
    glyph: '?',
    className: 'verdict--insufficient',
    gloss: 'This passage does not settle the subclaim.',
  },
};

export function overallVerdictStyle(verdict: OverallVerdict): VerdictStyle {
  return OVERALL[verdict];
}

export function subclaimVerdictStyle(verdict: SubclaimVerdict): VerdictStyle {
  return SUBCLAIM[verdict];
}

export function relationshipStyle(relationship: Relationship): VerdictStyle {
  return RELATIONSHIP[relationship];
}

/** Format a 0..1 score for display. Never recalculates, only formats. */
export function formatScore(value: number): string {
  return value.toFixed(4);
}

/** Format a 0..1 ratio as a percentage for readability. */
export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** Format an ISO timestamp for display, tolerating a bad value. */
export function formatDate(value: string | null | undefined): string {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/** Format a date-only value (published_at) without a time component. */
export function formatDateOnly(value: string | null | undefined): string {
  if (!value) return '—';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}
