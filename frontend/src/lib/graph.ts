/**
 * Builds the evidence graph from a Stage 6 report.
 *
 * Every node and edge comes from the report. Nothing is invented, and no
 * relationship is inferred: an edge exists because the backend returned an
 * evidence link with that relationship.
 *
 * Shape:
 *
 *     claim
 *       └─ subclaim (one per subclaim)
 *            └─ evidence (one per evidence link)
 *                 └─ source (deduplicated by source_id)
 *
 * Layout is deterministic: fixed column x-positions, rows spaced by index. No
 * layout library, no force simulation, so the same report always renders the
 * same picture and a screenshot is reproducible.
 */

import type { Edge, Node } from '@xyflow/react';

import type { ClaimReport, ReportEvidence } from '../api/types';
import { relationshipStyle } from './verdicts';

export type GraphNodeKind = 'claim' | 'subclaim' | 'evidence' | 'source';

export interface ClaimNodeData extends Record<string, unknown> {
  kind: 'claim';
  label: string;
  verdict: string;
  score: number;
}

export interface SubclaimNodeData extends Record<string, unknown> {
  kind: 'subclaim';
  ordinal: number;
  label: string;
  verdict: string;
  score: number;
  evidenceCount: number;
}

export interface EvidenceNodeData extends Record<string, unknown> {
  kind: 'evidence';
  label: string;
  relationship: string;
  quote: string;
  sourceDomain: string;
  evidenceLinkId: string;
}

export interface SourceNodeData extends Record<string, unknown> {
  kind: 'source';
  label: string;
  domain: string;
  sourceType: string;
  authority: number | null;
  recency: number | null;
}

export type ClaimChainNode =
  | Node<ClaimNodeData, 'claim'>
  | Node<SubclaimNodeData, 'subclaim'>
  | Node<EvidenceNodeData, 'evidence'>
  | Node<SourceNodeData, 'source'>;

/** Column x positions. Fixed, so layout is deterministic. */
const COLUMN_X = {
  claim: 0,
  subclaim: 320,
  evidence: 680,
  source: 1040,
} as const;

const ROW_HEIGHT = 96;

/** Truncate a string for a node label without cutting mid-word. */
function truncate(text: string, max: number): string {
  const clean = text.trim();
  if (clean.length <= max) return clean;
  return `${clean.slice(0, max - 1).trimEnd()}…`;
}

export interface GraphModel {
  nodes: ClaimChainNode[];
  edges: Edge[];
}

export function buildGraph(report: ClaimReport): GraphModel {
  const nodes: ClaimChainNode[] = [];
  const edges: Edge[] = [];

  const claimId = `claim-${report.claim.id}`;

  nodes.push({
    id: claimId,
    type: 'claim',
    position: { x: COLUMN_X.claim, y: 0 },
    data: {
      kind: 'claim',
      label: truncate(report.claim.raw_text, 90),
      verdict: report.verdict.verdict,
      score: report.verdict.score,
    },
  });

  // Sources are shared across subclaims, so they are placed once and reused.
  const sourceNodeIds = new Map<string, string>();
  const sourceRowIndex = new Map<string, number>();
  let nextSourceRow = 0;

  let rowCursor = 0;

  report.subclaims.forEach((subclaim, subIndex) => {
    const subclaimId = `subclaim-${subclaim.id}`;
    const subclaimY = rowCursor * ROW_HEIGHT;
    rowCursor += 1;

    nodes.push({
      id: subclaimId,
      type: 'subclaim',
      position: { x: COLUMN_X.subclaim, y: subclaimY },
      data: {
        kind: 'subclaim',
        ordinal: subclaim.ordinal,
        label: truncate(subclaim.text, 80),
        verdict: subclaim.verdict.verdict,
        score: subclaim.verdict.score,
        evidenceCount: subclaim.evidence.length,
      },
    });

    edges.push({
      id: `edge-${claimId}-${subclaimId}`,
      source: claimId,
      target: subclaimId,
      type: 'smoothstep',
      label: 'decomposes into',
      animated: false,
    });

    subclaim.evidence.forEach((evidence, evidenceIndex) => {
      const evidenceId = `evidence-${evidence.evidence_link_id}`;
      const evidenceY = (rowCursor + evidenceIndex) * ROW_HEIGHT;

      nodes.push({
        id: evidenceId,
        type: 'evidence',
        position: { x: COLUMN_X.evidence, y: evidenceY },
        data: {
          kind: 'evidence',
          label: truncate(evidence.quote || evidence.rationale, 70),
          relationship: evidence.relationship,
          quote: evidence.quote,
          sourceDomain: evidence.source_domain,
          evidenceLinkId: evidence.evidence_link_id,
        },
      });

      const style = relationshipStyle(evidence.relationship);

      edges.push({
        id: `edge-${subclaimId}-${evidenceId}`,
        source: subclaimId,
        target: evidenceId,
        type: 'smoothstep',
        label: style.label,
        className: `edge--${evidence.relationship.toLowerCase()}`,
        animated: false,
      });

      // --- source node, shared per source_id ---------------------------
      let sourceNodeId = sourceNodeIds.get(evidence.source_id);
      if (!sourceNodeId) {
        sourceNodeId = `source-${evidence.source_id}`;
        sourceNodeIds.set(evidence.source_id, sourceNodeId);

        const source = report.sources.find((s) => s.id === evidence.source_id);
        const row = nextSourceRow;
        nextSourceRow += 1;
        sourceRowIndex.set(evidence.source_id, row);

        nodes.push({
          id: sourceNodeId,
          type: 'source',
          position: { x: COLUMN_X.source, y: row * ROW_HEIGHT },
          data: {
            kind: 'source',
            label: truncate(source?.title || evidence.source_domain, 60),
            domain: source?.domain ?? evidence.source_domain,
            sourceType: source?.source_type ?? 'unknown',
            authority: source?.authority_score ?? null,
            recency: source?.recency_score ?? null,
          },
        });
      }

      edges.push({
        id: `edge-${evidenceId}-${sourceNodeId}`,
        source: evidenceId,
        target: sourceNodeId,
        type: 'smoothstep',
        label: 'from',
        animated: false,
      });
    });

    // Advance past this subclaim's evidence rows so the next subclaim does not
    // overlap them.
    rowCursor += Math.max(subclaim.evidence.length, 1);
    void subIndex;
  });

  return { nodes, edges };
}

/** Look up the evidence behind an evidence node id. */
export function findEvidence(
  report: ClaimReport,
  evidenceLinkId: string,
): ReportEvidence | undefined {
  for (const subclaim of report.subclaims) {
    const match = subclaim.evidence.find(
      (e) => e.evidence_link_id === evidenceLinkId,
    );
    if (match) return match;
  }
  return undefined;
}
