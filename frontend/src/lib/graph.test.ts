/**
 * Graph derivation tests.
 *
 * These assert the two properties that matter: the graph is built only from
 * backend data, and the deterministic layout never places two nodes at the
 * same coordinate.
 *
 * The collision test is a regression guard. The original implementation used
 * an independent row counter for sources, so a source could land on the same
 * y as a subclaim or evidence node in a different column.
 */

import { describe, expect, it } from 'vitest';

import type { ClaimReport } from '../api/types';
import { buildGraph, findEvidence } from './graph';
import {
  EV_1,
  SRC_1,
  SUB_1,
  canonicalReport,
  emptyReport,
  multiSourceReport,
} from '../test/fixtures';

/** Node positions grouped by column (x), to detect collisions within a column. */
function collisionsWithinColumn(report: ClaimReport) {
  const { nodes } = buildGraph(report);
  const byColumn = new Map<number, Map<number, string[]>>();

  for (const node of nodes) {
    const rows = byColumn.get(node.position.x) ?? new Map<number, string[]>();
    const atY = rows.get(node.position.y) ?? [];
    atY.push(node.id);
    rows.set(node.position.y, atY);
    byColumn.set(node.position.x, rows);
  }

  const collisions: Array<{ x: number; y: number; ids: string[] }> = [];
  for (const [x, rows] of byColumn) {
    for (const [y, ids] of rows) {
      if (ids.length > 1) collisions.push({ x, y, ids });
    }
  }
  return collisions;
}

describe('buildGraph', () => {
  it('creates one node per claim, subclaim, evidence link and unique source', () => {
    const report = canonicalReport();
    const { nodes } = buildGraph(report);

    const byKind = (kind: string) =>
      nodes.filter((n) => (n.data as { kind: string }).kind === kind);

    expect(byKind('claim')).toHaveLength(1);
    expect(byKind('subclaim')).toHaveLength(3);
    expect(byKind('evidence')).toHaveLength(3);
    expect(byKind('source')).toHaveLength(3);
  });

  it('deduplicates source nodes by source_id', () => {
    const report = multiSourceReport();
    const { nodes } = buildGraph(report);

    const sourceNodes = nodes.filter(
      (n) => (n.data as { kind: string }).kind === 'source',
    );
    const sourceIds = new Set(sourceNodes.map((n) => n.id));

    // SRC_1 is referenced by two evidence links but must appear once.
    expect(sourceIds.size).toBe(sourceNodes.length);
    expect(sourceNodes.some((n) => n.id === `source-${SRC_1}`)).toBe(true);
  });

  it('derives edges only from backend evidence links', () => {
    const report = canonicalReport();
    const { edges } = buildGraph(report);

    // 3 claim->subclaim, 3 subclaim->evidence, 3 evidence->source
    expect(edges).toHaveLength(9);

    for (const edge of edges) {
      expect(edge.id).toBeTruthy();
      expect(edge.source).toBeTruthy();
      expect(edge.target).toBeTruthy();
    }
  });

  it('labels evidence edges with the backend relationship', () => {
    const report = canonicalReport();
    const { edges } = buildGraph(report);

    const contradicting = edges.filter(
      (e) => e.className === 'edge--contradicts',
    );
    expect(contradicting).toHaveLength(1);
  });

  it('produces no edges when there is no evidence', () => {
    const { edges } = buildGraph(emptyReport());
    // Only claim -> subclaim edges remain.
    expect(edges).toHaveLength(3);
    expect(edges.every((e) => e.label === 'decomposes into')).toBe(true);
  });

  it('is deterministic across repeated builds', () => {
    const report = multiSourceReport();
    const first = buildGraph(report);
    const second = buildGraph(report);

    expect(first.nodes.map((n) => [n.id, n.position.x, n.position.y])).toEqual(
      second.nodes.map((n) => [n.id, n.position.x, n.position.y]),
    );
  });

  // --- The regression guard ------------------------------------------------

  it('places no two nodes at the same position within a column', () => {
    expect(collisionsWithinColumn(multiSourceReport())).toEqual([]);
  });

  it('places no two nodes at the same position for the canonical report', () => {
    expect(collisionsWithinColumn(canonicalReport())).toEqual([]);
  });

  it('places no two nodes at the same position for a report with no evidence', () => {
    expect(collisionsWithinColumn(emptyReport())).toEqual([]);
  });

  it('does not stack two evidence nodes from the same subclaim', () => {
    // Subclaim 1 in the fixture has two evidence links. Under the old layout
    // they could share a row; they must occupy distinct rows.
    const { nodes } = buildGraph(multiSourceReport());
    const evidenceYs = nodes
      .filter((n) => (n.data as { kind: string }).kind === 'evidence')
      .map((n) => n.position.y);

    expect(new Set(evidenceYs).size).toBe(evidenceYs.length);
  });

  it('keeps columns distinct', () => {
    const { nodes } = buildGraph(multiSourceReport());
    const xByKind = new Map<string, Set<number>>();

    for (const node of nodes) {
      const kind = (node.data as { kind: string }).kind;
      const set = xByKind.get(kind) ?? new Set<number>();
      set.add(node.position.x);
      xByKind.set(kind, set);
    }

    // Each kind occupies exactly one column.
    for (const [, xs] of xByKind) {
      expect(xs.size).toBe(1);
    }

    // And the columns are ordered claim < subclaim < evidence < source.
    const x = (kind: string) => [...(xByKind.get(kind) ?? [])][0];
    expect(x('claim')).toBeLessThan(x('subclaim'));
    expect(x('subclaim')).toBeLessThan(x('evidence'));
    expect(x('evidence')).toBeLessThan(x('source'));
  });

  it('positions each source beside the evidence that references it', () => {
    const report = canonicalReport();
    const { nodes } = buildGraph(report);

    for (const subclaim of report.subclaims) {
      for (const evidence of subclaim.evidence) {
        const evidenceNode = nodes.find(
          (n) => n.id === `evidence-${evidence.evidence_link_id}`,
        );
        const sourceNode = nodes.find((n) => n.id === `source-${evidence.source_id}`);
        expect(evidenceNode).toBeDefined();
        expect(sourceNode).toBeDefined();
        expect(sourceNode!.position.y).toBe(evidenceNode!.position.y);
      }
    }
  });
});

describe('findEvidence', () => {
  it('finds evidence by link id', () => {
    const report = canonicalReport();
    const found = findEvidence(report, EV_1);
    expect(found).toBeDefined();
    expect(found!.subclaim_id).toBe(SUB_1);
  });

  it('returns undefined for an unknown id', () => {
    expect(findEvidence(canonicalReport(), 'not-a-real-id')).toBeUndefined();
  });
});
