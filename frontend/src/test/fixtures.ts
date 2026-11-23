/**
 * Test fixtures.
 *
 * These mirror the backend's Stage 6 `ClaimReport` contract exactly, field for
 * field. They are test data, not production data: nothing in `src/` imports
 * this file, so no fake claim or verdict can reach the running application.
 *
 * The canonical report reproduces the project's reference case:
 * two supported subclaims, one contradicted, overall MISLEADING with
 * score 0.3333, coverage 1.0000, contradicted ratio 0.3333.
 */

import type { ClaimReport, ReportEvidence } from '../api/types';

export const CLAIM_ID = '11111111-1111-4111-8111-111111111111';
export const SUB_1 = '22222222-2222-4222-8222-222222222222';
export const SUB_2 = '33333333-3333-4333-8333-333333333333';
export const SUB_3 = '44444444-4444-4444-8444-444444444444';

export const SRC_1 = '55555555-5555-4555-8555-555555555555';
export const SRC_2 = '66666666-6666-4666-8666-666666666666';
export const SRC_3 = '77777777-7777-4777-8777-777777777777';

export const EV_1 = '88888888-8888-4888-8888-888888888888';
export const EV_2 = '99999999-9999-4999-8999-999999999999';
export const EV_3 = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';

function evidence(overrides: Partial<ReportEvidence>): ReportEvidence {
  return {
    evidence_link_id: EV_1,
    subclaim_id: SUB_1,
    passage_id: 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb',
    source_id: SRC_1,
    relationship: 'SUPPORTS',
    relevance: 0.95,
    evidence_quality: 0.9,
    stance_confidence: 0.92,
    quote: 'HTTPS encrypts traffic between the browser and the server.',
    rationale: 'Directly states the encryption property.',
    quote_validation: 'passed',
    passage_text:
      'HTTPS encrypts traffic between the browser and the server using TLS.',
    passage_chunk_index: 0,
    quote_start: 0,
    quote_end: 55,
    source_url: 'https://example.org/https-explained',
    source_domain: 'example.org',
    source_title: 'How HTTPS Works',
    ...overrides,
  };
}

export function canonicalReport(): ClaimReport {
  return {
    claim: {
      id: CLAIM_ID,
      raw_text: 'Using HTTPS makes a website completely secure.',
      domain: 'technology',
      status: 'queued',
      created_at: '2026-11-18T12:00:00Z',
      completed_at: null,
    },
    verdict: {
      verdict: 'MISLEADING',
      score: 0.3333,
      coverage: 1.0,
      contradicted_weight_ratio: 0.3333,
      subclaim_counts: { SUPPORTED: 2, CONTRADICTED: 1 },
      summary:
        'Overall score 0.3333 is positive but 0.3333 of evidenced subclaim weight is contradicted.',
      aggregation_version: 'agg-1.0.0',
      computed_at: '2026-11-18T12:05:00Z',
    },
    subclaims: [
      {
        id: SUB_1,
        ordinal: 1,
        text: 'HTTPS encrypts data in transit between browser and server.',
        claim_type: 'factual',
        weight: 1.0,
        decomposition_note: null,
        verdict: {
          verdict: 'SUPPORTED',
          score: 1.0,
          coverage: 1.0,
          support_weight: 1.0,
          contradict_weight: 0.0,
          mixed_weight: 0.0,
          link_count: 1,
          rationale: 'All 1 contributing link(s) support the subclaim.',
          aggregation_version: 'agg-1.0.0',
          computed_at: '2026-11-18T12:05:00Z',
        },
        evidence: [evidence({})],
      },
      {
        id: SUB_2,
        ordinal: 2,
        text: 'HTTPS protects against some classes of network attack.',
        claim_type: 'causal',
        weight: 1.0,
        decomposition_note: null,
        verdict: {
          verdict: 'SUPPORTED',
          score: 1.0,
          coverage: 1.0,
          support_weight: 1.0,
          contradict_weight: 0.0,
          mixed_weight: 0.0,
          link_count: 1,
          rationale: 'All 1 contributing link(s) support the subclaim.',
          aggregation_version: 'agg-1.0.0',
          computed_at: '2026-11-18T12:05:00Z',
        },
        evidence: [
          evidence({
            evidence_link_id: EV_2,
            subclaim_id: SUB_2,
            source_id: SRC_2,
            quote: 'HTTPS protects against passive eavesdropping.',
            rationale: 'States the protective benefit.',
            passage_text:
              'HTTPS protects against passive eavesdropping on untrusted networks.',
            source_url: 'https://example.com/transport-encryption',
            source_domain: 'example.com',
            source_title: 'Transport Encryption and Web Security',
          }),
        ],
      },
      {
        id: SUB_3,
        ordinal: 3,
        text: 'HTTPS makes a website completely secure against all attacks.',
        claim_type: 'factual',
        weight: 1.0,
        decomposition_note: null,
        verdict: {
          verdict: 'CONTRADICTED',
          score: -1.0,
          coverage: 1.0,
          support_weight: 0.0,
          contradict_weight: 1.0,
          mixed_weight: 0.0,
          link_count: 1,
          rationale: 'All 1 contributing link(s) contradict the subclaim.',
          aggregation_version: 'agg-1.0.0',
          computed_at: '2026-11-18T12:05:00Z',
        },
        evidence: [
          evidence({
            evidence_link_id: EV_3,
            subclaim_id: SUB_3,
            source_id: SRC_3,
            relationship: 'CONTRADICTS',
            quote: 'HTTPS does not prevent cross-site scripting.',
            rationale: 'Explicitly denies the absolute claim.',
            passage_text:
              'HTTPS does not prevent cross-site scripting or SQL injection.',
            source_url: 'https://example.org/standards/tls-overview',
            source_domain: 'example.org',
            source_title: 'TLS Overview',
          }),
        ],
      },
    ],
    sources: [
      {
        id: SRC_1,
        url: 'https://example.org/https-explained',
        domain: 'example.org',
        title: 'How HTTPS Works',
        publisher: 'Example Organisation',
        source_type: 'reference',
        published_at: '2023-04-12',
        retrieved_at: '2026-11-18T12:00:00Z',
        authority_score: 0.8,
        recency_score: 0.6,
        assessment_rationale: 'Authority established reference (domain example.org).',
        assessed_at: '2026-11-18T12:04:00Z',
        cites_primary_sources: true,
        duplicate_of: null,
        is_duplicate: false,
        evidence_count: 1,
      },
      {
        id: SRC_2,
        url: 'https://example.com/transport-encryption',
        domain: 'example.com',
        title: 'Transport Encryption and Web Security',
        publisher: 'Example Security Institute',
        source_type: 'secondary',
        published_at: '2022-11-03',
        retrieved_at: '2026-11-18T12:00:00Z',
        authority_score: 0.5,
        recency_score: 0.4,
        assessment_rationale: 'Authority general web (no tier match).',
        assessed_at: '2026-11-18T12:04:00Z',
        cites_primary_sources: null,
        duplicate_of: null,
        is_duplicate: false,
        evidence_count: 1,
      },
      {
        id: SRC_3,
        url: 'https://example.org/standards/tls-overview',
        domain: 'example.org',
        title: 'TLS Overview',
        publisher: 'Example Standards Body',
        source_type: 'primary',
        published_at: null,
        retrieved_at: '2026-11-18T12:00:00Z',
        authority_score: 0.8,
        recency_score: 0.5,
        assessment_rationale: 'Authority established reference; no publication date.',
        assessed_at: '2026-11-18T12:04:00Z',
        cites_primary_sources: false,
        duplicate_of: null,
        is_duplicate: false,
        evidence_count: 1,
      },
    ],
    explanation: {
      verdict_statement:
        'The claim was assessed as MISLEADING with an aggregate score of 0.3333 and coverage 1.0000.',
      subclaim_statements: [
        'Subclaim 1 was supported (score 1.0000, 1 contributing link(s)).',
        'Subclaim 2 was supported (score 1.0000, 1 contributing link(s)).',
        'Subclaim 3 was contradicted (score -1.0000, 1 contributing link(s)).',
      ],
      supporting_points: [
        'HTTPS encrypts traffic between the browser and the server.',
        'HTTPS protects against passive eavesdropping.',
      ],
      contradicting_points: ['HTTPS does not prevent cross-site scripting.'],
      narrative: '',
      narrative_source: 'deterministic',
    },
    limitations: [
      {
        code: 'sources_without_publication_date',
        message: '1 source(s) had no publication date.',
        source: 'sources',
      },
    ],
    report_version: 'report.v1',
    generated_at: '2026-11-18T12:06:00Z',
    evidence_link_count: 3,
  };
}

/** A report with no evidence at all: the abstention path. */
export function emptyReport(): ClaimReport {
  const report = canonicalReport();
  return {
    ...report,
    verdict: {
      ...report.verdict,
      verdict: 'INSUFFICIENT_EVIDENCE',
      score: 0.0,
      coverage: 0.0,
      contradicted_weight_ratio: 0.0,
      subclaim_counts: { INSUFFICIENT: 3 },
      summary: 'Only 0.0000 of subclaim weight is sufficiently evidenced.',
    },
    subclaims: report.subclaims.map((s) => ({
      ...s,
      evidence: [],
      verdict: {
        ...s.verdict,
        verdict: 'INSUFFICIENT',
        score: 0.0,
        coverage: 0.0,
        support_weight: 0.0,
        contradict_weight: 0.0,
        mixed_weight: 0.0,
        link_count: 0,
        rationale: 'Total contributing evidence weight 0.0 is below the minimum.',
      },
    })),
    sources: [],
    explanation: {
      ...report.explanation,
      supporting_points: [],
      contradicting_points: [],
    },
    limitations: [
      {
        code: 'insufficient_evidence',
        message: 'The available evidence did not meet the coverage floor.',
        source: 'verdict',
      },
    ],
    evidence_link_count: 0,
  };
}

/** A report with several subclaims and shared sources, for layout testing. */
export function multiSourceReport(): ClaimReport {
  const base = canonicalReport();

  // Reuse SRC_1 across two subclaims so source dedupe is exercised, and give
  // subclaim 1 two evidence links so the row cursor advances unevenly.
  const extra = evidence({
    evidence_link_id: 'cccccccc-cccc-4ccc-8ccc-cccccccccccc',    subclaim_id: SUB_1,
    source_id: SRC_1,
    quote: 'TLS provides confidentiality for the channel.',
    rationale: 'Second supporting passage from the same source.',
  });

  return {
    ...base,
    subclaims: [
      { ...base.subclaims[0], evidence: [...base.subclaims[0].evidence, extra] },
      { ...base.subclaims[1], evidence: [...base.subclaims[1].evidence] },
      { ...base.subclaims[2], evidence: [...base.subclaims[2].evidence] },
    ],
    evidence_link_count: 4,
  };
}
