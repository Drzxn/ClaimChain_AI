/**
 * TypeScript mirror of the backend's Stage 6 report contract.
 *
 * Source of truth: `backend/app/schemas/report.py`. These types are transcribed
 * field-for-field, not invented. If the backend contract changes, this file must
 * change with it and the compiler will surface every call site that breaks.
 *
 * Nothing here is computed. Every value is what the backend returned.
 */

// --- Unions, matching the backend's Literal types -------------------------

export type OverallVerdict =
  | 'SUPPORTED'
  | 'CONTRADICTED'
  | 'MIXED'
  | 'INSUFFICIENT_EVIDENCE'
  | 'MISLEADING';

export type SubclaimVerdict =
  | 'SUPPORTED'
  | 'CONTRADICTED'
  | 'MIXED'
  | 'INSUFFICIENT';

export type Relationship =
  | 'SUPPORTS'
  | 'CONTRADICTS'
  | 'MIXED'
  | 'INSUFFICIENT';

export type QuoteValidation = 'passed' | 'failed';

export type NarrativeSource = 'deterministic' | 'model';

export type LimitationSource = 'verdict' | 'evidence' | 'sources' | 'report';

// --- A. Claim -------------------------------------------------------------

export interface ReportClaim {
  id: string;
  raw_text: string;
  domain: string;
  status: string;
  created_at: string;
  completed_at: string | null;
}

// --- B. Verdict -----------------------------------------------------------

export interface ReportVerdict {
  verdict: OverallVerdict;
  score: number;
  coverage: number;
  contradicted_weight_ratio: number;
  subclaim_counts: Record<string, number>;
  summary: string | null;
  aggregation_version: string;
  computed_at: string;
}

// --- D. Evidence ----------------------------------------------------------

export interface ReportEvidence {
  evidence_link_id: string;
  subclaim_id: string;
  passage_id: string;
  source_id: string;

  relationship: Relationship;
  relevance: number;
  evidence_quality: number;
  stance_confidence: number;

  quote: string;
  rationale: string;
  quote_validation: QuoteValidation;

  passage_text: string;
  passage_chunk_index: number;
  quote_start: number;
  quote_end: number;

  source_url: string;
  source_domain: string;
  source_title: string | null;
}

// --- C. Subclaims ---------------------------------------------------------

export interface ReportSubclaimVerdict {
  verdict: SubclaimVerdict;
  score: number;
  coverage: number;
  support_weight: number;
  contradict_weight: number;
  mixed_weight: number;
  link_count: number;
  rationale: string;
  aggregation_version: string;
  computed_at: string;
}

export interface ReportSubclaim {
  id: string;
  ordinal: number;
  text: string;
  claim_type: string;
  weight: number;
  decomposition_note: string | null;
  verdict: ReportSubclaimVerdict;
  evidence: ReportEvidence[];
}

// --- E. Sources -----------------------------------------------------------

export interface ReportSource {
  id: string;
  url: string;
  domain: string;
  title: string | null;
  publisher: string | null;
  source_type: string;
  published_at: string | null;
  retrieved_at: string;

  authority_score: number | null;
  recency_score: number | null;
  assessment_rationale: string | null;
  assessed_at: string | null;
  cites_primary_sources: boolean | null;

  duplicate_of: string | null;
  is_duplicate: boolean;
  evidence_count: number;
}

// --- F. Explanation -------------------------------------------------------

export interface ReportExplanation {
  verdict_statement: string;
  subclaim_statements: string[];
  supporting_points: string[];
  contradicting_points: string[];
  narrative: string;
  narrative_source: NarrativeSource;
}

// --- G. Limitations -------------------------------------------------------

export interface ReportLimitation {
  code: string;
  message: string;
  source: LimitationSource;
}

// --- The report -----------------------------------------------------------

export interface ClaimReport {
  claim: ReportClaim;
  verdict: ReportVerdict;
  subclaims: ReportSubclaim[];
  sources: ReportSource[];
  explanation: ReportExplanation;
  limitations: ReportLimitation[];
  report_version: string;
  generated_at: string;
  evidence_link_count: number;
}

// --- Claim endpoints ------------------------------------------------------

export type ClaimStatus = 'queued' | 'running' | 'complete' | 'failed';

export type ClaimDomain = 'technology' | 'science' | 'general';

export interface CreateClaimResponse {
  id: string;
  status: ClaimStatus;
  domain: ClaimDomain;
  subclaim_count: number;
}

export interface SubclaimResponse {
  id: string;
  ordinal: number;
  text: string;
  claim_type: string;
  weight: number;
}

export interface ClaimResponse {
  id: string;
  raw_text: string;
  domain: ClaimDomain;
  status: ClaimStatus;
  error: string | null;
  created_at: string;
  completed_at: string | null;
  subclaims: SubclaimResponse[];
}

export interface HealthResponse {
  status: string;
  version: string;
  environment: string;
  database: string;
  llm_configured: boolean;
  search_configured: boolean;
}
