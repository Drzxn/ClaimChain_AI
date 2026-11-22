import { useCallback, useMemo, useState } from 'react';
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  type Node,
  type NodeMouseHandler,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import type { ClaimReport } from '../api/types';
import {
  buildGraph,
  findEvidence,
  type ClaimNodeData,
  type EvidenceNodeData,
  type SourceNodeData,
  type SubclaimNodeData,
} from '../lib/graph';
import { overallVerdictStyle, subclaimVerdictStyle } from '../lib/verdicts';
import { EmptyState } from './States';
import { VerdictBadge } from './VerdictBadge';

type SelectedNode =
  | { kind: 'claim' }
  | { kind: 'subclaim'; subclaimId: string }
  | { kind: 'evidence'; evidenceLinkId: string }
  | { kind: 'source'; sourceId: string }
  | null;

// --- Custom node renderers ------------------------------------------------

function ClaimNode({ data }: { data: ClaimNodeData }) {
  const style = overallVerdictStyle(data.verdict as never);
  return (
    <div className={`graph-node graph-node--claim ${style.className}`}>
      <span className="graph-node__kind">Claim</span>
      <p className="graph-node__label">{data.label}</p>
      <span className="graph-node__meta">{style.label}</span>
    </div>
  );
}

function SubclaimNode({ data }: { data: SubclaimNodeData }) {
  const style = subclaimVerdictStyle(data.verdict as never);
  return (
    <div className={`graph-node graph-node--subclaim ${style.className}`}>
      <span className="graph-node__kind">Subclaim {data.ordinal}</span>
      <p className="graph-node__label">{data.label}</p>
      <span className="graph-node__meta">
        {style.label} · {data.evidenceCount} evidence
      </span>
    </div>
  );
}

function EvidenceNode({ data }: { data: EvidenceNodeData }) {
  return (
    <div className="graph-node graph-node--evidence">
      <span className="graph-node__kind">Evidence</span>
      <p className="graph-node__label">{data.label}</p>
      <span className="graph-node__meta">{data.sourceDomain}</span>
    </div>
  );
}

function SourceNode({ data }: { data: SourceNodeData }) {
  return (
    <div className="graph-node graph-node--source">
      <span className="graph-node__kind">Source</span>
      <p className="graph-node__label">{data.label}</p>
      <span className="graph-node__meta">{data.domain}</span>
    </div>
  );
}

const nodeTypes = {
  claim: ClaimNode,
  subclaim: SubclaimNode,
  evidence: EvidenceNode,
  source: SourceNode,
};

// --- Panel ----------------------------------------------------------------

interface NodeDetailsProps {
  report: ClaimReport;
  selected: SelectedNode;
  onSelectSource: (sourceId: string) => void;
  onClose: () => void;
}

function NodeDetails({
  report,
  selected,
  onSelectSource,
  onClose,
}: NodeDetailsProps) {
  if (!selected) {
    return (
      <div className="graph-panel">
        <p className="text--muted">
          Select a node to inspect it. The graph shows how the claim decomposes
          into subclaims, and which evidence was judged against each.
        </p>
      </div>
    );
  }

  if (selected.kind === 'claim') {
    return (
      <div className="graph-panel">
        <header className="graph-panel__header">
          <h4 className="section-heading">Claim</h4>
          <button type="button" className="button button--ghost button--sm" onClick={onClose}>
            Close
          </button>
        </header>
        <p className="graph-panel__text">{report.claim.raw_text}</p>
        <VerdictBadge verdict={report.verdict.verdict} size="md" showGloss />
      </div>
    );
  }

  if (selected.kind === 'subclaim') {
    const subclaim = report.subclaims.find((s) => s.id === selected.subclaimId);
    if (!subclaim) return null;
    return (
      <div className="graph-panel">
        <header className="graph-panel__header">
          <h4 className="section-heading">Subclaim {subclaim.ordinal}</h4>
          <button type="button" className="button button--ghost button--sm" onClick={onClose}>
            Close
          </button>
        </header>
        <p className="graph-panel__text">{subclaim.text}</p>
        <VerdictBadge verdict={subclaim.verdict.verdict} size="md" showGloss />
        <p className="graph-panel__meta">
          {subclaim.evidence.length} evidence link
          {subclaim.evidence.length === 1 ? '' : 's'} · score{' '}
          {subclaim.verdict.score.toFixed(4)}
        </p>
      </div>
    );
  }

  if (selected.kind === 'evidence') {
    const evidence = findEvidence(report, selected.evidenceLinkId);
    if (!evidence) return null;
    return (
      <div className="graph-panel">
        <header className="graph-panel__header">
          <h4 className="section-heading">Evidence</h4>
          <button type="button" className="button button--ghost button--sm" onClick={onClose}>
            Close
          </button>
        </header>
        <VerdictBadge verdict={evidence.relationship} size="md" showGloss />
        {evidence.quote_validation === 'passed' && evidence.quote ? (
          <blockquote className="graph-panel__quote">{evidence.quote}</blockquote>
        ) : (
          <p className="graph-panel__meta">
            Quote could not be verified against the stored passage.
          </p>
        )}
        {evidence.rationale && (
          <p className="graph-panel__text">{evidence.rationale}</p>
        )}
        <button
          type="button"
          className="button button--secondary button--sm"
          onClick={() => onSelectSource(evidence.source_id)}
        >
          Open source
        </button>
      </div>
    );
  }

  const source = report.sources.find((s) => s.id === selected.sourceId);
  if (!source) return null;
  return (
    <div className="graph-panel">
      <header className="graph-panel__header">
        <h4 className="section-heading">Source</h4>
        <button type="button" className="button button--ghost button--sm" onClick={onClose}>
          Close
        </button>
      </header>
      <p className="graph-panel__text">{source.title || source.domain}</p>
      <a
        className="link"
        href={source.url}
        target="_blank"
        rel="noopener noreferrer"
      >
        {source.url}
      </a>
      <p className="graph-panel__meta">
        {source.source_type} · authority{' '}
        {source.authority_score === null ? 'not assessed' : source.authority_score.toFixed(3)}
      </p>
    </div>
  );
}

// --- The graph ------------------------------------------------------------

interface EvidenceGraphProps {
  report: ClaimReport;
  onSelectSource: (sourceId: string) => void;
}

/**
 * Interactive evidence graph.
 *
 * Nodes and edges are built entirely from the Stage 6 report, so every edge
 * corresponds to a real persisted relationship. Pan, zoom and node selection
 * come from React Flow; layout is computed deterministically in `lib/graph.ts`.
 */
export function EvidenceGraph({ report, onSelectSource }: EvidenceGraphProps) {
  const [selected, setSelected] = useState<SelectedNode>(null);

  const { nodes, edges } = useMemo(() => buildGraph(report), [report]);

  const handleNodeClick = useCallback<NodeMouseHandler>((_event, node: Node) => {
    const data = node.data as { kind?: string };

    if (data.kind === 'claim') {
      setSelected({ kind: 'claim' });
    } else if (data.kind === 'subclaim') {
      setSelected({ kind: 'subclaim', subclaimId: node.id.replace('subclaim-', '') });
    } else if (data.kind === 'evidence') {
      setSelected({
        kind: 'evidence',
        evidenceLinkId: node.id.replace('evidence-', ''),
      });
    } else if (data.kind === 'source') {
      setSelected({ kind: 'source', sourceId: node.id.replace('source-', '') });
    }
  }, []);

  if (nodes.length === 0) {
    return (
      <EmptyState
        title="No graph to show"
        message="This claim has no subclaims or evidence to visualise."
      />
    );
  }

  return (
    <div className="graph-layout">
      <div className="graph-canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onNodeClick={handleNodeClick}
          onPaneClick={() => setSelected(null)}
          fitView
          minZoom={0.2}
          maxZoom={1.5}
          nodesDraggable={false}
          nodesConnectable={false}
          proOptions={{ hideAttribution: false }}
        >
          <Background gap={16} />
          <Controls showInteractive={false} />
          <MiniMap pannable zoomable />
        </ReactFlow>
      </div>

      <NodeDetails
        report={report}
        selected={selected}
        onSelectSource={onSelectSource}
        onClose={() => setSelected(null)}
      />
    </div>
  );
}
