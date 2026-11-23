/**
 * EvidenceGraph component tests.
 *
 * React Flow needs a measurable container, which jsdom does not provide, so the
 * library is stubbed with a lightweight stand-in that renders each node's data
 * and forwards clicks. That keeps the test focused on what this component
 * actually owns: it derives nodes and edges from the report, and it shows the
 * right details when a node is selected.
 *
 * The graph *derivation* itself is covered directly in lib/graph.test.ts.
 *
 * Node labels and the detail panel deliberately repeat the same text, so
 * assertions about the panel are scoped to `.graph-panel` to avoid matching the
 * node that was clicked.
 */

import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { EvidenceGraph } from './EvidenceGraph';
import { canonicalReport } from '../test/fixtures';

// --- React Flow stand-in --------------------------------------------------

vi.mock('@xyflow/react', () => ({
  ReactFlow: ({
    nodes,
    edges,
    onNodeClick,
    nodeTypes,
  }: {
    nodes: Array<{ id: string; type: string; data: Record<string, unknown> }>;
    edges: Array<{ id: string }>;
    onNodeClick?: (event: unknown, node: unknown) => void;
    nodeTypes: Record<string, React.ComponentType<{ data: Record<string, unknown> }>>;
  }) => (
    <div data-testid="react-flow" data-edge-count={edges.length}>
      {nodes.map((node) => {
        const Renderer = nodeTypes[node.type];
        return (
          <div
            key={node.id}
            data-testid={`node-${node.id}`}
            onClick={() => onNodeClick?.({}, node)}
          >
            {Renderer ? <Renderer data={node.data} /> : node.id}
          </div>
        );
      })}
    </div>
  ),
  Background: () => null,
  Controls: () => null,
  MiniMap: () => null,
}));

/** The detail panel, scoped so node labels do not collide with its text. */
function panel(container: HTMLElement): HTMLElement {
  const el = container.querySelector('.graph-panel');
  if (!el) throw new Error('detail panel not rendered');
  return el as HTMLElement;
}

describe('EvidenceGraph', () => {
  it('renders the graph from the report', () => {
    render(<EvidenceGraph report={canonicalReport()} onSelectSource={vi.fn()} />);
    expect(screen.getByTestId('react-flow')).toBeInTheDocument();
  });

  it('creates a node for the claim, subclaims, evidence and sources', () => {
    const report = canonicalReport();
    render(<EvidenceGraph report={report} onSelectSource={vi.fn()} />);

    expect(screen.getByTestId(`node-claim-${report.claim.id}`)).toBeInTheDocument();
    for (const subclaim of report.subclaims) {
      expect(screen.getByTestId(`node-subclaim-${subclaim.id}`)).toBeInTheDocument();
      for (const evidence of subclaim.evidence) {
        expect(
          screen.getByTestId(`node-evidence-${evidence.evidence_link_id}`),
        ).toBeInTheDocument();
      }
    }
    for (const source of report.sources) {
      expect(screen.getByTestId(`node-source-${source.id}`)).toBeInTheDocument();
    }
  });

  it('passes the derived edge count to the graph', () => {
    render(<EvidenceGraph report={canonicalReport()} onSelectSource={vi.fn()} />);

    // 3 claim->subclaim, 3 subclaim->evidence, 3 evidence->source
    expect(screen.getByTestId('react-flow')).toHaveAttribute('data-edge-count', '9');
  });

  it('shows a prompt before any node is selected', () => {
    const { container } = render(
      <EvidenceGraph report={canonicalReport()} onSelectSource={vi.fn()} />,
    );
    expect(
      within(panel(container)).getByText(/select a node to inspect it/i),
    ).toBeInTheDocument();
  });

  it('shows claim details when the claim node is selected', async () => {
    const user = userEvent.setup();
    const report = canonicalReport();
    const { container } = render(
      <EvidenceGraph report={report} onSelectSource={vi.fn()} />,
    );

    await user.click(screen.getByTestId(`node-claim-${report.claim.id}`));

    const details = within(panel(container));
    expect(details.getByText(report.claim.raw_text)).toBeInTheDocument();
    expect(details.getByText(/misleading/i)).toBeInTheDocument();
  });

  it('shows subclaim details when a subclaim node is selected', async () => {
    const user = userEvent.setup();
    const report = canonicalReport();
    const { container } = render(
      <EvidenceGraph report={report} onSelectSource={vi.fn()} />,
    );

    const subclaim = report.subclaims[2];
    await user.click(screen.getByTestId(`node-subclaim-${subclaim.id}`));

    const details = within(panel(container));
    expect(details.getByText(/subclaim 3/i)).toBeInTheDocument();
    expect(details.getByText(subclaim.text)).toBeInTheDocument();
    expect(details.getByText(/contradicted/i)).toBeInTheDocument();
  });

  it('shows evidence details when an evidence node is selected', async () => {
    const user = userEvent.setup();
    const report = canonicalReport();
    const { container } = render(
      <EvidenceGraph report={report} onSelectSource={vi.fn()} />,
    );

    const evidence = report.subclaims[0].evidence[0];
    await user.click(
      screen.getByTestId(`node-evidence-${evidence.evidence_link_id}`),
    );

    const details = within(panel(container));
    expect(details.getByText(evidence.quote)).toBeInTheDocument();
    expect(details.getByText(evidence.rationale)).toBeInTheDocument();
  });

  it('shows source details when a source node is selected', async () => {
    const user = userEvent.setup();
    const report = canonicalReport();
    const { container } = render(
      <EvidenceGraph report={report} onSelectSource={vi.fn()} />,
    );

    const source = report.sources[0];
    await user.click(screen.getByTestId(`node-source-${source.id}`));

    const details = within(panel(container));
    expect(details.getByText(source.title!)).toBeInTheDocument();
    expect(details.getByRole('link', { name: source.url })).toHaveAttribute(
      'href',
      source.url,
    );
  });

  it('opens the source panel from an evidence node', async () => {
    const user = userEvent.setup();
    const onSelectSource = vi.fn();
    const report = canonicalReport();
    const { container } = render(
      <EvidenceGraph report={report} onSelectSource={onSelectSource} />,
    );

    const evidence = report.subclaims[0].evidence[0];
    await user.click(
      screen.getByTestId(`node-evidence-${evidence.evidence_link_id}`),
    );
    await user.click(
      within(panel(container)).getByRole('button', { name: /open source/i }),
    );

    expect(onSelectSource).toHaveBeenCalledWith(evidence.source_id);
  });

  it('closes the detail panel', async () => {
    const user = userEvent.setup();
    const report = canonicalReport();
    const { container } = render(
      <EvidenceGraph report={report} onSelectSource={vi.fn()} />,
    );

    await user.click(screen.getByTestId(`node-claim-${report.claim.id}`));
    expect(within(panel(container)).getByText(report.claim.raw_text)).toBeInTheDocument();

    await user.click(within(panel(container)).getByRole('button', { name: /^close$/i }));
    expect(
      within(panel(container)).getByText(/select a node to inspect it/i),
    ).toBeInTheDocument();
  });
});
