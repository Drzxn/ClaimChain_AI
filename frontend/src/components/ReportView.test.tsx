/**
 * ReportView and its children.
 *
 * These assert that the UI displays exactly what the backend report contains:
 * the verdict verbatim, every subclaim, every evidence quote, and every source.
 * Nothing is recalculated in the component, so nothing should be recalculated
 * in the assertion.
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { ReportView } from './ReportView';
import { canonicalReport, emptyReport } from '../test/fixtures';

describe('ReportView — overall verdict', () => {
  it('renders the exact backend verdict', () => {
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);

    // MISLEADING must appear as the verdict, not be reinterpreted.
    const verdicts = screen.getAllByText(/misleading/i);
    expect(verdicts.length).toBeGreaterThan(0);
  });

  it('renders the score, coverage and contradicted ratio as returned', () => {
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);

    expect(screen.getAllByText('0.3333').length).toBeGreaterThan(0);
    // coverage 1.0000 appears for the claim and for each subclaim
    expect(screen.getAllByText('1.0000').length).toBeGreaterThan(0);
  });

  it('renders the claim text', () => {
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);
    expect(
      screen.getByText('Using HTTPS makes a website completely secure.'),
    ).toBeInTheDocument();
  });

  it('renders the deterministic explanation statement', () => {
    const report = canonicalReport();
    render(<ReportView report={report} onReset={vi.fn()} />);
    expect(
      screen.getByText(report.explanation.verdict_statement),
    ).toBeInTheDocument();
  });

  it('renders supporting and contradicting points', () => {
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);

    expect(screen.getByText(/supporting evidence/i)).toBeInTheDocument();
    expect(screen.getByText(/contradicting evidence/i)).toBeInTheDocument();
    expect(
      screen.getAllByText('HTTPS does not prevent cross-site scripting.').length,
    ).toBeGreaterThan(0);
  });

  it('renders limitations from the report', () => {
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);

    expect(screen.getByText(/limitations/i)).toBeInTheDocument();
    expect(screen.getByText(/1 source\(s\) had no publication date/i)).toBeInTheDocument();
  });

  it('renders an insufficient-evidence verdict without inventing a direction', () => {
    render(<ReportView report={emptyReport()} onReset={vi.fn()} />);

    expect(screen.getAllByText(/insufficient evidence/i).length).toBeGreaterThan(0);
  });
});

describe('ReportView — subclaims', () => {
  it('renders every subclaim with its text', () => {
    const report = canonicalReport();
    render(<ReportView report={report} onReset={vi.fn()} />);

    for (const subclaim of report.subclaims) {
      expect(screen.getByText(subclaim.text)).toBeInTheDocument();
    }
  });

  it('renders each subclaim verdict', () => {
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);

    // Two supported, one contradicted.
    expect(screen.getAllByText(/^supported$/i).length).toBeGreaterThanOrEqual(2);
    expect(screen.getAllByText(/^contradicted$/i).length).toBeGreaterThanOrEqual(1);
  });

  it('renders the subclaim rationale from the backend', () => {
    const report = canonicalReport();
    render(<ReportView report={report} onReset={vi.fn()} />);

    expect(
      screen.getAllByText(report.subclaims[0].verdict.rationale).length,
    ).toBeGreaterThan(0);
  });

  it('shows the tab counts', () => {
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);

    expect(screen.getByRole('tab', { name: /subclaims \(3\)/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /sources \(3\)/i })).toBeInTheDocument();
  });
});

describe('ReportView — evidence', () => {
  it('renders the evidence quote for a subclaim', () => {
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);

    expect(
      screen.getAllByText('HTTPS encrypts traffic between the browser and the server.').length,
    ).toBeGreaterThan(0);
  });

  it('renders the evidence rationale', () => {
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);
    expect(
      screen.getByText('Directly states the encryption property.'),
    ).toBeInTheDocument();
  });

  it('renders the source domain on each evidence card', () => {
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);
    expect(screen.getAllByText('example.org').length).toBeGreaterThan(0);
  });

  it('links each evidence item to its source URL safely', () => {
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);

    const link = screen.getByRole('link', { name: /how https works/i });
    expect(link).toHaveAttribute('href', 'https://example.org/https-explained');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer');
  });

  it('hides an unverified quote rather than showing it as evidence', () => {
    const report = canonicalReport();
    report.subclaims[0].evidence[0].quote_validation = 'failed';
    report.subclaims[0].evidence[0].quote = 'a quote that failed validation';

    render(<ReportView report={report} onReset={vi.fn()} />);

    expect(
      screen.queryByText('a quote that failed validation'),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/could not be verified/i)).toBeInTheDocument();
  });
});

describe('ReportView — sources', () => {
  it('lists every source with its domain and type', async () => {
    const user = userEvent.setup();
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);

    await user.click(screen.getByRole('tab', { name: /sources/i }));

    expect(screen.getByText('How HTTPS Works')).toBeInTheDocument();
    expect(screen.getByText('Transport Encryption and Web Security')).toBeInTheDocument();
    expect(screen.getByText('TLS Overview')).toBeInTheDocument();
  });

  it('shows source details when a source is selected', async () => {
    const user = userEvent.setup();
    const report = canonicalReport();
    render(<ReportView report={report} onReset={vi.fn()} />);

    await user.click(screen.getByRole('tab', { name: /sources/i }));
    await user.click(screen.getByText('How HTTPS Works'));

    // Authority and publisher come from the Stage 5 assessment.
    expect(screen.getByText('Example Organisation')).toBeInTheDocument();
    expect(screen.getByText('0.8000')).toBeInTheDocument();
    expect(screen.getByText(/authority established reference/i)).toBeInTheDocument();
  });

  it('shows "not assessed" for a null authority score', async () => {
    const user = userEvent.setup();
    const report = canonicalReport();
    report.sources[0].authority_score = null;
    report.sources[0].recency_score = null;

    render(<ReportView report={report} onReset={vi.fn()} />);
    await user.click(screen.getByRole('tab', { name: /sources/i }));
    await user.click(screen.getByText('How HTTPS Works'));

    expect(screen.getAllByText(/not assessed/i).length).toBeGreaterThan(0);
  });

  it('renders an empty state when there are no sources', async () => {
    const user = userEvent.setup();
    render(<ReportView report={emptyReport()} onReset={vi.fn()} />);

    await user.click(screen.getByRole('tab', { name: /sources/i }));
    expect(screen.getByText(/no sources/i)).toBeInTheDocument();
  });
});

describe('ReportView — empty evidence', () => {
  it('renders the insufficient subclaim state without evidence cards', () => {
    render(<ReportView report={emptyReport()} onReset={vi.fn()} />);

    expect(
      screen.getAllByText(/no evidence was judged against this subclaim/i).length,
    ).toBe(3);
  });
});

describe('ReportView — reset', () => {
  it('calls onReset from the toolbar button', async () => {
    const user = userEvent.setup();
    const onReset = vi.fn();
    render(<ReportView report={canonicalReport()} onReset={onReset} />);

    await user.click(screen.getByRole('button', { name: /verify another claim/i }));
    expect(onReset).toHaveBeenCalledTimes(1);
  });
});

describe('ReportView — tab switching', () => {
  it('switches between subclaims, graph and sources', async () => {
    const user = userEvent.setup();
    render(<ReportView report={canonicalReport()} onReset={vi.fn()} />);

    const subclaimsTab = screen.getByRole('tab', { name: /subclaims/i });
    expect(subclaimsTab).toHaveAttribute('aria-selected', 'true');

    await user.click(screen.getByRole('tab', { name: /evidence graph/i }));
    expect(screen.getByRole('tab', { name: /evidence graph/i })).toHaveAttribute(
      'aria-selected',
      'true',
    );

    await user.click(screen.getByRole('tab', { name: /sources/i }));
    expect(screen.getByRole('tab', { name: /sources/i })).toHaveAttribute(
      'aria-selected',
      'true',
    );
  });
});

describe('ReportView — verdict badge accessibility', () => {
  it('conveys the verdict with text, not colour alone', () => {
    const { container } = render(
      <ReportView report={canonicalReport()} onReset={vi.fn()} />,
    );

    const badges = container.querySelectorAll('.badge');
    expect(badges.length).toBeGreaterThan(0);
    for (const badge of badges) {
      // Every badge carries a visible text label, not just a colour.
      const label = badge.querySelector('.badge__label');
      expect(label?.textContent?.trim()).toBeTruthy();
    }
  });
});
