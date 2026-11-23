/**
 * ClaimInput component tests.
 *
 * Covers the entry screen: it renders, it validates, and submitting calls the
 * parent with the trimmed claim text. No network is involved: `onSubmit` is a
 * spy, so this tests the component contract, not the pipeline.
 */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { ClaimInput } from './ClaimInput';

describe('ClaimInput', () => {
  it('renders the claim field and the submit button', () => {
    render(<ClaimInput onSubmit={vi.fn()} busy={false} />);

    expect(screen.getByRole('heading', { name: /verify a claim/i })).toBeInTheDocument();
    expect(screen.getByRole('textbox')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /verify claim/i })).toBeInTheDocument();
  });

  it('shows the character counter', () => {
    render(<ClaimInput onSubmit={vi.fn()} busy={false} />);
    expect(screen.getByText(/0 \/ 2000 characters/)).toBeInTheDocument();
  });

  it('disables submit while the claim is empty', () => {
    render(<ClaimInput onSubmit={vi.fn()} busy={false} />);
    expect(screen.getByRole('button', { name: /verify claim/i })).toBeDisabled();
  });

  it('enables submit once text is entered', async () => {
    const user = userEvent.setup();
    render(<ClaimInput onSubmit={vi.fn()} busy={false} />);

    await user.type(screen.getByRole('textbox'), 'HTTPS is secure.');
    expect(screen.getByRole('button', { name: /verify claim/i })).toBeEnabled();
  });

  it('calls onSubmit with the trimmed claim text', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ClaimInput onSubmit={onSubmit} busy={false} />);

    await user.type(
      screen.getByRole('textbox'),
      '  Using HTTPS makes a website completely secure.  ',
    );
    await user.click(screen.getByRole('button', { name: /verify claim/i }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith(
      'Using HTTPS makes a website completely secure.',
    );
  });

  it('does not submit whitespace-only input', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<ClaimInput onSubmit={onSubmit} busy={false} />);

    await user.type(screen.getByRole('textbox'), '     ');
    expect(screen.getByRole('button', { name: /verify claim/i })).toBeDisabled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('fills the field from an example button', async () => {
    const user = userEvent.setup();
    render(<ClaimInput onSubmit={vi.fn()} busy={false} />);

    await user.click(
      screen.getByRole('button', {
        name: 'Using HTTPS makes a website completely secure.',
      }),
    );

    expect(screen.getByRole('textbox')).toHaveValue(
      'Using HTTPS makes a website completely secure.',
    );
  });

  it('disables the form while busy', () => {
    render(<ClaimInput onSubmit={vi.fn()} busy={true} />);

    expect(screen.getByRole('textbox')).toBeDisabled();
    expect(screen.getByRole('button', { name: /verifying/i })).toBeDisabled();
  });

  it('renders an error message when one is passed', () => {
    render(
      <ClaimInput onSubmit={vi.fn()} busy={false} error="Claim is too long." />,
    );
    expect(screen.getByRole('alert')).toHaveTextContent('Claim is too long.');
  });
});
