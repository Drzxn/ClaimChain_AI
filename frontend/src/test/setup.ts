/**
 * Global test setup.
 *
 * Nothing here touches the network. Each test installs its own `fetch` mock;
 * this file only adds the DOM matchers, the browser APIs jsdom lacks, and
 * cleanup between tests.
 */

import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeEach, vi } from 'vitest';

// jsdom does not implement ResizeObserver, which React Flow uses to measure
// its container. A no-op stand-in is enough: the graph's layout is computed
// deterministically in lib/graph.ts, not measured at runtime.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

if (!('ResizeObserver' in globalThis)) {
  vi.stubGlobal('ResizeObserver', ResizeObserverStub);
}

// React Flow also reads these during mount.
if (!('DOMMatrixReadOnly' in globalThis)) {
  vi.stubGlobal(
    'DOMMatrixReadOnly',
    class {
      m22 = 1;
      constructor(_transform?: string) {}
    },
  );
}

beforeEach(() => {
  // A test that forgets to mock fetch should fail loudly rather than hit the
  // real backend. Individual tests override this with vi.stubGlobal.
  vi.stubGlobal(
    'fetch',
    vi.fn(() => {
      throw new Error(
        'fetch was called without a mock. Every test must stub fetch.',
      );
    }),
  );
});

afterEach(() => {
  cleanup();
});
