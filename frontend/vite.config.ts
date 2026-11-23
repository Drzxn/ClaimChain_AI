import { defineConfig } from "vitest/config";
import react from '@vitejs/plugin-react';

// The frontend dev server runs on 5173, which is the origin the backend's
// CORS policy allows. The backend is called directly via VITE_API_BASE_URL.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
  test: {
    // jsdom gives the components a DOM without a browser.
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    // No test may reach the network. Every fetch is mocked per test.
    restoreMocks: true,
    clearMocks: true,
  },
});
