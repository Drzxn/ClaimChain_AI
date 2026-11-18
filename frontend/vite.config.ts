import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// The frontend dev server runs on 5173, which is the origin the backend's
// CORS policy allows. The backend is called directly via VITE_API_BASE_URL.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
  },
});
