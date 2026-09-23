import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    watch: { usePolling: process.env.VITE_USE_POLLING === 'true' },
    proxy: { '/memory': process.env.MEMORY_API_PROXY || 'http://127.0.0.1:8010' },
  },
});
