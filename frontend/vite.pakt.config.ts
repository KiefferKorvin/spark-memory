import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Build once into PAKT's static assets; Flask deployments need no Node process.
export default defineConfig({
  plugins: [react()],
  base: '/static/atlas-explorer/',
  build: {
    outDir: '../../pakt/static/atlas-explorer',
    emptyOutDir: true,
    rollupOptions: {
      input: 'src/pakt.tsx',
      output: { entryFileNames: 'explorer.js', assetFileNames: asset => asset.names?.some(n => n.endsWith('.css')) ? 'explorer.css' : 'assets/[name]-[hash][extname]' },
    },
  },
});
