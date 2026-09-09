import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// A local, static editor: MuJoCo runs in Python, without a cloud runtime.
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': fileURLToPath(new URL('.', import.meta.url)) } },
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
  build: { outDir: 'dist', chunkSizeWarningLimit: 1200 },
});
