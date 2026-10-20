import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';

// A local, static editor: MuJoCo runs in Python, without a cloud runtime.
export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': fileURLToPath(new URL('.', import.meta.url)) } },
  server: { proxy: { '/api': { target: 'http://127.0.0.1:8765', ws: true } } },
  build: { outDir: 'dist', chunkSizeWarningLimit: 1200, rollupOptions: { input: {
    main: fileURLToPath(new URL('index.html', import.meta.url)),
    cylinderLab: fileURLToPath(new URL('cylinder-lab.html', import.meta.url)),
  } } },
});
