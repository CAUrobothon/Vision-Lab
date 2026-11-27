import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server on 5190 forwards /api to the FastAPI backend on 8010.
// `npm run build` writes dist/, which the backend serves at http://127.0.0.1:8010/.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5190,
    strictPort: true,
    proxy: {
      '/api': { target: process.env.LAB_API_URL ?? 'http://127.0.0.1:8010', changeOrigin: true },
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
