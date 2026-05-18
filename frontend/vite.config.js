import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  build: {
    // Option A: output directly into ../static so FastAPI serves it on port 8000
    outDir: path.resolve(__dirname, '../static'),
    emptyOutDir: true,
  }
})
