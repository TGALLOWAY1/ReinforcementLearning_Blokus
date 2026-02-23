import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        // WebSocket proxy target should be HTTP URL (Vite converts it to WS)
        target: process.env.VITE_API_URL || 'http://localhost:8000',
        ws: true,
      },
    },
  },
})
