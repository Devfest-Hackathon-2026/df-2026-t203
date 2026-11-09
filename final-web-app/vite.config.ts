import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const backend = 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(
      ['/api', '/ws', '/teacher_videos', '/recordings', '/outputs'].map((path) => [path, { target: backend, ws: path === '/ws' }]),
    ),
  },
})
