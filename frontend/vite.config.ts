import { fileURLToPath, URL } from 'node:url'

import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

// 构建产物直接落进 Python 包内，随包分发（docs/09 第二节）。
// 用户 `uv sync` 后 `quantstock ui` 即可，不需要装 Node。
export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  build: {
    outDir: '../src/quantstock/web/dist',
    emptyOutDir: true,
    // 验收 9：产物不得引用任何外部 CDN。所有依赖打进本地 chunk，
    // 离线可用，也避免把访问行为泄漏给第三方
    assetsInlineLimit: 4096,
    rollupOptions: {
      output: {
        manualChunks: {
          // ECharts 体积大且很少变，单独切出来让浏览器长期缓存
          echarts: ['echarts'],
          element: ['element-plus'],
        },
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8765', changeOrigin: true },
      '/ws': { target: 'ws://127.0.0.1:8765', ws: true },
    },
  },
})
