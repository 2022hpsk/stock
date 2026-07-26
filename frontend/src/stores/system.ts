/**
 * 系统状态与急停。
 *
 * 急停状态是**全局的**：一旦 `halted`，界面上所有下单入口必须变灰
 * （docs/09 第五节）。这是体验层的防护——真正的拦截在后端的 `HaltSwitch`，
 * 前端变灰只是让人别白点。
 */

import { defineStore } from 'pinia'
import { ref } from 'vue'

import { api } from '@/api/client'
import type { SystemStatus } from '@/api/types'

export const useSystemStore = defineStore('system', () => {
  const status = ref<SystemStatus | null>(null)
  const loading = ref(false)
  const error = ref('')

  async function refresh(): Promise<void> {
    loading.value = true
    error.value = ''
    try {
      status.value = await api.get<SystemStatus>('/api/system/status')
    } catch (e) {
      error.value = e instanceof Error ? e.message : String(e)
      status.value = null
    } finally {
      loading.value = false
    }
  }

  async function halt(reason: string): Promise<void> {
    await api.post('/api/system/halt', { reason, by: 'ui' })
    await refresh()
  }

  async function resume(): Promise<void> {
    await api.post('/api/system/resume')
    await refresh()
  }

  return { status, loading, error, refresh, halt, resume }
})
