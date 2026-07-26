/**
 * WebSocket 事件流（docs/09 第六节）。
 *
 * 两处不能省的细节：
 *
 * 1. **断线重连要带 `since`**。验收 8 要求"断开后重连，任务进度与订单状态
 *    能正确恢复，无事件丢失"。不带序号重连，断线那几秒的事件就永久消失了，
 *    界面上的进度条会卡在中途再也不动；
 * 2. **重连退避要有上限**。固定 1 秒重试会在后端没起来时打出每秒一次的
 *    无限循环；指数退避不封顶又会在长时间断开后退到几十分钟，用户重启了
 *    后端也半天连不上。这里 1s 起、翻倍、封顶 15s。
 */

import { defineStore } from 'pinia'
import { ref } from 'vue'

import { getToken } from '@/api/client'
import type { WsEvent } from '@/api/types'

const MAX_KEPT = 200
const BASE_DELAY_MS = 1000
const MAX_DELAY_MS = 15000

export const useEventStore = defineStore('events', () => {
  const connected = ref(false)
  const events = ref<WsEvent[]>([])
  const lastSeq = ref(0)
  const taskProgress = ref<Record<string, WsEvent>>({})

  let socket: WebSocket | null = null
  let retryDelay = BASE_DELAY_MS
  let timer: number | null = null
  let closedByUs = false

  function handle(event: WsEvent): void {
    if (typeof event.seq !== 'number') return
    lastSeq.value = Math.max(lastSeq.value, event.seq)
    events.value = [event, ...events.value].slice(0, MAX_KEPT)
    const task = event.payload?.task
    if (event.channel === 'tasks' && typeof task === 'string') {
      taskProgress.value = { ...taskProgress.value, [task]: event }
    }
  }

  function connect(): void {
    if (socket && socket.readyState <= WebSocket.OPEN) return
    const token = getToken()
    if (!token) return

    closedByUs = false
    const scheme = location.protocol === 'https:' ? 'wss' : 'ws'
    // WebSocket 不能带自定义头，口令只能走查询串
    const url = `${scheme}://${location.host}/ws?token=${encodeURIComponent(token)}&since=${lastSeq.value}`
    socket = new WebSocket(url)

    socket.onopen = () => {
      connected.value = true
      retryDelay = BASE_DELAY_MS
    }
    socket.onmessage = (message) => {
      try {
        handle(JSON.parse(message.data as string) as WsEvent)
      } catch {
        // 单条消息解析失败不该拖垮整条连接
      }
    }
    socket.onclose = () => {
      connected.value = false
      socket = null
      if (closedByUs) return
      timer = window.setTimeout(connect, retryDelay)
      retryDelay = Math.min(retryDelay * 2, MAX_DELAY_MS)
    }
    socket.onerror = () => socket?.close()
  }

  function disconnect(): void {
    closedByUs = true
    if (timer !== null) {
      window.clearTimeout(timer)
      timer = null
    }
    socket?.close()
    socket = null
    connected.value = false
  }

  return { connected, events, lastSeq, taskProgress, connect, disconnect }
})
