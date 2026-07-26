/**
 * HTTP 客户端。
 *
 * 所有请求带 `X-Access-Token`。口令由后端启动时随机生成并打印到终端，
 * 用户首次进入界面时填一次，存 localStorage（docs/09 第五节）。
 *
 * **金额一律按字符串处理**：后端返回的价格、市值都是字符串而不是数字。
 * 前端只做展示与拼接，绝不用 `parseFloat` 去算总额——那样会在界面上
 * 显示出 `1596.5200000000001` 这种数字，而用户看到的是钱。
 */

const TOKEN_KEY = 'quantstock.token'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly context: Record<string, string> = {},
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) ?? ''
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token.trim())
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      'X-Access-Token': getToken(),
      ...(init.headers ?? {}),
    },
  })

  if (response.status === 204) return undefined as T

  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    // 后端把领域异常转成 {error, message, context}，把 context 一并带上——
    // "数据不足" 远不如 "数据不足：600519.SH 只有 12 根 K 线，需要 60 根" 有用
    const message =
      (payload as { message?: string; detail?: string }).message ??
      (payload as { detail?: string }).detail ??
      `请求失败（HTTP ${response.status}）`
    throw new ApiError(message, response.status, (payload as { context?: Record<string, string> }).context ?? {})
  }
  return payload as T
}

export const api = {
  get: <T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> => {
    const query = params
      ? '?' +
        new URLSearchParams(
          Object.entries(params)
            .filter(([, v]) => v !== undefined && v !== '')
            .map(([k, v]) => [k, String(v)]),
        ).toString()
      : ''
    return request<T>(`${path}${query}`)
  },
  post: <T>(path: string, body?: unknown): Promise<T> =>
    request<T>(path, { method: 'POST', body: JSON.stringify(body ?? {}) }),
  put: <T>(path: string, body?: unknown): Promise<T> =>
    request<T>(path, { method: 'PUT', body: JSON.stringify(body ?? {}) }),
  upload: async <T>(path: string, file: File): Promise<T> => {
    const form = new FormData()
    form.append('file', file)
    const response = await fetch(path, {
      method: 'POST',
      headers: { 'X-Access-Token': getToken() },
      body: form,
    })
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new ApiError(
        (payload as { message?: string; detail?: string }).message ??
          (payload as { detail?: string }).detail ??
          '上传失败',
        response.status,
      )
    }
    return payload as T
  },
}
