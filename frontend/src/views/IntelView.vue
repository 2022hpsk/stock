<script setup lang="ts">
/**
 * P5 情报（docs/09 第三节）：时间线、分域摘要、拖拽导入、源健康、黑名单。
 *
 * 界面上要一直可见的两件事：
 *
 * - **每条情报都带原文链接**（红线 I-R4）。没有出处的情报看着像证据、
 *   实际上不可核实，比没有更危险；
 * - **今日无情报的域要显式列出**。分不清"没查到"和"查了没有"是最糟的状态。
 *
 * 这里**没有任何"据此下单"的入口**——情报不能单独触发买入（红线 I-R1）。
 */
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'

import { api, ApiError } from '@/api/client'
import type { IntelDigest, IntelItem } from '@/api/types'
import { useSystemStore } from '@/stores/system'

const system = useSystemStore()

const digest = ref<IntelDigest | null>(null)
const status = ref<{ sources: number; inbox_pending: number; blacklisted: number; message: string; inbox_dir: string } | null>(null)
const blacklist = ref<Array<{ symbol: string; reason: string; rule: string; triggered_at: string; expires_at: string; urls: string[] }>>([])
const health = ref<Array<{ source: string; ok: boolean; fetched: number; error: string }>>([])
const busy = ref(false)
const errorMessage = ref('')

const domainFilter = ref('')
const minImportance = ref(0)
const keyword = ref('')

const note = ref({ text: '', title: '', url: '', domain: '', symbols: '', importance: 50 })

const allItems = computed<IntelItem[]>(() => {
  if (!digest.value) return []
  const seen = new Set<string>()
  const out: IntelItem[] = []
  for (const bucket of Object.values(digest.value.by_domain)) {
    for (const item of bucket.items) {
      if (seen.has(item.item_id)) continue
      seen.add(item.item_id)
      out.push(item)
    }
  }
  return out.sort((a, b) => (b.publish_at ?? '').localeCompare(a.publish_at ?? ''))
})

const filtered = computed(() =>
  allItems.value.filter((i) => {
    if (domainFilter.value && i.domain !== domainFilter.value) return false
    if (i.importance < minImportance.value) return false
    if (keyword.value && !`${i.title}${i.body}`.includes(keyword.value)) return false
    return true
  }),
)

const domains = computed(() => Object.keys(digest.value?.by_domain ?? {}))

async function loadAll(): Promise<void> {
  errorMessage.value = ''
  const results = await Promise.allSettled([
    api.get<typeof status.value>('/api/intel/status'),
    api.get<IntelDigest>('/api/intel/digest', { lookback_days: 7 }),
    api.get<{ entries: typeof blacklist.value }>('/api/intel/blacklist'),
  ])
  if (results[0].status === 'fulfilled') status.value = results[0].value
  if (results[1].status === 'fulfilled') digest.value = results[1].value
  if (results[2].status === 'fulfilled') blacklist.value = results[2].value.entries
}

async function fetchNow(): Promise<void> {
  busy.value = true
  errorMessage.value = ''
  try {
    const res = await api.post<{ summary: string }>('/api/intel/fetch', { lookback_days: 7 })
    ElMessage.success(res.summary)
    await loadAll()
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

async function checkHealth(): Promise<void> {
  busy.value = true
  try {
    const res = await api.get<{ health: typeof health.value }>('/api/intel/health')
    health.value = res.health
  } finally {
    busy.value = false
  }
}

async function submitNote(): Promise<void> {
  if (!note.value.text.trim()) {
    ElMessage.warning('正文不能为空')
    return
  }
  busy.value = true
  try {
    await api.post('/api/intel/note', {
      text: note.value.text,
      title: note.value.title,
      url: note.value.url,
      domain: note.value.domain || null,
      importance: note.value.importance,
      symbols: note.value.symbols
        .split(/[,，\s]+/)
        .map((s) => s.trim())
        .filter(Boolean),
    })
    ElMessage.success('已录入')
    note.value = { text: '', title: '', url: '', domain: '', symbols: '', importance: 50 }
    await loadAll()
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

async function onDrop(event: DragEvent): Promise<void> {
  event.preventDefault()
  const file = event.dataTransfer?.files?.[0]
  if (!file) return
  busy.value = true
  try {
    const res = await api.upload<{ ok: boolean; parsed?: number; error?: string }>('/api/intel/upload', file)
    if (res.ok) {
      ElMessage.success(`已解析并入库 ${res.parsed} 条`)
      await loadAll()
    } else {
      ElMessage.error(res.error ?? '导入失败')
    }
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

function sentimentClass(v: number | null): string {
  if (v === null || Math.abs(v) < 0.1) return 'qs-flat'
  return v > 0 ? 'qs-up' : 'qs-down'
}

onMounted(loadAll)
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>情报</h2>
        <div class="qs-sub">
          情报只用于解释、有界软调节与单向风险否决，**不能单独触发买入**。每条都可点回原文。
        </div>
      </div>
      <div style="display: flex; gap: 8px">
        <el-button :loading="busy" @click="checkHealth">检查源健康</el-button>
        <el-button type="primary" :loading="busy" :disabled="system.status?.readonly" @click="fetchNow">
          立即采集
        </el-button>
      </div>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" style="margin-bottom: 12px" />
    <el-alert
      v-if="digest?.missing_domains.length"
      type="info"
      show-icon
      :closable="false"
      :title="`今日无情报的域：${digest.missing_domains.join('、')}`"
      description="明写出来是为了区分「没查到」和「查了没有」——后者才是可以放心的状态。"
      style="margin-bottom: 12px"
    />

    <el-row :gutter="16">
      <el-col :span="16">
        <el-card shadow="never">
          <template #header>
            <div class="qs-filters">
              <span>情报流（{{ filtered.length }}）</span>
              <el-select v-model="domainFilter" placeholder="全部域" clearable style="width: 130px" size="small">
                <el-option v-for="d in domains" :key="d" :label="d" :value="d" />
              </el-select>
              <el-input-number v-model="minImportance" :min="0" :max="100" :step="10" size="small" style="width: 120px" />
              <el-input v-model="keyword" placeholder="全文搜索" clearable size="small" style="width: 160px" />
            </div>
          </template>

          <div v-if="!filtered.length" class="qs-empty">
            暂无情报。点「立即采集」，或把 .md / .txt / .json / .csv 拖到右侧导入区。
          </div>
          <div v-else class="qs-timeline">
            <div v-for="item in filtered" :key="item.item_id" class="qs-item">
              <div class="qs-item-head">
                <el-tag size="small" type="info">{{ item.domain }}</el-tag>
                <el-tag v-if="item.importance >= 70" size="small" type="danger">
                  重要 {{ item.importance }}
                </el-tag>
                <el-tag v-else size="small">{{ item.importance }}</el-tag>
                <span :class="sentimentClass(item.sentiment)">
                  情绪 {{ item.sentiment?.toFixed(2) ?? '—' }}
                </span>
                <el-tag v-if="item.classifier.startsWith('llm')" size="small" type="warning">🤖</el-tag>
                <span class="qs-meta">{{ item.source }} · {{ item.publish_at }}</span>
              </div>
              <a v-if="item.url" :href="item.url" target="_blank" rel="noopener noreferrer" class="qs-item-title">
                {{ item.title }}
              </a>
              <div v-else class="qs-item-title qs-no-url">{{ item.title }}（无原文链接）</div>
              <p v-if="item.body" class="qs-item-body">{{ item.body.slice(0, 220) }}</p>
              <div v-if="item.symbols.length" class="qs-meta">
                关联标的：<span class="qs-mono">{{ item.symbols.join('、') }}</span>
              </div>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :span="8">
        <el-card shadow="never">
          <template #header>外置导入</template>
          <div class="qs-drop" @drop="onDrop" @dragover.prevent>
            把 <code>.md</code> / <code>.txt</code> / <code>.json</code> / <code>.csv</code> 拖到这里
            <div class="qs-meta">收件箱：{{ status?.inbox_dir ?? '—' }}</div>
          </div>

          <el-divider>或手工录入一条</el-divider>
          <el-form label-position="top" size="small">
            <el-form-item label="正文（必填）">
              <el-input v-model="note.text" type="textarea" :rows="3" />
            </el-form-item>
            <el-form-item label="标题">
              <el-input v-model="note.title" />
            </el-form-item>
            <el-form-item label="原文链接">
              <el-input v-model="note.url" placeholder="https://…" />
            </el-form-item>
            <el-form-item label="关联标的（逗号分隔）">
              <el-input v-model="note.symbols" placeholder="600519.SH, 000001.SZ" />
            </el-form-item>
            <el-form-item label="重要性">
              <el-slider v-model="note.importance" :min="0" :max="100" />
            </el-form-item>
            <el-button type="primary" :loading="busy" :disabled="system.status?.readonly" @click="submitNote">
              录入
            </el-button>
          </el-form>
        </el-card>

        <el-card v-if="health.length" shadow="never" style="margin-top: 16px">
          <template #header>源健康</template>
          <el-table :data="health" size="small">
            <el-table-column prop="source" label="源" width="100" />
            <el-table-column label="状态" width="70">
              <template #default="{ row }">
                <el-tag :type="row.ok ? 'success' : 'danger'" size="small">
                  {{ row.ok ? '可用' : '不可用' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="error" label="说明" show-overflow-tooltip />
          </el-table>
        </el-card>

        <el-card shadow="never" style="margin-top: 16px">
          <template #header>情报黑名单（{{ blacklist.length }}）</template>
          <div v-if="!blacklist.length" class="qs-empty">无</div>
          <div v-for="e in blacklist" :key="e.symbol" class="qs-bl">
            <strong class="qs-mono">{{ e.symbol }}</strong>
            <div class="qs-meta">{{ e.rule }} · 至 {{ e.expires_at }}</div>
            <div>{{ e.reason }}</div>
            <div v-if="e.urls.length">
              <a v-for="(u, i) in e.urls" :key="i" :href="u" target="_blank" rel="noopener noreferrer">
                原文 {{ i + 1 }}
              </a>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.qs-filters {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.qs-timeline {
  max-height: 680px;
  overflow: auto;
}
.qs-item {
  padding: 10px 0;
  border-bottom: 1px solid var(--qs-border);
}
.qs-item-head {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
  font-size: 12px;
  margin-bottom: 4px;
}
.qs-item-title {
  display: block;
  font-weight: 600;
  margin-bottom: 4px;
}
.qs-no-url {
  color: var(--qs-muted);
}
.qs-item-body {
  margin: 4px 0;
  color: #4b5563;
  font-size: 13px;
  line-height: 1.7;
}
.qs-meta {
  color: var(--qs-muted);
  font-size: 12px;
}
.qs-drop {
  border: 2px dashed var(--qs-border);
  border-radius: 8px;
  padding: 24px 12px;
  text-align: center;
  color: var(--qs-muted);
  font-size: 13px;
}
.qs-bl {
  padding: 8px 0;
  border-bottom: 1px solid var(--qs-border);
  font-size: 13px;
}
</style>
