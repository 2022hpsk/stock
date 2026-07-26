<script setup lang="ts">
/**
 * P4 数据（docs/09 第三节）：源健康、更新任务、K 线浏览器。
 */
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'

import { api, ApiError } from '@/api/client'
import type { Bar, DataStatus } from '@/api/types'
import CandleChart from '@/components/CandleChart.vue'
import { useSystemStore } from '@/stores/system'

const system = useSystemStore()

const status = ref<DataStatus | null>(null)
const universe = ref<string[]>([])
const selectedSymbol = ref('')
const bars = ref<Bar[]>([])
const adjust = ref('hfq')
const updating = ref(false)
const loadingBars = ref(false)
const errorMessage = ref('')

async function loadStatus(): Promise<void> {
  status.value = await api.get<DataStatus>('/api/data/status')
}

async function loadUniverse(): Promise<void> {
  const res = await api.get<{ symbols: string[] }>('/api/data/universe', { tier: 'core' })
  universe.value = res.symbols
  if (!selectedSymbol.value && res.symbols.length) selectedSymbol.value = res.symbols[0]
}

async function loadBars(): Promise<void> {
  if (!selectedSymbol.value) return
  loadingBars.value = true
  errorMessage.value = ''
  try {
    const res = await api.get<{ bars: Bar[]; adjust: string }>('/api/data/bars', {
      symbol: selectedSymbol.value,
      limit: 500,
    })
    bars.value = res.bars
    adjust.value = res.adjust
    if (!res.bars.length) errorMessage.value = '该标的在数据湖里没有行情，请先运行更新'
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    loadingBars.value = false
  }
}

async function runUpdate(): Promise<void> {
  updating.value = true
  errorMessage.value = ''
  try {
    const res = await api.post<{ summary: string }>('/api/data/update', {
      tier: 'core',
      sync_instruments: true,
    })
    ElMessage.success(res.summary)
    await loadStatus()
    await loadBars()
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    updating.value = false
  }
}

onMounted(async () => {
  await Promise.all([loadStatus(), loadUniverse()])
  await loadBars()
})
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>数据</h2>
        <div class="qs-sub">
          数据是一切的前置条件。这里的"最新行情日"落后于今天时，当天的建议就不该用。
        </div>
      </div>
      <el-button
        type="primary"
        :loading="updating"
        :disabled="system.status?.readonly"
        @click="runUpdate"
      >
        增量更新
      </el-button>
    </div>

    <el-alert v-if="errorMessage" type="warning" :title="errorMessage" show-icon :closable="false" style="margin-bottom: 12px" />

    <div class="qs-cards">
      <div class="qs-card">
        <div class="qs-label">标的数</div>
        <div class="qs-value">{{ status?.symbols ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">分区文件</div>
        <div class="qs-value">{{ status?.files ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">占用空间</div>
        <div class="qs-value">
          {{ status ? (status.bytes_on_disk / 1024 / 1024).toFixed(1) + ' MB' : '—' }}
        </div>
      </div>
      <div class="qs-card">
        <div class="qs-label">最新行情日</div>
        <div class="qs-value qs-mono" style="font-size: 18px">{{ status?.latest_date ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">标的表</div>
        <div class="qs-value">
          {{ status?.instruments ?? '—' }}
          <span class="qs-sub" v-if="status">（退市 {{ status.delisted }}）</span>
        </div>
      </div>
    </div>

    <el-card shadow="never" style="margin-bottom: 16px">
      <template #header>数据源健康</template>
      <div v-if="!status?.health.length" class="qs-empty">暂无健康记录</div>
      <el-table v-else :data="status.health" size="small">
        <el-table-column prop="source" label="数据源" width="140" />
        <el-table-column label="状态" width="80">
          <template #default="{ row }">
            <el-tag :type="row.ok ? 'success' : 'danger'" size="small">
              {{ row.ok ? '可用' : '不可用' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="checked_at" label="检查时间" width="220" />
        <el-table-column prop="consecutive_failures" label="连续失败" width="90" />
        <el-table-column prop="message" label="说明" show-overflow-tooltip />
      </el-table>
    </el-card>

    <el-card shadow="never">
      <template #header>
        <div style="display: flex; justify-content: space-between; align-items: center">
          <span>K 线浏览器</span>
          <el-select v-model="selectedSymbol" style="width: 170px" filterable @change="loadBars">
            <el-option v-for="s in universe" :key="s" :label="s" :value="s" />
          </el-select>
        </div>
      </template>
      <div v-if="loadingBars" class="qs-empty">加载中…</div>
      <div v-else-if="!bars.length" class="qs-empty">没有可显示的 K 线</div>
      <CandleChart v-else :bars="bars" :symbol="selectedSymbol" :adjust="adjust" />
    </el-card>
  </div>
</template>
