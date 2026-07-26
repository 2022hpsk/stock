<script setup lang="ts">
/**
 * P0 仪表盘（docs/09 第三节）。
 *
 * 这一页要在**三秒内**回答"现在能不能放心地按建议下单"。所以放的都是
 * 前置条件而不是收益数字：风控状态、数据是否就绪、情报源是否活着、
 * LLM 处于什么模式。收益好看但数据停在三天前，那些建议就不能用。
 */
import { onMounted, ref } from 'vue'

import { api } from '@/api/client'
import type { DataStatus, LlmStatus } from '@/api/types'
import { useEventStore } from '@/stores/events'
import { useSystemStore } from '@/stores/system'

const system = useSystemStore()
const events = useEventStore()

const data = ref<DataStatus | null>(null)
const intel = ref<{ sources: number; inbox_pending: number; latest_date: string | null; blacklisted: number; message: string } | null>(null)
const llm = ref<LlmStatus | null>(null)
const loading = ref(true)

onMounted(async () => {
  await system.refresh()
  const results = await Promise.allSettled([
    api.get<DataStatus>('/api/data/status'),
    api.get<typeof intel.value>('/api/intel/status'),
    api.get<LlmStatus>('/api/llm/status'),
  ])
  if (results[0].status === 'fulfilled') data.value = results[0].value
  if (results[1].status === 'fulfilled') intel.value = results[1].value
  if (results[2].status === 'fulfilled') llm.value = results[2].value
  loading.value = false
})
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>仪表盘</h2>
        <div class="qs-sub">先看前置条件是否齐备，再看建议——数据停摆时的建议不能用。</div>
      </div>
    </div>

    <div class="qs-cards">
      <div class="qs-card">
        <div class="qs-label">风控状态</div>
        <div class="qs-value" :class="system.status?.halt.halted ? 'qs-up' : 'qs-down'">
          {{ system.status?.halt.halted ? 'HALTED' : 'NORMAL' }}
        </div>
      </div>
      <div class="qs-card">
        <div class="qs-label">交易通道</div>
        <div class="qs-value">{{ system.status?.broker ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">数据湖标的</div>
        <div class="qs-value">{{ data?.symbols ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">最新行情日</div>
        <div class="qs-value qs-mono" style="font-size: 18px">{{ data?.latest_date ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">情报源</div>
        <div class="qs-value">{{ intel?.sources ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">情报黑名单</div>
        <div class="qs-value">{{ intel?.blacklisted ?? '—' }}</div>
      </div>
    </div>

    <el-row :gutter="16">
      <el-col :span="12">
        <el-card shadow="never">
          <template #header>组件健康</template>
          <el-table :data="system.status?.components ?? []" size="small">
            <el-table-column prop="name" label="组件" width="140" />
            <el-table-column label="状态" width="80">
              <template #default="{ row }">
                <el-tag :type="row.ok ? 'success' : 'danger'" size="small">
                  {{ row.ok ? '正常' : '异常' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="detail" label="说明" show-overflow-tooltip />
          </el-table>
        </el-card>

        <el-card shadow="never" style="margin-top: 16px">
          <template #header>数据与情报</template>
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item label="数据湖">{{ data?.message ?? '—' }}</el-descriptions-item>
            <el-descriptions-item label="情报">{{ intel?.message ?? '—' }}</el-descriptions-item>
            <el-descriptions-item label="大模型">{{ llm?.message ?? '—' }}</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card shadow="never">
          <template #header>
            <div style="display: flex; justify-content: space-between; align-items: center">
              <span>实时事件</span>
              <el-tag :type="events.connected ? 'success' : 'info'" size="small">
                {{ events.connected ? '已连接' : '未连接' }}
              </el-tag>
            </div>
          </template>
          <div v-if="!events.events.length" class="qs-empty">
            暂无事件。运行数据更新、生成建议或回测时，进度会实时推到这里。
          </div>
          <div v-else class="qs-feed">
            <div v-for="e in events.events.slice(0, 30)" :key="e.seq" class="qs-feed-item">
              <span class="qs-mono qs-flat">#{{ e.seq }}</span>
              <el-tag size="small" type="info">{{ e.channel }}</el-tag>
              <strong>{{ e.kind }}</strong>
              <span class="qs-mono qs-feed-payload">{{ JSON.stringify(e.payload) }}</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.qs-feed {
  max-height: 480px;
  overflow: auto;
}
.qs-feed-item {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid var(--qs-border);
  font-size: 12px;
}
.qs-feed-payload {
  color: var(--qs-muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>
