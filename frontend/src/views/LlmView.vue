<script setup lang="ts">
/**
 * P16 大模型（docs/09 第三节）。
 *
 * 这一页显示的都是**约束**而不是能力：α 的当前值与硬上限、花了多少钱、
 * 缓存有多少、降级了没有。因为 LLM 在本系统里的定位就是"有界的增强项"——
 * 它不能直接决定下单方向、数量、价格（红线 LR1），影响力被 α 限幅
 * 且可一键关闭（红线 LR2）。
 *
 * 所以这里也**没有任何通往执行页的入口**。
 */
import { onMounted, ref } from 'vue'

import { api } from '@/api/client'
import type { LlmStatus } from '@/api/types'

const status = ref<LlmStatus | null>(null)
const ALPHA_HARD_CAP = 0.2

onMounted(async () => {
  status.value = await api.get<LlmStatus>('/api/llm/status')
})
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>大模型</h2>
        <div class="qs-sub">
          大模型只做文本理解与归纳，不做价格预测；对打分的影响受 α 限幅且可一键关闭，
          关闭后系统功能完整。
        </div>
      </div>
    </div>

    <el-alert
      v-if="status && !status.enabled"
      type="info"
      show-icon
      :closable="false"
      title="大模型已关闭，系统运行在纯量化模式（功能完整）"
      description="打开需要在配置页启用，并在环境变量里设置 ANTHROPIC_API_KEY——密钥绝不写进配置文件。"
      style="margin-bottom: 12px"
    />
    <el-alert
      v-if="status?.degraded"
      type="warning"
      show-icon
      :closable="false"
      :title="`已降级：${status.degraded_reason}`"
      description="降级后本次调用不使用大模型，系统继续走纯量化路径。"
      style="margin-bottom: 12px"
    />

    <div class="qs-cards">
      <div class="qs-card">
        <div class="qs-label">状态</div>
        <div class="qs-value">{{ status?.enabled ? status.mode : '关闭' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">α 影响系数（硬上限 {{ ALPHA_HARD_CAP }}）</div>
        <div class="qs-value qs-mono" :class="(status?.alpha ?? 0) > ALPHA_HARD_CAP ? 'qs-up' : ''">
          {{ status?.alpha?.toFixed(3) ?? '—' }}
        </div>
      </div>
      <div class="qs-card">
        <div class="qs-label">提示词版本</div>
        <div class="qs-value qs-mono" style="font-size: 18px">{{ status?.prompt_version ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">缓存条目</div>
        <div class="qs-value">{{ status?.cached_entries ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">当日花费</div>
        <div class="qs-value qs-mono">${{ status?.daily_spent_usd?.toFixed(4) ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">当月花费</div>
        <div class="qs-value qs-mono">${{ status?.monthly_spent_usd?.toFixed(4) ?? '—' }}</div>
      </div>
    </div>

    <el-row :gutter="16">
      <el-col :span="12">
        <el-card shadow="never">
          <template #header>各任务</template>
          <el-table :data="status?.tasks ?? []" size="small">
            <el-table-column prop="name" label="任务" width="150" />
            <el-table-column label="状态" width="90">
              <template #default="{ row }">
                <el-tag :type="row.enabled ? 'success' : 'info'" size="small">
                  {{ row.enabled ? '启用' : '停用' }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="model" label="模型" show-overflow-tooltip />
          </el-table>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card shadow="never">
          <template #header>可追溯性</template>
          <p class="qs-sub" style="margin-top: 0">
            提示词版本与模型 ID 都进 <code>param_hash</code>——**改提示词等同于改策略**，
            不记进去的话，同样的参数会给出不同的建议而无从追查。
          </p>
          <el-descriptions :column="1" border size="small">
            <el-descriptions-item
              v-for="(v, k) in status?.param_hash_parts ?? {}"
              :key="k"
              :label="String(k)"
            >
              <span class="qs-mono">{{ v }}</span>
            </el-descriptions-item>
            <el-descriptions-item label="缓存目录">
              <span class="qs-mono">{{ status?.cache_dir ?? '—' }}</span>
            </el-descriptions-item>
          </el-descriptions>
          <el-alert
            type="info"
            show-icon
            :closable="false"
            title="回测强制走 replay"
            description="回测前必须先 llm backfill 预计算缓存；回测中的实时调用会直接抛异常，避免用今天的模型去解释昨天的行情。"
            style="margin-top: 12px"
          />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>
