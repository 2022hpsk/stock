<script setup lang="ts">
/**
 * P12 复盘（docs/09 第三节、docs/08 D3）。
 *
 * 这一页回答的问题只有一个：**你的人工干预到底是帮忙还是添乱。**
 *
 * 所以界面上必须一直显示两个"扫兴"的数字：
 *
 * - `sample_count` / `has_enough_samples`：样本不够时不给结论。
 *   三次跳过里对了两次，胜率 67%，图画出来很好看，但那是噪声；
 * - `unpriced_skips`：因缺事后行情没能纳入统计的跳过数。
 *   不显示的话，用户会以为统计覆盖了全部干预。
 */
import { computed, onMounted, ref } from 'vue'

import { api, ApiError } from '@/api/client'

interface Deviation {
  trade_date: string
  planned: number
  executed: number
  skipped: number
  aborted: boolean
  execution_rate: number
  planned_amount: string
  executed_amount: string
  amount_drift: string
  needs_attention: boolean
  by_reason: Record<string, number>
  explain: string
}

interface Intervention {
  reason: string
  count: number
  win_rate: number
  mean_forgone_return: number
  total_forgone: string
  has_enough_samples: boolean
  verdict: string
  explain: string
}

interface Summary {
  start: string
  end: string
  plans: number
  total_planned: number
  total_executed: number
  total_skipped: number
  execution_rate: number
  sample_count: number
  has_enough_samples: boolean
  unpriced_skips: number
  explain: string
  deviations: Deviation[]
  interventions: Intervention[]
}

const summary = ref<Summary | null>(null)
const horizon = ref(20)
const errorMessage = ref('')
const loading = ref(false)

const reasonLabels: Record<string, string> = {
  disagree_logic: '不认同策略逻辑',
  cash_reserved: '资金另有安排',
  bad_timing: '认为时机不对',
  other_info: '已有其他渠道信息',
  other: '其他',
  unknown: '未记录原因',
}

const hasData = computed(() => (summary.value?.deviations.length ?? 0) > 0)

async function load(): Promise<void> {
  loading.value = true
  errorMessage.value = ''
  try {
    summary.value = await api.get<Summary>('/api/review/summary', { horizon_days: horizon.value })
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>复盘</h2>
        <div class="qs-sub">
          按跳过原因分组，用事后实际价格衡量人工干预的价值：某类跳过长期跑赢程序，
          说明策略在这个维度上有系统性缺陷；长期跑输，说明该更信任程序。
        </div>
      </div>
      <div style="display: flex; gap: 8px; align-items: center">
        <span class="qs-sub">事后持有期</span>
        <el-input-number v-model="horizon" :min="1" :max="120" style="width: 120px" @change="load" />
        <el-button :loading="loading" @click="load">刷新</el-button>
      </div>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" style="margin-bottom: 12px" />

    <el-alert
      v-if="summary && !summary.has_enough_samples"
      type="warning"
      show-icon
      :closable="false"
      title="样本不足，暂不给出「该更信任谁」的结论"
      :description="`已评估干预 ${summary.sample_count} 次。样本太少时算出的胜率是噪声，但读起来很像信号——所以这里不画结论。`"
      style="margin-bottom: 12px"
    />
    <el-alert
      v-if="summary && summary.unpriced_skips > 0"
      type="info"
      show-icon
      :closable="false"
      :title="`${summary.unpriced_skips} 笔跳过因缺少事后行情未纳入统计`"
      description="用 0 收益填充会把统计悄悄拉向「干预没有影响」这个错误结论，所以宁可排除。补齐这段行情后重新查看即可。"
      style="margin-bottom: 12px"
    />

    <div class="qs-cards">
      <div class="qs-card">
        <div class="qs-label">计划份数</div>
        <div class="qs-value">{{ summary?.plans ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">计划 / 执行</div>
        <div class="qs-value">
          {{ summary?.total_planned ?? '—' }} / {{ summary?.total_executed ?? '—' }}
        </div>
      </div>
      <div class="qs-card">
        <div class="qs-label">执行率</div>
        <div class="qs-value">
          {{ summary ? (summary.execution_rate * 100).toFixed(0) + '%' : '—' }}
        </div>
      </div>
      <div class="qs-card">
        <div class="qs-label">跳过笔数</div>
        <div class="qs-value">{{ summary?.total_skipped ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">已评估样本</div>
        <div class="qs-value">{{ summary?.sample_count ?? '—' }}</div>
      </div>
    </div>

    <el-card shadow="never" style="margin-bottom: 16px">
      <template #header>人工干预价值（按跳过原因分组）</template>
      <div v-if="!summary?.interventions.length" class="qs-empty">
        还没有可评估的干预样本。跳过一些建议并等待事后行情补齐后，这里会给出结论。
      </div>
      <el-table v-else :data="summary.interventions" size="small">
        <el-table-column label="跳过原因" width="160">
          <template #default="{ row }">{{ reasonLabels[row.reason] ?? row.reason }}</template>
        </el-table-column>
        <el-table-column prop="count" label="次数" width="80" />
        <el-table-column label="跳对率" width="100">
          <template #default="{ row }">
            <span class="qs-mono">{{ (row.win_rate * 100).toFixed(0) }}%</span>
          </template>
        </el-table-column>
        <el-table-column label="平均错过收益" width="130">
          <template #default="{ row }">
            <span class="qs-mono" :class="row.mean_forgone_return > 0 ? 'qs-up' : 'qs-down'">
              {{ (row.mean_forgone_return * 100).toFixed(2) }}%
            </span>
          </template>
        </el-table-column>
        <el-table-column label="结论" min-width="260">
          <template #default="{ row }">
            <template v-if="row.has_enough_samples">
              <el-tag size="small" :type="row.win_rate >= 0.5 ? 'success' : 'danger'">
                {{ row.verdict }}
              </el-tag>
              <span class="qs-sub" style="margin-left: 8px">{{ row.explain }}</span>
            </template>
            <span v-else class="qs-flat">样本不足，不给结论</span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card shadow="never">
      <template #header>计划-实际偏差（逐日）</template>
      <div v-if="!hasData" class="qs-empty">
        区间内没有执行记录。执行过交易计划后，这里会显示计划与实际的差异。
      </div>
      <el-table v-else :data="summary!.deviations" size="small">
        <el-table-column prop="trade_date" label="交易日" width="120" />
        <el-table-column prop="planned" label="计划" width="80" />
        <el-table-column prop="executed" label="执行" width="80" />
        <el-table-column prop="skipped" label="跳过" width="80" />
        <el-table-column label="执行率" width="90">
          <template #default="{ row }">
            <span class="qs-mono">{{ (row.execution_rate * 100).toFixed(0) }}%</span>
          </template>
        </el-table-column>
        <el-table-column label="金额偏差" width="120">
          <template #default="{ row }">
            <span class="qs-mono">{{ row.amount_drift }}</span>
          </template>
        </el-table-column>
        <el-table-column label="跳过原因分布" min-width="240">
          <template #default="{ row }">
            <el-tag
              v-for="(n, reason) in row.by_reason"
              :key="reason"
              size="small"
              style="margin-right: 6px"
            >
              {{ reasonLabels[String(reason)] ?? reason }} × {{ n }}
            </el-tag>
            <span v-if="!Object.keys(row.by_reason).length" class="qs-flat">无</span>
          </template>
        </el-table-column>
        <el-table-column label="需关注" width="90">
          <template #default="{ row }">
            <el-tag v-if="row.needs_attention" type="warning" size="small">是</el-tag>
            <span v-else class="qs-flat">否</span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>
