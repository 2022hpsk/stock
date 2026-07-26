<script setup lang="ts">
/**
 * P2 每日建议（docs/09 第三节）。
 *
 * 三块内容缺一不可：
 * - 建议列表（展开是四支柱）；
 * - **被风控否决的候选**——"为什么没买"和"为什么买了"同样重要；
 * - **组合层跳过的项**——常常是"资金不够买一手"这类实操问题，
 *   不显示的话用户会以为系统漏了标的。
 */
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'

import { api, ApiError } from '@/api/client'
import type { AdviceResponse, TradePlan } from '@/api/types'
import RationalePanel from '@/components/RationalePanel.vue'
import { useSystemStore } from '@/stores/system'

const system = useSystemStore()

const plan = ref<TradePlan | null>(null)
const scores = ref<{ base: Record<string, number>; final: Record<string, number> }>({
  base: {},
  final: {},
})
const skipped = ref<Array<{ symbol: string; reason: string }>>([])
const llmUsed = ref(false)
const dates = ref<string[]>([])
const selectedDate = ref('')
const loading = ref(false)
const generating = ref(false)
const errorMessage = ref('')

const buys = computed(() => plan.value?.intents.filter((i) => i.side === 'buy') ?? [])
const sells = computed(() => plan.value?.intents.filter((i) => i.side === 'sell') ?? [])

async function loadDates(): Promise<void> {
  const res = await api.get<{ dates: string[] }>('/api/advisor/dates')
  dates.value = res.dates.slice().reverse()
  if (!selectedDate.value && dates.value.length) selectedDate.value = dates.value[0]
}

async function loadPlan(): Promise<void> {
  if (!selectedDate.value) return
  loading.value = true
  errorMessage.value = ''
  try {
    plan.value = await api.get<TradePlan>(`/api/advisor/plan/${selectedDate.value}`)
    skipped.value = []
    scores.value = { base: {}, final: {} }
  } catch (e) {
    plan.value = null
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

async function generate(): Promise<void> {
  generating.value = true
  errorMessage.value = ''
  try {
    const res = await api.post<AdviceResponse>('/api/advisor/advise', { save: true })
    plan.value = res.plan
    scores.value = { base: res.base_scores, final: res.final_scores }
    skipped.value = res.skipped
    llmUsed.value = res.llm_used
    selectedDate.value = res.plan.trade_date
    await loadDates()
    ElMessage.success(res.summary)
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    generating.value = false
  }
}

onMounted(async () => {
  await loadDates()
  await loadPlan()
})
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>每日建议</h2>
        <div class="qs-sub">
          每条建议都附四支柱解释；被否决与被跳过的候选一并展示——只看"买什么"会漏掉一半信息。
        </div>
      </div>
      <div style="display: flex; gap: 8px; align-items: center">
        <el-select v-model="selectedDate" placeholder="选择日期" style="width: 150px" @change="loadPlan">
          <el-option v-for="d in dates" :key="d" :label="d" :value="d" />
        </el-select>
        <el-button
          type="primary"
          :loading="generating"
          :disabled="system.status?.readonly"
          @click="generate"
        >
          生成今日建议
        </el-button>
      </div>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" />

    <template v-if="plan">
      <div class="qs-cards">
        <div class="qs-card">
          <div class="qs-label">建议条数</div>
          <div class="qs-value">{{ plan.intents.length }}</div>
        </div>
        <div class="qs-card">
          <div class="qs-label">买入金额</div>
          <div class="qs-value qs-up qs-mono">{{ plan.total_buy_amount ?? '—' }}</div>
        </div>
        <div class="qs-card">
          <div class="qs-label">卖出金额</div>
          <div class="qs-value qs-down qs-mono">{{ plan.total_sell_amount ?? '—' }}</div>
        </div>
        <div class="qs-card">
          <div class="qs-label">风控状态</div>
          <div class="qs-value">{{ plan.circuit_state }}</div>
        </div>
      </div>

      <el-card shadow="never" style="margin-bottom: 16px">
        <template #header>
          <div style="display: flex; justify-content: space-between; align-items: center">
            <span>建议明细（{{ buys.length }} 买 / {{ sells.length }} 卖）</span>
            <el-tag v-if="llmUsed" type="warning" size="small">🤖 本次有大模型参与</el-tag>
          </div>
        </template>

        <div v-if="!plan.intents.length" class="qs-empty">今日无交易建议</div>
        <el-collapse v-else>
          <el-collapse-item v-for="intent in plan.intents" :key="intent.intent_id" :name="intent.intent_id">
            <template #title>
              <div class="qs-intent-title">
                <el-tag :type="intent.side === 'buy' ? 'danger' : 'success'" size="small">
                  {{ intent.side === 'buy' ? '买入' : '卖出' }}
                </el-tag>
                <strong class="qs-mono">{{ intent.symbol }}</strong>
                <span class="qs-mono">{{ intent.qty }} 股</span>
                <span class="qs-mono">限价 {{ intent.price_low }} ~ {{ intent.price_high }}</span>
                <span class="qs-mono">约 {{ intent.estimated_amount }} 元</span>
                <el-tag size="small" type="info">{{ intent.urgency }}</el-tag>
                <el-tag v-if="intent.rationale.llm_involved" size="small" type="warning">🤖</el-tag>
                <span class="qs-verdict-brief">{{ intent.rationale.verdict }}</span>
              </div>
            </template>
            <RationalePanel
              :rationale="intent.rationale"
              :base-score="scores.base[intent.symbol]"
              :final-score="scores.final[intent.symbol]"
            />
          </el-collapse-item>
        </el-collapse>
      </el-card>

      <el-row :gutter="16">
        <el-col :span="12">
          <el-card shadow="never">
            <template #header>被风控否决的候选（{{ plan.rejected.length }}）</template>
            <div v-if="!plan.rejected.length" class="qs-empty">无</div>
            <el-table v-else :data="plan.rejected" size="small">
              <el-table-column prop="symbol" label="标的" width="120" />
              <el-table-column prop="rule_id" label="规则" width="110" />
              <el-table-column prop="reason" label="原因" show-overflow-tooltip />
            </el-table>
          </el-card>
        </el-col>
        <el-col :span="12">
          <el-card shadow="never">
            <template #header>
              组合层跳过（{{ skipped.length }}）与解释不完整剔除（{{ plan.incomplete.length }}）
            </template>
            <div v-if="!skipped.length && !plan.incomplete.length" class="qs-empty">无</div>
            <template v-else>
              <el-table v-if="skipped.length" :data="skipped" size="small">
                <el-table-column prop="symbol" label="标的" width="120" />
                <el-table-column prop="reason" label="跳过原因" show-overflow-tooltip />
              </el-table>
              <el-table v-if="plan.incomplete.length" :data="plan.incomplete" size="small">
                <el-table-column prop="symbol" label="标的" width="120" />
                <el-table-column prop="missing" label="缺失支柱" show-overflow-tooltip />
              </el-table>
            </template>
          </el-card>
        </el-col>
      </el-row>

      <el-card shadow="never" style="margin-top: 16px">
        <template #header>可追溯性（红线 R6）</template>
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="计划 ID">
            <span class="qs-mono">{{ plan.plan_id }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="生成时间">{{ plan.generated_at }}</el-descriptions-item>
          <el-descriptions-item label="数据指纹">
            <span class="qs-mono">{{ plan.data_fingerprint || '—' }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="参数哈希">
            <span class="qs-mono">{{ plan.param_hash || '—' }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="策略版本" :span="2">
            <span class="qs-mono">
              {{ Object.entries(plan.strategy_versions).map(([k, v]) => `${k}@${v}`).join('  ') || '—' }}
            </span>
          </el-descriptions-item>
        </el-descriptions>
      </el-card>
    </template>

    <div v-else-if="!loading && !errorMessage" class="qs-empty">
      还没有交易计划。点右上角「生成今日建议」——需要数据湖里已有行情。
    </div>
  </div>
</template>

<style scoped>
.qs-intent-title {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  font-size: 13px;
}
.qs-verdict-brief {
  color: var(--qs-muted);
  font-size: 12px;
}
</style>
