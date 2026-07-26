<script setup lang="ts">
/**
 * P3 执行（docs/09 第三节）——**界面上唯一能动真钱的页面**。
 *
 * 交互上的几条硬约束，都是为了让"误操作"变难：
 *
 * - 默认全部**不勾选**。默认全选会把"没来得及看"变成"下单了"，
 *   而这个方向的错误不可逆；
 * - 跳过必须从下拉框里选原因，不能自由输入。复盘要按原因分组统计
 *   人工干预到底是帮忙还是添乱（docs/08 D3）；
 * - 真实通道要勾"我已核对"**并**输入确认码，两者都在后端再校验一次；
 * - 急停时整页禁用。
 */
import { computed, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'

import { api, ApiError } from '@/api/client'
import type { ExecutionPreview, TradePlan } from '@/api/types'
import { useSystemStore } from '@/stores/system'

const system = useSystemStore()

const dates = ref<string[]>([])
const selectedDate = ref('')
const plan = ref<TradePlan | null>(null)
const preview = ref<ExecutionPreview | null>(null)
const skipReasons = ref<Array<{ value: string; label: string }>>([])
const brokerInfo = ref<{ broker: string; bridge_dir: string | null } | null>(null)

const decisions = ref<Record<string, { accepted: boolean; adjusted_qty: number | null; skip_reason: string; skip_note: string }>>({})
const prices = ref<Record<string, string>>({})
const live = ref(false)
const acknowledged = ref(false)
const confirmationCode = ref('')
const busy = ref(false)
const report = ref<Record<string, unknown> | null>(null)
const errorMessage = ref('')

const halted = computed(() => system.status?.halt.halted ?? false)
const readonly = computed(() => system.status?.readonly ?? false)
const disabled = computed(() => halted.value || readonly.value)
const acceptedCount = computed(() => Object.values(decisions.value).filter((d) => d.accepted).length)

const canSubmit = computed(() => {
  if (disabled.value || !plan.value) return false
  if (live.value && (!acknowledged.value || !confirmationCode.value.trim())) return false
  // 跳过必须给原因，否则后端会拒
  return Object.values(decisions.value).every((d) => d.accepted || Boolean(d.skip_reason))
})

async function loadDates(): Promise<void> {
  const res = await api.get<{ dates: string[] }>('/api/advisor/dates')
  dates.value = res.dates.slice().reverse()
  if (!selectedDate.value && dates.value.length) selectedDate.value = dates.value[0]
}

async function loadPlan(): Promise<void> {
  if (!selectedDate.value) return
  errorMessage.value = ''
  report.value = null
  try {
    plan.value = await api.get<TradePlan>(`/api/advisor/plan/${selectedDate.value}`)
    decisions.value = Object.fromEntries(
      plan.value.intents.map((i) => [
        i.intent_id,
        // 默认不接受：必须逐条看过再勾
        { accepted: false, adjusted_qty: null, skip_reason: '', skip_note: '' },
      ]),
    )
    prices.value = Object.fromEntries(
      plan.value.intents.map((i) => [i.symbol, i.price_high ?? i.price_low ?? '0']),
    )
    await runPreview()
  } catch (e) {
    plan.value = null
    preview.value = null
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  }
}

async function runPreview(): Promise<void> {
  if (!plan.value) return
  try {
    preview.value = await api.post<ExecutionPreview>('/api/execution/preview', {
      trade_date: plan.value.trade_date,
      plan_id: plan.value.plan_id,
      prices: prices.value,
    })
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  }
}

async function submit(): Promise<void> {
  if (!plan.value) return
  const skippedCount = plan.value.intents.length - acceptedCount.value
  await ElMessageBox.confirm(
    `将提交 ${acceptedCount.value} 笔，跳过 ${skippedCount} 笔，通道 ${brokerInfo.value?.broker ?? '—'}${
      live.value ? '（真实资金）' : '（模拟）'
    }。确认执行？`,
    '提交前汇总确认',
    { type: live.value ? 'error' : 'warning', confirmButtonText: '确认执行' },
  )

  busy.value = true
  errorMessage.value = ''
  try {
    report.value = await api.post('/api/execution/execute', {
      trade_date: plan.value.trade_date,
      plan_id: plan.value.plan_id,
      prices: prices.value,
      live: live.value,
      confirmation_code: confirmationCode.value,
      confirmed_by: 'ui',
      decisions: Object.entries(decisions.value).map(([intent_id, d]) => ({
        intent_id,
        accepted: d.accepted,
        adjusted_qty: d.adjusted_qty,
        skip_reason: d.accepted ? null : d.skip_reason,
        skip_note: d.skip_note,
      })),
    })
    ElMessage.success('执行完成')
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

async function cancelAll(): Promise<void> {
  await ElMessageBox.confirm('撤销所有未成交委托？', '全部撤单', { type: 'warning' })
  const res = await api.post<{ cancelled: number }>('/api/execution/cancel-all')
  ElMessage.success(`已撤 ${res.cancelled} 笔`)
}

function previewOf(intentId: string) {
  return preview.value?.items.find((i) => i.intent_id === intentId) ?? null
}

onMounted(async () => {
  ;[brokerInfo.value, skipReasons.value] = await Promise.all([
    api.get<{ broker: string; bridge_dir: string | null }>('/api/execution/status'),
    api.get<{ reasons: Array<{ value: string; label: string }> }>('/api/execution/skip-reasons').then((r) => r.reasons),
  ])
  await loadDates()
  await loadPlan()
})
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>执行</h2>
        <div class="qs-sub">
          逐单确认。**默认全部不勾选**——需要你逐条看过再决定；跳过必须选原因，复盘要用。
        </div>
      </div>
      <div style="display: flex; gap: 8px; align-items: center">
        <el-select v-model="selectedDate" style="width: 150px" @change="loadPlan">
          <el-option v-for="d in dates" :key="d" :label="d" :value="d" />
        </el-select>
        <el-button :disabled="disabled" @click="cancelAll">全部撤单</el-button>
      </div>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" style="margin-bottom: 12px" />
    <el-alert
      v-if="preview?.halted"
      type="error"
      show-icon
      :closable="false"
      :title="`系统已急停：${preview.halt_reason}`"
      style="margin-bottom: 12px"
    />
    <el-alert
      v-if="preview && preview.review_count > 0"
      type="warning"
      show-icon
      :closable="false"
      :title="`${preview.review_count} 笔价格漂移超阈值，需要重新判断而非照单执行`"
      style="margin-bottom: 12px"
    />

    <div class="qs-cards">
      <div class="qs-card">
        <div class="qs-label">通道</div>
        <div class="qs-value">{{ brokerInfo?.broker ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">拟买入</div>
        <div class="qs-value qs-up qs-mono">{{ preview?.total_buy ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">拟卖出</div>
        <div class="qs-value qs-down qs-mono">{{ preview?.total_sell ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">已勾选 / 总数</div>
        <div class="qs-value">{{ acceptedCount }} / {{ plan?.intents.length ?? 0 }}</div>
      </div>
    </div>

    <el-card shadow="never">
      <template #header>逐单确认</template>
      <div v-if="!plan?.intents.length" class="qs-empty">该日没有待执行的建议</div>
      <el-table v-else :data="plan.intents" size="small" row-key="intent_id">
        <el-table-column label="执行" width="70">
          <template #default="{ row }">
            <el-checkbox v-model="decisions[row.intent_id].accepted" :disabled="disabled" />
          </template>
        </el-table-column>
        <el-table-column label="标的" width="120">
          <template #default="{ row }">
            <span class="qs-mono">{{ row.symbol }}</span>
          </template>
        </el-table-column>
        <el-table-column label="方向" width="70">
          <template #default="{ row }">
            <el-tag :type="row.side === 'buy' ? 'danger' : 'success'" size="small">
              {{ row.side === 'buy' ? '买' : '卖' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="数量" width="110">
          <template #default="{ row }">
            <el-input-number
              v-model="decisions[row.intent_id].adjusted_qty"
              :placeholder="String(row.qty)"
              :step="100"
              :min="0"
              size="small"
              controls-position="right"
              style="width: 100px"
              :disabled="disabled"
            />
          </template>
        </el-table-column>
        <el-table-column label="限价" width="100">
          <template #default="{ row }">
            <span class="qs-mono">{{ previewOf(row.intent_id)?.limit_price ?? '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="当前价" width="110">
          <template #default="{ row }">
            <el-input v-model="prices[row.symbol]" size="small" :disabled="disabled" @blur="runPreview" />
          </template>
        </el-table-column>
        <el-table-column label="漂移" width="110">
          <template #default="{ row }">
            <el-tag v-if="previewOf(row.intent_id)?.needs_review" type="warning" size="small">
              STALE {{ ((previewOf(row.intent_id)?.drift?.drift_pct ?? 0) * 100).toFixed(2) }}%
            </el-tag>
            <span v-else class="qs-flat">正常</span>
          </template>
        </el-table-column>
        <el-table-column label="金额" width="110">
          <template #default="{ row }">
            <span class="qs-mono">{{ previewOf(row.intent_id)?.estimated_amount ?? '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="跳过原因（跳过时必填）" min-width="230">
          <template #default="{ row }">
            <el-select
              v-model="decisions[row.intent_id].skip_reason"
              size="small"
              placeholder="选择原因"
              :disabled="disabled || decisions[row.intent_id].accepted"
              style="width: 100%"
            >
              <el-option v-for="r in skipReasons" :key="r.value" :label="r.label" :value="r.value" />
            </el-select>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card shadow="never" style="margin-top: 16px">
      <template #header>提交</template>
      <el-form label-width="120px">
        <el-form-item label="资金通道">
          <el-switch
            v-model="live"
            :disabled="disabled"
            active-text="真实资金"
            inactive-text="模拟"
          />
          <span class="qs-sub" style="margin-left: 12px">
            真实通道必须勾选核对项并输入确认码；后端会再校验一次，前端拦截只是防误点。
          </span>
        </el-form-item>
        <template v-if="live">
          <el-form-item label="核对确认">
            <el-checkbox v-model="acknowledged" :disabled="disabled">
              我已逐条核对标的、方向、数量与限价
            </el-checkbox>
          </el-form-item>
          <el-form-item label="确认码">
            <el-input v-model="confirmationCode" style="width: 240px" :disabled="disabled" />
          </el-form-item>
        </template>
        <el-form-item>
          <el-button type="primary" :loading="busy" :disabled="!canSubmit" @click="submit">
            执行 {{ acceptedCount }} 笔
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card v-if="report" shadow="never" style="margin-top: 16px">
      <template #header>执行报告</template>
      <pre class="qs-mono qs-report">{{ JSON.stringify(report, null, 2) }}</pre>
    </el-card>
  </div>
</template>

<style scoped>
.qs-report {
  max-height: 420px;
  overflow: auto;
  font-size: 12px;
  background: #f8f9fa;
  padding: 12px;
  border-radius: 6px;
}
</style>
