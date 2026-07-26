<script setup lang="ts">
/**
 * P9 组合（docs/09 第三节）：当前权重 vs 目标权重、约束满足情况。
 *
 * 目标权重取自**已保存的那份交易计划**，而不是现场重算。那份建议已经过了
 * 风控与解释完整性检查；现场重算会得到一个没经过那些关卡的数字，
 * 两个页面显示不同的目标值时，用户不知道该信哪个。
 */
import { computed, onMounted, ref } from 'vue'

import { api, ApiError } from '@/api/client'

interface Row {
  symbol: string
  current_weight: number
  target_weight: number
  current_qty: number
  target_qty: number
  delta_qty: number
  weight_drift: number
  priced: boolean
}

interface Weights {
  trade_date: string | null
  plan_id: string | null
  total_value: string
  cash: string
  cash_weight: number
  rows: Row[]
  unpriced_symbols: string[]
  is_empty: boolean
}

interface Breach {
  rule_id: string
  symbol: string
  actual: number
  limit: number
  message: string
}

interface Constraints {
  satisfied: boolean
  breaches: Breach[]
  limits: { max_single_position: number; max_holdings: number; min_cash_ratio: number }
  current: { holdings: number; cash_ratio: number; total_value: string }
}

const weights = ref<Weights | null>(null)
const constraints = ref<Constraints | null>(null)
const errorMessage = ref('')

const hasRows = computed(() => (weights.value?.rows.length ?? 0) > 0)

function pct(v: number): string {
  return `${(v * 100).toFixed(2)}%`
}

onMounted(async () => {
  try {
    const [w, c] = await Promise.all([
      api.get<Weights>('/api/portfolio/weights'),
      api.get<Constraints>('/api/portfolio/constraints'),
    ])
    weights.value = w
    constraints.value = c
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  }
})
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>组合</h2>
        <div class="qs-sub">
          目标权重来自已保存的交易计划——那份建议过了风控与解释完整性检查，
          现场重算的数字没有。
        </div>
      </div>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" style="margin-bottom: 12px" />
    <el-alert
      v-if="weights?.is_empty"
      type="info"
      show-icon
      :closable="false"
      title="账本为空，当前权重全部为 0"
      description="到账户页录入持仓后，这里才能显示真实的当前 vs 目标对比。"
      style="margin-bottom: 12px"
    />
    <el-alert
      v-if="weights?.unpriced_symbols.length"
      type="warning"
      show-icon
      :closable="false"
      :title="`${weights.unpriced_symbols.join('、')} 缺现价，权重按 0 计`"
      description="它们会在权重图上凭空消失。请先在数据页更新这些标的的行情。"
      style="margin-bottom: 12px"
    />

    <div class="qs-cards">
      <div class="qs-card">
        <div class="qs-label">总资产</div>
        <div class="qs-value qs-mono">{{ weights?.total_value ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">现金 / 占比</div>
        <div class="qs-value qs-mono" style="font-size: 18px">
          {{ weights?.cash ?? '—' }}（{{ weights ? pct(weights.cash_weight) : '—' }}）
        </div>
      </div>
      <div class="qs-card">
        <div class="qs-label">持仓只数 / 上限</div>
        <div class="qs-value">
          {{ constraints?.current.holdings ?? '—' }} / {{ constraints?.limits.max_holdings ?? '—' }}
        </div>
      </div>
      <div class="qs-card">
        <div class="qs-label">约束</div>
        <div class="qs-value" :class="constraints?.satisfied ? 'qs-down' : 'qs-up'">
          {{ constraints === null ? '—' : constraints.satisfied ? '全部满足' : `${constraints.breaches.length} 项超限` }}
        </div>
      </div>
    </div>

    <el-card v-if="constraints && !constraints.satisfied" shadow="never" style="margin-bottom: 16px">
      <template #header>超限项</template>
      <el-alert
        v-for="(b, i) in constraints.breaches"
        :key="i"
        type="warning"
        show-icon
        :closable="false"
        :title="`[${b.rule_id}] ${b.message}`"
        style="margin-bottom: 8px"
      />
      <p class="qs-sub" style="margin-bottom: 0">
        逐条列出而不是只给一个「不合规」的总判定——知道是哪一条、超了多少，才知道该卖什么。
      </p>
    </el-card>

    <el-card shadow="never">
      <template #header>
        当前 vs 目标权重
        <span class="qs-sub" v-if="weights?.trade_date">（依据 {{ weights.trade_date }} 的计划）</span>
      </template>
      <div v-if="!hasRows" class="qs-empty">还没有持仓，也没有交易计划</div>
      <el-table v-else :data="weights!.rows" size="small">
        <el-table-column label="标的" width="120">
          <template #default="{ row }">
            <strong class="qs-mono">{{ row.symbol }}</strong>
            <el-tag v-if="!row.priced" size="small" type="info" style="margin-left: 6px">缺价</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="当前权重" width="110">
          <template #default="{ row }">
            <span class="qs-mono">{{ pct(row.current_weight) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="目标权重" width="110">
          <template #default="{ row }">
            <span class="qs-mono">{{ pct(row.target_weight) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="偏离" width="200">
          <template #default="{ row }">
            <div class="qs-drift">
              <span class="qs-mono" :class="row.weight_drift > 0 ? 'qs-up' : row.weight_drift < 0 ? 'qs-down' : 'qs-flat'">
                {{ row.weight_drift >= 0 ? '+' : '' }}{{ pct(row.weight_drift) }}
              </span>
              <el-progress
                :percentage="Math.min(100, Math.abs(row.weight_drift) * 500)"
                :show-text="false"
                :stroke-width="6"
                :color="row.weight_drift > 0 ? '#e5484d' : '#30a46c'"
                style="flex: 1"
              />
            </div>
          </template>
        </el-table-column>
        <el-table-column label="当前 → 目标（股）" min-width="160">
          <template #default="{ row }">
            <span class="qs-mono">{{ row.current_qty }} → {{ row.target_qty }}</span>
            <el-tag
              v-if="row.delta_qty"
              size="small"
              :type="row.delta_qty > 0 ? 'danger' : 'success'"
              style="margin-left: 8px"
            >
              {{ row.delta_qty > 0 ? '买入' : '卖出' }} {{ Math.abs(row.delta_qty) }}
            </el-tag>
          </template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<style scoped>
.qs-drift {
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
