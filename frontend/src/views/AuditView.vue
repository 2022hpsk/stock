<script setup lang="ts">
/**
 * P15 审计（docs/09 第三节）。
 *
 * 红线 R6 要求每条建议可追溯可复现。把数据指纹、策略版本、参数哈希
 * 存下来只证明"当时算过"；**重算一遍并比对**才证明"现在还能算出同样的结果"。
 * 这一页的「复现」按钮就是那条红线唯一的证据。
 *
 * 复现结果分三种，界面必须区分：完全一致 / 输入变了（不可复现）/
 * 输入没变但结果变了（漂移，需排查）。把后两种混成一个"失败"，
 * 会让人分不清是数据被回补了还是代码出了问题。
 */
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'

import { api, ApiError } from '@/api/client'

interface Chain {
  intent_id: string
  symbol: string
  side: string
  suggested_qty: number
  orders: Array<Record<string, unknown>>
  fills: Array<Record<string, unknown>>
  outcome: string
}

interface AuditPlan {
  plan: {
    plan_id: string
    trade_date: string
    generated_at: string
    data_fingerprint: string
    param_hash: string
    strategy_versions: Record<string, string>
    confirmed_by: string
    confirmed_at: string | null
    is_confirmed: boolean
    intents: unknown[]
  }
  executions: Array<{
    executed_at: string
    broker: string
    confirmed_by: string
    aborted: boolean
    abort_reason: string
    orders: number
    fills: number
  }>
  chain: Chain[]
}

interface Reproduction {
  verdict: string
  explain: string
  fingerprint: { archived: string; fresh: string; match: boolean }
  param_hash: { archived: string; fresh: string; match: boolean }
  intents: { match: boolean; only_archived: string[]; only_fresh: string[] }
}

const planDates = ref<string[]>([])
const orphans = ref<string[]>([])
const selected = ref('')
const detail = ref<AuditPlan | null>(null)
const reproduction = ref<Reproduction | null>(null)
const busy = ref(false)
const errorMessage = ref('')

const verdictType: Record<string, string> = {
  identical: 'success',
  drifted: 'warning',
  unreproducible: 'info',
}

async function loadDates(): Promise<void> {
  const res = await api.get<{ plan_dates: string[]; orphan_executions: string[] }>('/api/audit/dates')
  planDates.value = res.plan_dates.slice().reverse()
  orphans.value = res.orphan_executions
  if (!selected.value && planDates.value.length) selected.value = planDates.value[0]
}

async function loadDetail(): Promise<void> {
  if (!selected.value) return
  errorMessage.value = ''
  reproduction.value = null
  try {
    detail.value = await api.get<AuditPlan>(`/api/audit/plan/${selected.value}`)
  } catch (e) {
    detail.value = null
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  }
}

async function reproduce(): Promise<void> {
  if (!selected.value) return
  busy.value = true
  errorMessage.value = ''
  try {
    reproduction.value = await api.post<Reproduction>(`/api/audit/reproduce/${selected.value}`)
    ElMessage.info(reproduction.value.explain)
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

onMounted(async () => {
  await loadDates()
  await loadDetail()
})
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>审计</h2>
        <div class="qs-sub">
          任一天的建议可完整复现（红线 R6）。存下指纹只证明「当时算过」，
          重算并比对才证明「现在还能算出同样的结果」。
        </div>
      </div>
      <div style="display: flex; gap: 8px; align-items: center">
        <el-select v-model="selected" style="width: 150px" @change="loadDetail">
          <el-option v-for="d in planDates" :key="d" :label="d" :value="d" />
        </el-select>
        <el-button type="primary" :loading="busy" :disabled="!selected" @click="reproduce">
          用当时的参数重新计算
        </el-button>
      </div>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" style="margin-bottom: 12px" />
    <el-alert
      v-if="orphans.length"
      type="warning"
      show-icon
      :closable="false"
      :title="`${orphans.length} 天有执行记录却没有对应的交易计划`"
      :description="`涉及 ${orphans.join('、')}。这意味着有委托没经过计划就发出去了——红线 R5 要求下单必须来自计划，值得逐一核对。`"
      style="margin-bottom: 12px"
    />

    <el-card v-if="reproduction" shadow="never" style="margin-bottom: 16px">
      <template #header>复现结果</template>
      <el-alert
        :type="verdictType[reproduction.verdict] ?? 'info'"
        show-icon
        :closable="false"
        :title="reproduction.verdict"
        :description="reproduction.explain"
        style="margin-bottom: 12px"
      />
      <el-descriptions :column="1" border size="small">
        <el-descriptions-item label="数据指纹">
          <el-tag :type="reproduction.fingerprint.match ? 'success' : 'danger'" size="small">
            {{ reproduction.fingerprint.match ? '一致' : '不一致' }}
          </el-tag>
          <span class="qs-mono" style="margin-left: 8px">
            存档 {{ reproduction.fingerprint.archived.slice(0, 24) }} ／ 重算
            {{ reproduction.fingerprint.fresh.slice(0, 24) }}
          </span>
        </el-descriptions-item>
        <el-descriptions-item label="参数哈希">
          <el-tag :type="reproduction.param_hash.match ? 'success' : 'danger'" size="small">
            {{ reproduction.param_hash.match ? '一致' : '不一致' }}
          </el-tag>
          <span class="qs-mono" style="margin-left: 8px">
            存档 {{ reproduction.param_hash.archived.slice(0, 24) }} ／ 重算
            {{ reproduction.param_hash.fresh.slice(0, 24) }}
          </span>
        </el-descriptions-item>
        <el-descriptions-item label="建议差异">
          <template v-if="reproduction.intents.match">完全一致</template>
          <template v-else>
            <div v-if="reproduction.intents.only_archived.length">
              仅存档有：<span class="qs-mono">{{ reproduction.intents.only_archived.join('，') }}</span>
            </div>
            <div v-if="reproduction.intents.only_fresh.length">
              仅重算有：<span class="qs-mono">{{ reproduction.intents.only_fresh.join('，') }}</span>
            </div>
          </template>
        </el-descriptions-item>
      </el-descriptions>
    </el-card>

    <template v-if="detail">
      <el-card shadow="never" style="margin-bottom: 16px">
        <template #header>计划快照</template>
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="计划 ID">
            <span class="qs-mono">{{ detail.plan.plan_id }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="生成时间">{{ detail.plan.generated_at }}</el-descriptions-item>
          <el-descriptions-item label="数据指纹">
            <span class="qs-mono">{{ detail.plan.data_fingerprint || '—' }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="参数哈希">
            <span class="qs-mono">{{ detail.plan.param_hash || '—' }}</span>
          </el-descriptions-item>
          <el-descriptions-item label="策略版本" :span="2">
            <span class="qs-mono">
              {{ Object.entries(detail.plan.strategy_versions).map(([k, v]) => `${k}@${v}`).join('  ') || '—' }}
            </span>
          </el-descriptions-item>
          <el-descriptions-item label="人工确认" :span="2">
            <el-tag :type="detail.plan.is_confirmed ? 'success' : 'info'" size="small">
              {{ detail.plan.is_confirmed ? `${detail.plan.confirmed_by} 于 ${detail.plan.confirmed_at}` : '未确认' }}
            </el-tag>
          </el-descriptions-item>
        </el-descriptions>
      </el-card>

      <el-card shadow="never" style="margin-bottom: 16px">
        <template #header>执行记录（{{ detail.executions.length }}）</template>
        <div v-if="!detail.executions.length" class="qs-empty">当日没有执行记录</div>
        <el-table v-else :data="detail.executions" size="small">
          <el-table-column prop="executed_at" label="执行时间" width="230" />
          <el-table-column prop="broker" label="通道" width="100" />
          <el-table-column prop="confirmed_by" label="确认人" width="100" />
          <el-table-column prop="orders" label="订单" width="80" />
          <el-table-column prop="fills" label="成交" width="80" />
          <el-table-column label="结果" min-width="240">
            <template #default="{ row }">
              <el-tag v-if="row.aborted" type="danger" size="small">已中止</el-tag>
              <span v-if="row.aborted" class="qs-sub" style="margin-left: 8px">{{ row.abort_reason }}</span>
              <el-tag v-else type="success" size="small">正常</el-tag>
            </template>
          </el-table-column>
        </el-table>
      </el-card>

      <el-card shadow="never">
        <template #header>建议 → 确认 → 提交 → 成交 链路（{{ detail.chain.length }}）</template>
        <el-table :data="detail.chain" size="small">
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
          <el-table-column prop="suggested_qty" label="建议数量" width="100" />
          <el-table-column label="订单 / 成交" width="120">
            <template #default="{ row }">{{ row.orders.length }} / {{ row.fills.length }}</template>
          </el-table-column>
          <el-table-column prop="outcome" label="最终去向" min-width="260" />
          <el-table-column label="intent_id" min-width="220">
            <template #default="{ row }">
              <span class="qs-mono qs-flat">{{ row.intent_id }}</span>
            </template>
          </el-table-column>
        </el-table>
      </el-card>
    </template>

    <div v-else-if="!errorMessage" class="qs-empty">还没有可审计的交易计划</div>
  </div>
</template>
