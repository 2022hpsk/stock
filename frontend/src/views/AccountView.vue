<script setup lang="ts">
/**
 * P1 账户（docs/09 第三节、docs/11-持仓账本规格.md）。
 *
 * 持仓表可展开到**批次明细**。这不是炫技：红利税按持股期限分三档
 * （≤1 月 20% / 1 月-1 年 10% / >1 年 免），"再持有 N 天可免税"的倒计时，
 * 全都要知道每一份股票是哪天买的。只看平均成本算不出这些，
 * 而对高股息标的，差几天卖掉可能就是几千块钱的事。
 *
 * 界面上**没有"修改持仓"的按钮**，只有"录一笔流水"（红线 R8）。
 * 持仓是重放出来的结果，不是可以直接编辑的状态。
 */
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'

import { api, ApiError } from '@/api/client'
import { useSystemStore } from '@/stores/system'

interface Lot {
  lot_id: string
  open_date: string
  original_qty: number
  remaining_qty: number
  cost_price: string
  accrued_dividend: string
}

interface PositionRow {
  symbol: string
  qty: number
  available_qty: number
  frozen_qty: number
  cost_basis_avg: string
  cost_total: string
  market_price: string | null
  market_value: string | null
  unrealized_pnl: string | null
  unrealized_pnl_pct: number | null
  first_open_date: string
  holding_days: number
  days_to_tax_free: number | null
  realized_pnl: string
  total_dividend: string
  lots: Lot[]
}

interface Summary {
  account_id: string
  as_of: string
  cash: string
  market_value: string
  total_value: string
  position_count: number
  realized_pnl: string
  unrealized_pnl: string
  total_fee: string
  total_dividend: string
  total_dividend_tax: string
  transactions: number
  is_empty: boolean
  unpriced_symbols: string[]
  message: string
  ledger_path: string
}

interface Txn {
  txn_id: string
  txn_type: string
  trade_date: string
  symbol: string | null
  qty: number
  price: string
  amount: string
  net_cash: string
  total_fee: string
  source: string
  note: string
}

const system = useSystemStore()

const summary = ref<Summary | null>(null)
const positions = ref<PositionRow[]>([])
const transactions = ref<Txn[]>([])
const busy = ref(false)
const errorMessage = ref('')

const tradeForm = ref({
  symbol: '',
  side: 'buy',
  qty: 100,
  price: '',
  trade_date: '',
  commission: '0',
  stamp_tax: '0',
  transfer_fee: '0',
  note: '',
})
const cashForm = ref({ amount: '', note: '' })

const readonly = computed(() => system.status?.readonly ?? false)

function pnlClass(value: string | number | null): string {
  if (value === null) return 'qs-flat'
  const n = Number(value)
  if (!n) return 'qs-flat'
  return n > 0 ? 'qs-up' : 'qs-down'
}

async function loadAll(): Promise<void> {
  errorMessage.value = ''
  try {
    const [s, p, t] = await Promise.all([
      api.get<Summary>('/api/account/summary'),
      api.get<{ positions: PositionRow[] }>('/api/account/positions'),
      api.get<{ transactions: Txn[] }>('/api/account/transactions', { limit: 200 }),
    ])
    summary.value = s
    positions.value = p.positions
    transactions.value = t.transactions
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  }
}

async function submitTrade(): Promise<void> {
  if (!tradeForm.value.symbol.trim() || !tradeForm.value.price.trim()) {
    ElMessage.warning('标的与成交价必填')
    return
  }
  busy.value = true
  try {
    await api.post('/api/account/trade', {
      ...tradeForm.value,
      trade_date: tradeForm.value.trade_date || null,
    })
    ElMessage.success('已入账')
    tradeForm.value.note = ''
    await loadAll()
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

async function submitCash(kind: 'deposit' | 'withdraw'): Promise<void> {
  if (!cashForm.value.amount.trim()) {
    ElMessage.warning('金额必填')
    return
  }
  busy.value = true
  try {
    await api.post(`/api/account/${kind}`, cashForm.value)
    ElMessage.success(kind === 'deposit' ? '已入金' : '已出金')
    cashForm.value = { amount: '', note: '' }
    await loadAll()
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

onMounted(loadAll)
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>账户</h2>
        <div class="qs-sub">
          持仓由流水重放得出，不可直接编辑（红线 R8）。录错了用一笔反向流水冲正并写明理由。
        </div>
      </div>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" style="margin-bottom: 12px" />
    <el-alert
      v-if="summary?.is_empty"
      type="info"
      show-icon
      :closable="false"
      title="账本为空，系统正按冷启动出建议"
      description="录入真实持仓后，建议里的「持仓与技术分析」才会带上真实成本与持有期，卖出建议与免税倒计时也才会出现。"
      style="margin-bottom: 12px"
    />
    <el-alert
      v-if="summary?.unpriced_symbols.length"
      type="warning"
      show-icon
      :closable="false"
      :title="`${summary.unpriced_symbols.join('、')} 在数据湖里没有行情，已按 0 计入市值`"
      description="总资产会因此偏低。请先在数据页更新这些标的的行情。"
      style="margin-bottom: 12px"
    />

    <div class="qs-cards">
      <div class="qs-card">
        <div class="qs-label">总资产</div>
        <div class="qs-value qs-mono">{{ summary?.total_value ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">现金</div>
        <div class="qs-value qs-mono">{{ summary?.cash ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">持仓市值</div>
        <div class="qs-value qs-mono">{{ summary?.market_value ?? '—' }}</div>
      </div>
      <div class="qs-card">
        <div class="qs-label">浮动盈亏</div>
        <div class="qs-value qs-mono" :class="pnlClass(summary?.unrealized_pnl ?? null)">
          {{ summary?.unrealized_pnl ?? '—' }}
        </div>
      </div>
      <div class="qs-card">
        <div class="qs-label">已实现盈亏</div>
        <div class="qs-value qs-mono" :class="pnlClass(summary?.realized_pnl ?? null)">
          {{ summary?.realized_pnl ?? '—' }}
        </div>
      </div>
      <div class="qs-card">
        <div class="qs-label">累计费用 / 红利税</div>
        <div class="qs-value qs-mono" style="font-size: 17px">
          {{ summary?.total_fee ?? '—' }} / {{ summary?.total_dividend_tax ?? '—' }}
        </div>
      </div>
    </div>

    <el-card shadow="never" style="margin-bottom: 16px">
      <template #header>持仓（{{ positions.length }}）· 展开查看批次明细</template>
      <div v-if="!positions.length" class="qs-empty">暂无持仓</div>
      <el-table v-else :data="positions" size="small" row-key="symbol">
        <el-table-column type="expand">
          <template #default="{ row }">
            <div class="qs-lots">
              <div class="qs-lots-title">
                批次明细——红利税按持股期限分三档（≤1月 20% / 1月-1年 10% / &gt;1年 免），
                只有批次级记录才算得出来
              </div>
              <el-table :data="row.lots" size="small">
                <el-table-column prop="open_date" label="建仓日" width="120" />
                <el-table-column prop="remaining_qty" label="剩余" width="90" />
                <el-table-column prop="original_qty" label="原始" width="90" />
                <el-table-column label="成本价" width="110">
                  <template #default="{ row: lot }">
                    <span class="qs-mono">{{ lot.cost_price }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="已收分红" width="110">
                  <template #default="{ row: lot }">
                    <span class="qs-mono">{{ lot.accrued_dividend }}</span>
                  </template>
                </el-table-column>
                <el-table-column prop="lot_id" label="批次 ID" show-overflow-tooltip />
              </el-table>
            </div>
          </template>
        </el-table-column>
        <el-table-column label="标的" width="110">
          <template #default="{ row }">
            <strong class="qs-mono">{{ row.symbol }}</strong>
          </template>
        </el-table-column>
        <el-table-column label="持仓 / 可卖" width="110">
          <template #default="{ row }">
            <span class="qs-mono">{{ row.qty }}</span>
            <span class="qs-flat"> / {{ row.available_qty }}</span>
          </template>
        </el-table-column>
        <el-table-column label="成本" width="100">
          <template #default="{ row }">
            <span class="qs-mono">{{ row.cost_basis_avg }}</span>
          </template>
        </el-table-column>
        <el-table-column label="现价" width="100">
          <template #default="{ row }">
            <span class="qs-mono">{{ row.market_price ?? '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="市值" width="120">
          <template #default="{ row }">
            <span class="qs-mono">{{ row.market_value ?? '—' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="浮动盈亏" width="150">
          <template #default="{ row }">
            <span class="qs-mono" :class="pnlClass(row.unrealized_pnl)">
              {{ row.unrealized_pnl ?? '—' }}
              <template v-if="row.unrealized_pnl_pct !== null">
                （{{ (row.unrealized_pnl_pct * 100).toFixed(2) }}%）
              </template>
            </span>
          </template>
        </el-table-column>
        <el-table-column label="持有天数" width="90" prop="holding_days" />
        <el-table-column label="免税倒计时" min-width="130">
          <template #default="{ row }">
            <el-tag v-if="row.days_to_tax_free !== null" size="small" type="warning">
              再持 {{ row.days_to_tax_free }} 天免红利税
            </el-tag>
            <span v-else class="qs-flat">已满一年或不适用</span>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-row :gutter="16">
      <el-col :span="14">
        <el-card shadow="never">
          <template #header>流水（{{ transactions.length }}）· 只追加，不可修改</template>
          <div v-if="!transactions.length" class="qs-empty">暂无流水</div>
          <el-table v-else :data="transactions" size="small" max-height="440">
            <el-table-column prop="trade_date" label="日期" width="110" />
            <el-table-column prop="txn_type" label="类型" width="90" />
            <el-table-column label="标的" width="110">
              <template #default="{ row }">
                <span class="qs-mono">{{ row.symbol ?? '—' }}</span>
              </template>
            </el-table-column>
            <el-table-column label="数量" width="90">
              <template #default="{ row }">
                <span class="qs-mono">{{ row.qty || '—' }}</span>
              </template>
            </el-table-column>
            <el-table-column label="价格" width="90">
              <template #default="{ row }">
                <span class="qs-mono">{{ row.price }}</span>
              </template>
            </el-table-column>
            <el-table-column label="现金变化" width="120">
              <template #default="{ row }">
                <span class="qs-mono" :class="pnlClass(row.net_cash)">{{ row.net_cash }}</span>
              </template>
            </el-table-column>
            <el-table-column prop="source" label="来源" width="110" />
            <el-table-column prop="note" label="备注" show-overflow-tooltip />
          </el-table>
          <div v-if="summary" class="qs-sub" style="margin-top: 8px">
            账本文件：<span class="qs-mono">{{ summary.ledger_path }}</span>
          </div>
        </el-card>
      </el-col>

      <el-col :span="10">
        <el-card shadow="never" style="margin-bottom: 16px">
          <template #header>录入成交</template>
          <el-form label-width="80px" size="small">
            <el-form-item label="标的">
              <el-input v-model="tradeForm.symbol" placeholder="600519.SH" />
            </el-form-item>
            <el-form-item label="方向">
              <el-radio-group v-model="tradeForm.side">
                <el-radio-button value="buy">买入</el-radio-button>
                <el-radio-button value="sell">卖出</el-radio-button>
              </el-radio-group>
              <span class="qs-sub" style="margin-left: 10px">数量填正数，方向由这里决定</span>
            </el-form-item>
            <el-form-item label="数量">
              <el-input-number v-model="tradeForm.qty" :min="1" :step="100" style="width: 140px" />
            </el-form-item>
            <el-form-item label="成交价">
              <el-input v-model="tradeForm.price" placeholder="1596.52" style="width: 140px" />
            </el-form-item>
            <el-form-item label="成交日">
              <el-input v-model="tradeForm.trade_date" placeholder="留空表示今天" style="width: 160px" />
            </el-form-item>
            <el-form-item label="佣金">
              <el-input v-model="tradeForm.commission" style="width: 100px" />
              <span class="qs-sub" style="margin-left: 8px">印花税</span>
              <el-input v-model="tradeForm.stamp_tax" style="width: 100px; margin-left: 6px" />
            </el-form-item>
            <el-form-item label="备注">
              <el-input v-model="tradeForm.note" />
            </el-form-item>
            <el-button type="primary" :loading="busy" :disabled="readonly" @click="submitTrade">
              入账
            </el-button>
          </el-form>
        </el-card>

        <el-card shadow="never">
          <template #header>资金流水</template>
          <el-form label-width="80px" size="small">
            <el-form-item label="金额">
              <el-input v-model="cashForm.amount" placeholder="100000" style="width: 160px" />
              <span class="qs-sub" style="margin-left: 8px">填正数，方向由按钮决定</span>
            </el-form-item>
            <el-form-item label="备注">
              <el-input v-model="cashForm.note" />
            </el-form-item>
            <el-button type="primary" :loading="busy" :disabled="readonly" @click="submitCash('deposit')">
              入金
            </el-button>
            <el-button :loading="busy" :disabled="readonly" @click="submitCash('withdraw')">
              出金
            </el-button>
          </el-form>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.qs-lots {
  padding: 8px 16px 12px 48px;
  background: #fafbfc;
}
.qs-lots-title {
  font-size: 12px;
  color: var(--qs-muted);
  margin-bottom: 8px;
  line-height: 1.7;
}
</style>
