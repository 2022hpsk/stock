<script setup lang="ts">
/**
 * P8 回测（docs/09 第三节）。
 *
 * 这一页的设计目标是**让人别把一次好看的回测当结论**：
 *
 * - `warnings` 直接贴在指标上方，不折叠；
 * - 准入结论（DSR / PBO）与净值指标**并排**显示。漂亮的 Sharpe 配上
 *   "该 Sharpe 用随机噪声即可试出"，人才会真的停下来想一想；
 * - trials 表展示全部历史尝试。试了 200 次挑出的 Sharpe 2.0 和试了 3 次
 *   得到的 Sharpe 2.0，含金量差着数量级。
 */
import { onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'

import { api, ApiError } from '@/api/client'
import type { Admission, BacktestReport } from '@/api/types'
import { useSystemStore } from '@/stores/system'

const system = useSystemStore()

const form = ref({
  start: '2024-01-01',
  end: '2025-12-31',
  tier: 'core',
  initial_cash: '200000',
  rebalance_days: 5,
  segment: 'train',
})

const report = ref<BacktestReport | null>(null)
const admission = ref<Admission | null>(null)
const trials = ref<Array<Record<string, unknown>>>([])
const running = ref(false)
const errorMessage = ref('')

function pct(v: number | undefined): string {
  return v === undefined ? '—' : `${(v * 100).toFixed(2)}%`
}

async function loadTrials(): Promise<void> {
  const [t, a] = await Promise.all([
    api.get<{ trials: Array<Record<string, unknown>> }>('/api/backtest/trials'),
    api.get<Admission>('/api/backtest/admission'),
  ])
  trials.value = t.trials
  admission.value = a
}

async function run(): Promise<void> {
  running.value = true
  errorMessage.value = ''
  try {
    report.value = await api.post<BacktestReport>('/api/backtest/run', form.value)
    ElMessage.success(report.value.explain)
    await loadTrials()
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    running.value = false
  }
}

onMounted(loadTrials)
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>回测</h2>
        <div class="qs-sub">
          跑的是每日建议的同一套打分与组合逻辑。**每次尝试都会记入 trials**——
          删掉失败尝试会让 DSR 系统性偏乐观。
        </div>
      </div>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" style="margin-bottom: 12px" />

    <el-card shadow="never" style="margin-bottom: 16px">
      <template #header>发起回测</template>
      <el-form :inline="true" size="small">
        <el-form-item label="起始日">
          <el-input v-model="form.start" style="width: 130px" />
        </el-form-item>
        <el-form-item label="结束日">
          <el-input v-model="form.end" style="width: 130px" />
        </el-form-item>
        <el-form-item label="初始资金">
          <el-input v-model="form.initial_cash" style="width: 120px" />
        </el-form-item>
        <el-form-item label="调仓间隔">
          <el-input-number v-model="form.rebalance_days" :min="1" :max="60" style="width: 110px" />
        </el-form-item>
        <el-form-item label="数据段">
          <el-select v-model="form.segment" style="width: 130px">
            <el-option label="train 训练" value="train" />
            <el-option label="validation 验证" value="validation" />
            <el-option label="test 测试（只跑一次）" value="test" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="running" :disabled="system.status?.readonly" @click="run">
            运行
          </el-button>
        </el-form-item>
      </el-form>
      <div class="qs-sub">
        test 段每个策略只允许跑一次——反复在测试集上调参，它就变成了第二个训练集。
      </div>
    </el-card>

    <template v-if="report">
      <el-alert
        v-for="(w, i) in report.warnings"
        :key="i"
        type="warning"
        :title="w"
        show-icon
        :closable="false"
        style="margin-bottom: 8px"
      />

      <div class="qs-cards">
        <div class="qs-card">
          <div class="qs-label">总收益</div>
          <div class="qs-value" :class="report.stats.total_return >= 0 ? 'qs-up' : 'qs-down'">
            {{ pct(report.stats.total_return) }}
          </div>
        </div>
        <div class="qs-card">
          <div class="qs-label">年化</div>
          <div class="qs-value" :class="report.stats.annualized_return >= 0 ? 'qs-up' : 'qs-down'">
            {{ pct(report.stats.annualized_return) }}
          </div>
        </div>
        <div class="qs-card">
          <div class="qs-label">Sharpe</div>
          <div class="qs-value qs-mono">{{ report.stats.sharpe.toFixed(2) }}</div>
        </div>
        <div class="qs-card">
          <div class="qs-label">最大回撤</div>
          <div class="qs-value qs-down">{{ pct(report.stats.max_drawdown) }}</div>
        </div>
        <div class="qs-card">
          <div class="qs-label">TWR / MWR</div>
          <div class="qs-value qs-mono" style="font-size: 17px">
            {{ pct(report.stats.twr) }} / {{ pct(report.stats.mwr) }}
          </div>
        </div>
        <div class="qs-card">
          <div class="qs-label">成交笔数</div>
          <div class="qs-value">{{ report.fills }}</div>
        </div>
      </div>

      <el-card shadow="never" style="margin-bottom: 16px">
        <template #header>详情</template>
        <el-descriptions :column="3" border size="small">
          <el-descriptions-item label="区间">{{ report.start }} ~ {{ report.end }}</el-descriptions-item>
          <el-descriptions-item label="交易日">{{ report.trading_days }}</el-descriptions-item>
          <el-descriptions-item label="试验 ID">{{ report.trial_id || '—' }}</el-descriptions-item>
          <el-descriptions-item label="Sortino">{{ report.stats.sortino.toFixed(2) }}</el-descriptions-item>
          <el-descriptions-item label="Calmar">{{ report.stats.calmar.toFixed(2) }}</el-descriptions-item>
          <el-descriptions-item label="胜率">{{ pct(report.stats.win_rate) }}</el-descriptions-item>
          <el-descriptions-item label="年化波动">{{ pct(report.stats.annualized_volatility) }}</el-descriptions-item>
          <el-descriptions-item label="盈亏比">{{ report.stats.profit_loss_ratio.toFixed(2) }}</el-descriptions-item>
          <el-descriptions-item label="LLM 模式">{{ report.llm_mode }}</el-descriptions-item>
        </el-descriptions>
        <div v-if="Object.keys(report.rejections).length" style="margin-top: 12px">
          <div class="qs-sub">拒单统计</div>
          <el-tag v-for="(n, k) in report.rejections" :key="k" size="small" style="margin: 4px 6px 0 0">
            {{ k }}：{{ n }}
          </el-tag>
        </div>
      </el-card>
    </template>

    <el-row :gutter="16">
      <el-col :span="10">
        <el-card shadow="never">
          <template #header>实盘候选池准入（A5 强制门槛）</template>
          <div v-if="!admission?.available" class="qs-empty">{{ admission?.message ?? '加载中…' }}</div>
          <template v-else>
            <el-alert
              :type="admission.admitted ? 'success' : 'error'"
              :title="admission.admitted ? '通过：可进入实盘候选池' : '未通过：禁止进入实盘候选池'"
              show-icon
              :closable="false"
              style="margin-bottom: 12px"
            />
            <el-descriptions :column="1" border size="small">
              <el-descriptions-item label="DSR（需 ≥ 0.95）">
                <span class="qs-mono" :class="(admission.dsr ?? 0) >= 0.95 ? 'qs-down' : 'qs-up'">
                  {{ admission.dsr?.toFixed(4) }}
                </span>
              </el-descriptions-item>
              <el-descriptions-item label="PBO（需 ≤ 0.5）">
                <span class="qs-mono" :class="(admission.pbo ?? 1) <= 0.5 ? 'qs-down' : 'qs-up'">
                  {{ admission.pbo?.toFixed(4) }}
                </span>
              </el-descriptions-item>
              <el-descriptions-item label="试验次数">{{ admission.n_trials }}</el-descriptions-item>
            </el-descriptions>
            <ul v-if="admission.reasons?.length" class="qs-reasons">
              <li v-for="(r, i) in admission.reasons" :key="i">{{ r }}</li>
            </ul>
          </template>
        </el-card>
      </el-col>

      <el-col :span="14">
        <el-card shadow="never">
          <template #header>试验记录（{{ trials.length }}）</template>
          <div v-if="!trials.length" class="qs-empty">还没有回测记录</div>
          <el-table v-else :data="trials" size="small" max-height="420">
            <el-table-column prop="trial_id" label="ID" width="180" show-overflow-tooltip />
            <el-table-column prop="segment" label="段" width="90" />
            <el-table-column label="Sharpe" width="90">
              <template #default="{ row }">
                <span class="qs-mono">{{ Number(row.sharpe).toFixed(3) }}</span>
              </template>
            </el-table-column>
            <el-table-column label="年化" width="90">
              <template #default="{ row }">
                <span class="qs-mono">{{ (Number(row.annual_return) * 100).toFixed(2) }}%</span>
              </template>
            </el-table-column>
            <el-table-column label="回撤" width="90">
              <template #default="{ row }">
                <span class="qs-mono qs-down">{{ (Number(row.max_drawdown) * 100).toFixed(2) }}%</span>
              </template>
            </el-table-column>
            <el-table-column prop="created_at" label="时间" show-overflow-tooltip />
          </el-table>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.qs-reasons {
  margin: 12px 0 0;
  padding-left: 18px;
  color: var(--qs-up);
  font-size: 13px;
  line-height: 1.8;
}
</style>
