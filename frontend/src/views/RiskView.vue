<script setup lang="ts">
/**
 * P10 风控（docs/09 第三节）。
 *
 * 这一页最要紧的是**它不提供什么**：
 *
 * - A 类规则**根本不渲染开关**（验收 5）。画一个开关再在点击时拒绝，
 *   和根本不画，是两回事——前者会让人一直去试，也让人误以为"理论上能关"；
 * - 阈值不能在这里改。改阈值走配置页，那条路径有校验、Diff、备份与审计；
 *   在风控页开一个"快速调整"，等于把最该留痕的操作做成最容易悄悄做掉的。
 *
 * 熔断部分显示的是**距离**而不只是状态：状态只有到了才变，
 * 而距离能让人提前看到自己正在往哪走。
 */
import { computed, onMounted, ref } from 'vue'

import { api } from '@/api/client'
import { useSystemStore } from '@/stores/system'

interface Rule {
  rule_id: string
  name: string
  rule_class: string
  description: string
  closable: boolean
  threshold_editable: boolean
  threshold_key: string
  current_threshold: string
}

interface HardLimits {
  enabled: boolean
  max_single_order_amount: string
  max_daily_total_amount: string
  max_daily_order_count: number
  min_account_value_sanity: string
  max_account_value_sanity: string
  message: string
}

interface Circuit {
  daily_loss: { current: number; watch: number; halted: number }
  drawdown_20d: { current: number; watch: number; halted: number }
  recover_drawdown: number
  auto_recovers: boolean
  halt: { halted: boolean; reason: string; halted_at: string; halted_by: string }
  note: string
}

const system = useSystemStore()

const rules = ref<Rule[]>([])
const legend = ref<Record<string, string>>({})
const hardLimits = ref<HardLimits | null>(null)
const circuit = ref<Circuit | null>(null)
const classFilter = ref('')

const filtered = computed(() =>
  classFilter.value ? rules.value.filter((r) => r.rule_class === classFilter.value) : rules.value,
)

const classTag = (c: string): string => (c === 'A' ? 'danger' : c === 'B' ? 'warning' : 'info')

function pct(v: number | undefined): string {
  return v === undefined ? '—' : `${(v * 100).toFixed(2)}%`
}

/** 距阈值还有多远，画成进度条。100% 表示已触及。 */
function progress(current: number, threshold: number): number {
  if (threshold <= 0) return 0
  return Math.min(100, Math.max(0, (current / threshold) * 100))
}

onMounted(async () => {
  const [r, h, c] = await Promise.all([
    api.get<{ rules: Rule[]; legend: Record<string, string> }>('/api/risk/rules'),
    api.get<HardLimits>('/api/risk/hard-limits'),
    api.get<Circuit>('/api/risk/circuit'),
  ])
  rules.value = r.rules
  legend.value = r.legend
  hardLimits.value = h
  circuit.value = c
})
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>风控</h2>
        <div class="qs-sub">
          A 类规则在界面上**没有关闭入口**——界面不能成为绕过风控的后门。阈值修改请到配置页，
          那条路径有校验、Diff 预览、备份与审计。
        </div>
      </div>
      <el-select v-model="classFilter" placeholder="全部分级" clearable style="width: 130px">
        <el-option label="A 市场规则" value="A" />
        <el-option label="B 组合约束" value="B" />
        <el-option label="C 建议性" value="C" />
      </el-select>
    </div>

    <el-alert
      v-if="circuit?.halt.halted"
      type="error"
      show-icon
      :closable="false"
      :title="`系统已急停：${circuit.halt.reason}`"
      :description="`由 ${circuit.halt.halted_by || '—'} 于 ${circuit.halt.halted_at || '—'} 触发。${circuit.note}`"
      style="margin-bottom: 12px"
    />

    <el-row :gutter="16" style="margin-bottom: 16px">
      <el-col :span="12">
        <el-card shadow="never">
          <template #header>熔断距离</template>
          <div v-if="!circuit" class="qs-empty">加载中…</div>
          <template v-else>
            <div class="qs-gauge">
              <div class="qs-gauge-head">
                <span>当日亏损</span>
                <span class="qs-mono">
                  {{ pct(circuit.daily_loss.current) }} / WATCH {{ pct(circuit.daily_loss.watch) }} /
                  HALT {{ pct(circuit.daily_loss.halted) }}
                </span>
              </div>
              <el-progress
                :percentage="progress(circuit.daily_loss.current, circuit.daily_loss.halted)"
                :status="
                  circuit.daily_loss.current >= circuit.daily_loss.halted
                    ? 'exception'
                    : circuit.daily_loss.current >= circuit.daily_loss.watch
                      ? 'warning'
                      : 'success'
                "
              />
            </div>
            <div class="qs-gauge">
              <div class="qs-gauge-head">
                <span>20 日回撤</span>
                <span class="qs-mono">
                  {{ pct(circuit.drawdown_20d.current) }} / WATCH
                  {{ pct(circuit.drawdown_20d.watch) }} / HALT
                  {{ pct(circuit.drawdown_20d.halted) }}
                </span>
              </div>
              <el-progress
                :percentage="progress(circuit.drawdown_20d.current, circuit.drawdown_20d.halted)"
                :status="
                  circuit.drawdown_20d.current >= circuit.drawdown_20d.halted
                    ? 'exception'
                    : circuit.drawdown_20d.current >= circuit.drawdown_20d.watch
                      ? 'warning'
                      : 'success'
                "
              />
            </div>
            <el-alert type="info" :closable="false" show-icon :title="circuit.note" style="margin-top: 12px" />
          </template>
        </el-card>
      </el-col>

      <el-col :span="12">
        <el-card shadow="never">
          <template #header>绝对金额硬闸（A10 / A11）</template>
          <div v-if="!hardLimits" class="qs-empty">加载中…</div>
          <template v-else>
            <el-alert
              :type="hardLimits.enabled ? 'success' : 'error'"
              show-icon
              :closable="false"
              :title="hardLimits.message"
              style="margin-bottom: 12px"
            />
            <el-descriptions :column="1" border size="small">
              <el-descriptions-item label="单笔上限">
                <span class="qs-mono">{{ hardLimits.max_single_order_amount }}</span>
              </el-descriptions-item>
              <el-descriptions-item label="单日总额上限">
                <span class="qs-mono">{{ hardLimits.max_daily_total_amount }}</span>
              </el-descriptions-item>
              <el-descriptions-item label="单日笔数上限">
                {{ hardLimits.max_daily_order_count }}
              </el-descriptions-item>
              <el-descriptions-item label="账户总资产合理区间">
                <span class="qs-mono">
                  {{ hardLimits.min_account_value_sanity }} ~
                  {{ hardLimits.max_account_value_sanity }}
                </span>
              </el-descriptions-item>
            </el-descriptions>
            <p class="qs-sub" style="margin-bottom: 0">
              阈值必须按自己的实际资金规模**手工设定**，程序不自动推导——自动推导会被同一个
              错误数据污染，失去防护意义。比例风控挡不住计算基数出错：总资产算成十倍时，
              「单票不超过 15%」照样会放出一笔十倍大的委托。
            </p>
          </template>
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="never">
      <template #header>
        <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap">
          <span>规则表（{{ filtered.length }}）</span>
          <span v-for="(text, key) in legend" :key="key" class="qs-legend">
            <el-tag :type="classTag(String(key))" size="small">{{ key }}</el-tag>
            {{ text }}
          </span>
        </div>
      </template>
      <el-table :data="filtered" size="small">
        <el-table-column prop="rule_id" label="编号" width="80" />
        <el-table-column prop="name" label="规则" width="150" />
        <el-table-column label="分级" width="80">
          <template #default="{ row }">
            <el-tag :type="classTag(row.rule_class)" size="small">{{ row.rule_class }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="说明" min-width="300" />
        <el-table-column label="当前阈值" width="130">
          <template #default="{ row }">
            <span class="qs-mono">{{ row.current_threshold || '不可配' }}</span>
          </template>
        </el-table-column>
        <el-table-column label="开关" width="130">
          <template #default="{ row }">
            <!-- A/B 类连开关都不渲染。画出来再拒绝会让人一直去试 -->
            <el-switch v-if="row.closable" :model-value="true" disabled />
            <el-tag v-else type="info" size="small">锁定，不可关闭</el-tag>
          </template>
        </el-table-column>
      </el-table>
      <p class="qs-sub" style="margin-bottom: 0">
        阈值在配置页修改（{{ system.status?.readonly ? '当前只读模式' : '需二次确认并记审计' }}）。
      </p>
    </el-card>
  </div>
</template>

<style scoped>
.qs-gauge {
  margin-bottom: 16px;
}
.qs-gauge-head {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  margin-bottom: 6px;
}
.qs-legend {
  font-size: 12px;
  color: var(--qs-muted);
}
</style>
