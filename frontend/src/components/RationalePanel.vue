<script setup lang="ts">
/**
 * 四支柱解释展开态（docs/09 P2）。
 *
 * 四个支柱**全都要显示**，不能因为"看起来不重要"折叠掉任何一个：
 *
 * ① 量化依据 —— 因子分与信号；
 * ② 持仓与技术分析 —— 已持仓标的用真实成本，不是市价近似；
 * ③ 情报证据 —— **每条必须带原文链接与发布时间**（红线 I-R4）。
 *    无相关情报时显示"近 N 日无相关消息"而不是留空，
 *    留空让人分不清"没查"和"查了没有"；
 * ④ 反面证据与证伪条件 —— 只展示看多理由的界面会助长确认偏误，
 *    所以这一块给了显眼底色。
 *
 * 🤖 标记只在 `llm_involved` 时出现，并显示**具体调整量**而不是笼统一句
 * "AI 参与了"——红线 LR2 要求 LLM 的影响有界且可见。
 */
import { computed } from 'vue'

import type { Rationale } from '@/api/types'

const props = defineProps<{ rationale: Rationale; baseScore?: number; finalScore?: number }>()

const confidencePct = computed(() =>
  props.rationale.confidence === null ? '—' : `${(props.rationale.confidence * 100).toFixed(0)}%`,
)

const scoreShift = computed(() => {
  if (props.baseScore === undefined || props.finalScore === undefined) return ''
  const delta = props.finalScore - props.baseScore
  const sign = delta >= 0 ? '+' : ''
  return `${props.baseScore.toFixed(3)} → ${props.finalScore.toFixed(3)}（${sign}${delta.toFixed(3)}）`
})

function sentimentClass(value: number | null): string {
  if (value === null || Math.abs(value) < 0.1) return 'qs-flat'
  return value > 0 ? 'qs-up' : 'qs-down'
}
</script>

<template>
  <div class="qs-rationale">
    <div class="qs-verdict">
      <strong>{{ rationale.verdict }}</strong>
      <el-tag size="small" type="info">置信度 {{ confidencePct }}</el-tag>
      <el-tag v-if="rationale.llm_involved" size="small" type="warning">🤖 大模型参与</el-tag>
      <el-tag v-if="!rationale.is_complete" size="small" type="danger">
        解释不完整：{{ rationale.missing_pillars.join('、') }}
      </el-tag>
    </div>
    <p v-if="rationale.confidence_basis" class="qs-basis">{{ rationale.confidence_basis }}</p>

    <div v-if="rationale.llm_involved" class="qs-llm">
      <div class="qs-pillar-title">🤖 大模型影响</div>
      <p>
        对打分的调整量 <strong>{{ (rationale.llm_adjustment ?? 0).toFixed(4) }}</strong>
        <template v-if="scoreShift">，{{ scoreShift }}</template>
      </p>
      <p class="qs-note">
        大模型只做文本理解与归纳，不参与价格预测；其影响受 α 限幅且可一键关闭，
        关闭后系统功能完整。
      </p>
    </div>

    <div class="qs-pillar">
      <h4>① 量化依据</h4>
      <ul v-if="rationale.quant_evidence.length">
        <li v-for="(e, i) in rationale.quant_evidence" :key="i">
          <strong>{{ e.name }}</strong>
          <span v-if="e.value !== null" class="qs-mono"> {{ e.value.toFixed(4) }}</span>
          <span v-if="e.detail"> — {{ e.detail }}</span>
        </li>
      </ul>
      <p v-else class="qs-note">无</p>
    </div>

    <div class="qs-pillar">
      <h4>② 持仓与技术分析</h4>
      <ul v-if="rationale.technical.statements.length">
        <li v-for="(s, i) in rationale.technical.statements" :key="i">{{ s }}</li>
      </ul>
      <p v-else class="qs-note">无</p>
      <div v-if="rationale.technical.days_to_tax_free !== null" class="qs-tax">
        距满 1 年免红利税还有 {{ rationale.technical.days_to_tax_free }} 天<template
          v-if="rationale.technical.tax_saving_if_wait"
          >，预计可省 {{ rationale.technical.tax_saving_if_wait }} 元</template
        >
      </div>
    </div>

    <div class="qs-pillar">
      <h4>③ 情报证据</h4>
      <ul v-if="rationale.intel_evidence.length" class="qs-intel-list">
        <li v-for="(item, i) in rationale.intel_evidence" :key="i">
          <a :href="item.url" target="_blank" rel="noopener noreferrer">{{ item.title }}</a>
          <span class="qs-meta">
            {{ item.source }} · {{ item.published_at }} · 重要性 {{ item.importance }} ·
            <span :class="sentimentClass(item.sentiment)">
              情绪 {{ item.sentiment?.toFixed(2) ?? '—' }}
            </span>
            · {{ item.impact }}
          </span>
          <p v-if="item.summary" class="qs-note">{{ item.summary }}</p>
        </li>
      </ul>
      <p v-else class="qs-note">{{ rationale.intel_absent_note || '近 N 日无相关消息' }}</p>
    </div>

    <div class="qs-pillar qs-counter">
      <h4>④ 反面证据与证伪条件</h4>
      <ul v-if="rationale.counter_evidence.length">
        <li v-for="(e, i) in rationale.counter_evidence" :key="i">
          <strong>{{ e.name }}</strong>
          <span v-if="e.value !== null" class="qs-mono"> {{ e.value.toFixed(4) }}</span>
          <span v-if="e.detail"> — {{ e.detail }}</span>
        </li>
      </ul>
      <div v-if="rationale.falsification.length">
        <div class="qs-pillar-title">什么情况下这个判断会被推翻</div>
        <ul>
          <li v-for="(f, i) in rationale.falsification" :key="i">{{ f }}</li>
        </ul>
      </div>
    </div>

    <div v-if="rationale.risk_notes.length" class="qs-pillar">
      <h4>风险提示</h4>
      <ul>
        <li v-for="(r, i) in rationale.risk_notes" :key="i">{{ r }}</li>
      </ul>
    </div>
  </div>
</template>

<style scoped>
.qs-rationale {
  padding: 4px 2px 2px;
}
.qs-verdict {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 6px;
}
.qs-basis,
.qs-note {
  color: var(--qs-muted);
  font-size: 12px;
  margin: 4px 0;
  line-height: 1.7;
}
.qs-llm {
  background: #fffbe6;
  border-left: 3px solid #e6a23c;
  padding: 8px 12px;
  border-radius: 0 4px 4px 0;
  margin-bottom: 14px;
}
.qs-pillar-title {
  font-size: 12px;
  color: var(--qs-muted);
  font-weight: 600;
  margin: 6px 0 2px;
}
.qs-intel-list li {
  margin-bottom: 8px;
}
.qs-meta {
  display: block;
  font-size: 12px;
  color: var(--qs-muted);
}
.qs-tax {
  margin-top: 6px;
  font-size: 12px;
  color: #b45309;
}
</style>
