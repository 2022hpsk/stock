<script setup lang="ts">
/**
 * 应用外壳：侧边导航 + 顶栏（急停 / 只读标识 / WS 状态）+ 口令闸门。
 *
 * 顶栏上常驻的两样东西不是装饰：
 * - **急停按钮**（docs/09 第五节 F19）：出事时找不到开关是最糟的；
 * - **急停横幅**：`halted` 时红色置顶，并把所有下单入口变灰。
 */
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'

import { getToken, setToken } from '@/api/client'
import { routes } from '@/router'
import { useEventStore } from '@/stores/events'
import { useSystemStore } from '@/stores/system'

const system = useSystemStore()
const events = useEventStore()
const route = useRoute()

const tokenInput = ref('')
const authed = ref(Boolean(getToken()))

const halted = computed(() => system.status?.halt.halted ?? false)
const readonly = computed(() => system.status?.readonly ?? false)

const navItems = routes.map((r) => ({
  path: r.path,
  title: (r.meta as { title: string }).title,
  tag: (r.meta as { icon: string }).icon,
}))

async function submitToken(): Promise<void> {
  setToken(tokenInput.value)
  await system.refresh()
  if (system.status) {
    authed.value = true
    events.connect()
  } else {
    ElMessage.error(system.error || '口令无效')
  }
}

async function onHalt(): Promise<void> {
  try {
    const { value } = await ElMessageBox.prompt(
      '急停后所有下单路径一律拒绝，直到显式恢复。请填写原因（事后复盘要靠它）。',
      '确认急停',
      { inputPlaceholder: '例如：盘中出现异常波动，先停下来看看', inputPattern: /\S/, inputErrorMessage: '原因必填' },
    )
    await system.halt(value)
    ElMessage.success('已急停')
  } catch {
    // 用户取消
  }
}

async function onResume(): Promise<void> {
  await ElMessageBox.confirm('确认解除急停？解除后下单路径恢复可用。', '解除急停', { type: 'warning' })
  await system.resume()
  ElMessage.success('已解除急停')
}

onMounted(async () => {
  if (authed.value) {
    await system.refresh()
    if (system.status) events.connect()
    else authed.value = false
  }
})

onUnmounted(() => events.disconnect())

// 切页时刷新状态：急停可能是由另一个终端（CLI）触发的
watch(
  () => route.path,
  () => {
    if (authed.value) void system.refresh()
  },
)
</script>

<template>
  <div v-if="!authed" class="qs-gate">
    <div class="qs-gate-box">
      <h1>quantstock</h1>
      <p class="qs-sub">
        界面是下单入口，必须凭口令访问。口令在 <code>quantstock ui</code> 启动时打印在终端。
      </p>
      <el-input
        v-model="tokenInput"
        placeholder="粘贴访问口令"
        show-password
        size="large"
        @keyup.enter="submitToken"
      />
      <el-button type="primary" size="large" style="width: 100%; margin-top: 12px" @click="submitToken">
        进入
      </el-button>
    </div>
  </div>

  <el-container v-else class="qs-shell">
    <el-aside width="196px" class="qs-aside">
      <div class="qs-brand">quantstock</div>
      <el-menu :default-active="route.path" router class="qs-menu">
        <el-menu-item v-for="item in navItems" :key="item.path" :index="item.path">
          <span class="qs-nav-tag">{{ item.tag }}</span>
          <span>{{ item.title }}</span>
        </el-menu-item>
      </el-menu>
    </el-aside>

    <el-container>
      <el-header height="52px" class="qs-header">
        <div class="qs-header-left">
          <el-tag v-if="readonly" type="info" effect="dark">只读模式</el-tag>
          <el-tag v-if="system.status" :type="system.status.broker === 'paper' ? 'success' : 'danger'">
            通道 {{ system.status.broker }}
          </el-tag>
          <el-tag v-if="system.status" :type="system.status.llm.enabled ? 'warning' : 'info'">
            LLM {{ system.status.llm.enabled ? system.status.llm.mode : '关闭' }}
          </el-tag>
        </div>
        <div class="qs-header-right">
          <span class="qs-ws" :class="events.connected ? 'qs-down' : 'qs-flat'">
            ● {{ events.connected ? '实时已连接' : '实时未连接' }}
          </span>
          <el-button v-if="!halted" type="danger" :disabled="readonly" @click="onHalt">急停</el-button>
          <el-button v-else type="warning" :disabled="readonly" @click="onResume">解除急停</el-button>
        </div>
      </el-header>

      <el-alert
        v-if="halted"
        type="error"
        show-icon
        :closable="false"
        class="qs-halt-banner"
        :title="`系统已急停：${system.status?.halt.reason || '未填写原因'}`"
        :description="`触发于 ${system.status?.halt.halted_at ?? '—'}，由 ${system.status?.halt.halted_by || '—'} 操作。所有下单入口已禁用。`"
      />

      <el-main class="qs-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.qs-gate {
  min-height: 100vh;
  display: grid;
  place-items: center;
}
.qs-gate-box {
  width: 380px;
  background: var(--qs-panel);
  border: 1px solid var(--qs-border);
  border-radius: 10px;
  padding: 28px;
}
.qs-gate-box h1 {
  margin: 0 0 8px;
  font-size: 22px;
}
.qs-gate-box .qs-sub {
  color: var(--qs-muted);
  font-size: 13px;
  line-height: 1.7;
  margin: 0 0 16px;
}
.qs-shell {
  min-height: 100vh;
}
.qs-aside {
  background: #fff;
  border-right: 1px solid var(--qs-border);
}
.qs-brand {
  font-weight: 700;
  font-size: 17px;
  padding: 16px 20px;
  border-bottom: 1px solid var(--qs-border);
}
.qs-menu {
  border-right: none;
}
.qs-nav-tag {
  display: inline-block;
  width: 30px;
  font-size: 11px;
  color: var(--qs-muted);
}
.qs-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #fff;
  border-bottom: 1px solid var(--qs-border);
}
.qs-header-left,
.qs-header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.qs-ws {
  font-size: 12px;
}
.qs-halt-banner {
  border-radius: 0;
}
.qs-main {
  padding: 0;
  background: var(--qs-bg);
}
</style>
