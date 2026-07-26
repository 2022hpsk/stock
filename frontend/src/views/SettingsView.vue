<script setup lang="ts">
/**
 * P13 配置（docs/09 第四节）。
 *
 * 保存流程刻意是**三步**而不是一步：校验 → Diff 预览 → 确认写入（自动备份）。
 * 配置直接决定下单行为，"点错一下就生效"太危险；有 Diff 才能看清
 * 自己到底改了什么，有备份才能滚回去。
 *
 * 密钥**只显示是否已配置**，永不回显明文（红线 R7）。
 */
import { computed, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'

import { api, ApiError } from '@/api/client'
import { useSystemStore } from '@/stores/system'

const system = useSystemStore()

const raw = ref('')
const original = ref('')
const diff = ref('')
const issues = ref<Array<{ location: string; message: string; input: unknown }>>([])
const backups = ref<string[]>([])
const secrets = ref<Record<string, boolean>>({})
const busy = ref(false)
const errorMessage = ref('')

const dirty = computed(() => raw.value !== original.value)

function parsed(): Record<string, unknown> {
  return JSON.parse(raw.value) as Record<string, unknown>
}

async function load(): Promise<void> {
  const [config, backupList, secretStatus] = await Promise.all([
    api.get<Record<string, unknown>>('/api/config'),
    api.get<{ versions: string[] }>('/api/config/backups'),
    api.get<Record<string, boolean>>('/api/secrets/status'),
  ])
  raw.value = JSON.stringify(config, null, 2)
  original.value = raw.value
  backups.value = backupList.versions
  secrets.value = secretStatus
}

async function preview(): Promise<void> {
  busy.value = true
  errorMessage.value = ''
  diff.value = ''
  issues.value = []
  try {
    const res = await api.post<{ valid: boolean; issues: typeof issues.value; diff: string }>(
      '/api/config/preview',
      { config: parsed() },
    )
    issues.value = res.issues
    diff.value = res.diff
    if (res.valid && !res.diff) ElMessage.info('没有变更')
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

async function save(): Promise<void> {
  await ElMessageBox.confirm(
    '保存前会自动备份当前配置，可随时回滚。确认写入？',
    '保存配置',
    { type: 'warning' },
  )
  busy.value = true
  errorMessage.value = ''
  try {
    const res = await api.put<{ saved: boolean; backup: string; issues: typeof issues.value }>(
      '/api/config',
      { config: parsed(), changed_by: 'ui' },
    )
    issues.value = res.issues
    if (res.saved) {
      ElMessage.success(`已保存，备份于 ${res.backup}`)
      await load()
      await system.refresh()
    } else {
      ElMessage.error('未保存，请看校验问题')
    }
  } catch (e) {
    errorMessage.value = e instanceof ApiError ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

async function rollback(version: string): Promise<void> {
  await ElMessageBox.confirm(`回滚到 ${version}？当前配置会先被备份。`, '回滚', { type: 'warning' })
  await api.post('/api/config/rollback', { version })
  ElMessage.success('已回滚')
  await load()
  await system.refresh()
}

onMounted(load)
</script>

<template>
  <div class="qs-page">
    <div class="qs-page-header">
      <div>
        <h2>配置</h2>
        <div class="qs-sub">
          校验 → Diff 预览 → 保存（自动备份）。A 类风控规则在界面上没有关闭入口——
          界面不能成为绕过风控的后门。
        </div>
      </div>
      <div style="display: flex; gap: 8px">
        <el-button :loading="busy" @click="preview">校验并预览 Diff</el-button>
        <el-button type="primary" :loading="busy" :disabled="!dirty || system.status?.readonly" @click="save">
          保存
        </el-button>
      </div>
    </div>

    <el-alert v-if="errorMessage" type="error" :title="errorMessage" show-icon :closable="false" style="margin-bottom: 12px" />

    <el-row :gutter="16">
      <el-col :span="15">
        <el-card shadow="never">
          <template #header>
            当前配置 <span class="qs-sub">{{ dirty ? '（已修改，未保存）' : '' }}</span>
          </template>
          <el-input v-model="raw" type="textarea" :rows="26" class="qs-mono" />
        </el-card>
      </el-col>

      <el-col :span="9">
        <el-card v-if="issues.length" shadow="never" style="margin-bottom: 16px">
          <template #header>校验问题（{{ issues.length }}）</template>
          <div v-for="(i, idx) in issues" :key="idx" class="qs-issue">
            <strong class="qs-mono">{{ i.location }}</strong>
            <div>{{ i.message }}</div>
          </div>
        </el-card>

        <el-card v-if="diff" shadow="never" style="margin-bottom: 16px">
          <template #header>Diff 预览</template>
          <pre class="qs-mono qs-diff">{{ diff }}</pre>
        </el-card>

        <el-card shadow="never" style="margin-bottom: 16px">
          <template #header>密钥状态</template>
          <p class="qs-sub" style="margin-top: 0">
            **只显示是否已配置，永不回显明文。** 密钥请写进 <code>.env</code> 或环境变量，
            不要写进 <code>config/*.yaml</code>——后者会被提交进版本库。
          </p>
          <el-table :data="Object.entries(secrets).map(([name, ok]) => ({ name, ok }))" size="small">
            <el-table-column prop="name" label="密钥" />
            <el-table-column label="状态" width="100">
              <template #default="{ row }">
                <el-tag :type="row.ok ? 'success' : 'info'" size="small">
                  {{ row.ok ? '已配置' : '未配置' }}
                </el-tag>
              </template>
            </el-table-column>
          </el-table>
        </el-card>

        <el-card shadow="never">
          <template #header>备份与回滚（{{ backups.length }}）</template>
          <div v-if="!backups.length" class="qs-empty">暂无备份</div>
          <div v-for="v in backups" :key="v" class="qs-backup">
            <span class="qs-mono">{{ v }}</span>
            <el-button size="small" :disabled="system.status?.readonly" @click="rollback(v)">回滚</el-button>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.qs-issue {
  padding: 8px 0;
  border-bottom: 1px solid var(--qs-border);
  font-size: 13px;
}
.qs-diff {
  max-height: 320px;
  overflow: auto;
  font-size: 12px;
  background: #f8f9fa;
  padding: 10px;
  border-radius: 6px;
  white-space: pre-wrap;
}
.qs-backup {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  border-bottom: 1px solid var(--qs-border);
  font-size: 13px;
}
</style>
