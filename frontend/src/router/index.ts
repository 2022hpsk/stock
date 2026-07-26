/**
 * 路由表。页面编号对应 docs/09-可视化界面规格.md 第三节。
 *
 * 全部懒加载：仪表盘不该为了一个用不到的回测页去下载 ECharts 的全部图表类型。
 */

import { createRouter, createWebHistory } from 'vue-router'

export const routes = [
  { path: '/', name: 'dashboard', component: () => import('@/views/DashboardView.vue'), meta: { title: '仪表盘', icon: 'P0' } },
  { path: '/account', name: 'account', component: () => import('@/views/AccountView.vue'), meta: { title: '账户', icon: 'P1' } },
  { path: '/advisor', name: 'advisor', component: () => import('@/views/AdvisorView.vue'), meta: { title: '每日建议', icon: 'P2' } },
  { path: '/execution', name: 'execution', component: () => import('@/views/ExecutionView.vue'), meta: { title: '执行', icon: 'P3' } },
  { path: '/data', name: 'data', component: () => import('@/views/DataView.vue'), meta: { title: '数据', icon: 'P4' } },
  { path: '/intel', name: 'intel', component: () => import('@/views/IntelView.vue'), meta: { title: '情报', icon: 'P5' } },
  { path: '/risk', name: 'risk', component: () => import('@/views/RiskView.vue'), meta: { title: '风控', icon: 'P10' } },
  { path: '/backtest', name: 'backtest', component: () => import('@/views/BacktestView.vue'), meta: { title: '回测', icon: 'P8' } },
  { path: '/review', name: 'review', component: () => import('@/views/ReviewView.vue'), meta: { title: '复盘', icon: 'P12' } },
  { path: '/llm', name: 'llm', component: () => import('@/views/LlmView.vue'), meta: { title: '大模型', icon: 'P16' } },
  { path: '/settings', name: 'settings', component: () => import('@/views/SettingsView.vue'), meta: { title: '配置', icon: 'P13' } },
]

export const router = createRouter({
  history: createWebHistory(),
  routes,
})
