<script setup lang="ts">
/**
 * 蜡烛图 + 成交量副图（docs/09 第七节）。
 *
 * 两处 A 股特有的处理：
 *
 * 1. **红涨绿跌**。ECharts 的默认配色是国际习惯（绿涨红跌），
 *    直接用会让中文用户扫一眼时把盈利读成亏损；
 * 2. **复权口径必须显示在标题里**。数据湖存的是后复权价，
 *    跟看盘软件的不复权价对不上——不标口径的话，用户会以为数据错了。
 */
import * as echarts from 'echarts/core'
import { BarChart, CandlestickChart } from 'echarts/charts'
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  TitleComponent,
  TooltipComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { onBeforeUnmount, onMounted, ref, watch } from 'vue'

import type { Bar } from '@/api/types'

echarts.use([
  CandlestickChart,
  BarChart,
  GridComponent,
  TooltipComponent,
  TitleComponent,
  LegendComponent,
  DataZoomComponent,
  CanvasRenderer,
])

const props = defineProps<{ bars: Bar[]; symbol: string; adjust: string }>()

const container = ref<HTMLDivElement | null>(null)
let chart: echarts.ECharts | null = null

const UP = '#e5484d'
const DOWN = '#30a46c'

function render(): void {
  if (!chart) return
  const dates = props.bars.map((b) => b.date)
  // ECharts 蜡烛图的顺序是 [开, 收, 低, 高]，不是 OHLC。
  // 顺序写错图不会报错，只会画出一堆形状怪异的蜡烛
  const candles = props.bars.map((b) => [
    Number(b.open),
    Number(b.close),
    Number(b.low),
    Number(b.high),
  ])
  const volumes = props.bars.map((b, i) => ({
    value: b.volume,
    itemStyle: { color: candles[i][1] >= candles[i][0] ? UP : DOWN },
  }))

  chart.setOption(
    {
      animation: false,
      title: {
        text: `${props.symbol}`,
        subtext: `复权口径 ${props.adjust}（与看盘软件的不复权价不同）`,
        left: 8,
        top: 4,
        textStyle: { fontSize: 14 },
        subtextStyle: { fontSize: 11 },
      },
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      grid: [
        { left: 56, right: 20, top: 56, height: '52%' },
        { left: 56, right: 20, bottom: 46, height: '16%' },
      ],
      xAxis: [
        { type: 'category', data: dates, boundaryGap: false, axisLine: { onZero: false } },
        {
          type: 'category',
          gridIndex: 1,
          data: dates,
          boundaryGap: false,
          axisLabel: { show: false },
        },
      ],
      yAxis: [
        { scale: true, splitArea: { show: true } },
        { gridIndex: 1, splitNumber: 2, axisLabel: { show: false } },
      ],
      dataZoom: [
        { type: 'inside', xAxisIndex: [0, 1], start: 60, end: 100 },
        { type: 'slider', xAxisIndex: [0, 1], bottom: 8, start: 60, end: 100 },
      ],
      series: [
        {
          name: 'K线',
          type: 'candlestick',
          data: candles,
          itemStyle: {
            color: UP,
            color0: DOWN,
            borderColor: UP,
            borderColor0: DOWN,
          },
        },
        {
          name: '成交量',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 1,
          data: volumes,
        },
      ],
    },
    true,
  )
}

function resize(): void {
  chart?.resize()
}

onMounted(() => {
  if (!container.value) return
  chart = echarts.init(container.value)
  render()
  window.addEventListener('resize', resize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resize)
  chart?.dispose()
  chart = null
})

watch(() => props.bars, render, { deep: false })
</script>

<template>
  <div ref="container" class="qs-chart" />
</template>

<style scoped>
.qs-chart {
  width: 100%;
  height: 460px;
}
</style>
