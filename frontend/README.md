# frontend — Web UI 前端

Vue 3 + TypeScript + Vite + Element Plus + ECharts。详见 [../docs/09-可视化界面规格.md](../docs/09-可视化界面规格.md)。

## 终端用户

**不需要安装 Node。** 发版时前端已预编译，`uv sync` 后直接：

```bash
quantstock ui --open
```

## 开发者（需要改前端时）

```bash
make ui-dev     # Vite 热更新，代理 API 到本地 FastAPI
make ui-build   # 构建到 src/quantstock/web/static/，随 Python 包分发
```

## 约束

- **不引用任何外部 CDN**：所有 JS/CSS/字体本地打包，保证离线可用且不泄漏访问信息。
- 涨跌颜色默认遵循 A 股习惯（红涨绿跌），可在设置中切换。
- 页面只调用 `/api/*`，不实现任何业务逻辑——所有计算都在后端 `services/` 层。
- 表格数据量大的页面（因子表、全市场行情）必须用虚拟滚动。
