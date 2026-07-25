# CLAUDE.md — AI 协作指引

## 开始任何工作前必读
1. [docs/01-开发规范.md](docs/01-开发规范.md) — **强制规范，红线 R1–R7 不可违反**
2. [docs/02-系统架构.md](docs/02-系统架构.md) — 分层与依赖方向
3. [docs/03-功能规格.md](docs/03-功能规格.md) — 功能编号与验收标准
4. 涉及情报模块时另读 [docs/07-信息情报模块.md](docs/07-信息情报模块.md)（红线 I-R1–I-R6）

## 不可违反的红线速查
| | |
|---|---|
| R1 | 金额用 `Decimal`，禁止 `float` |
| R2 | 禁止未来函数，财务数据必须 PIT（按 `ann_date`） |
| R3 | 时间必须 tz-aware（Asia/Shanghai），禁止 `datetime.now()`，走 `infra.clock` |
| R4 | 复权口径必须显式：研究用 `hfq`，下单/展示用 `none` |
| R5 | 下单必须经过 `RiskEngine` + 人工确认；默认 `PaperBroker`，真实通道需 `--live` |
| R6 | 每条建议必须可追溯可复现（数据指纹 + 策略版本 + 参数哈希） |
| R7 | 严禁提交密钥、资金账号、真实持仓 |
| I-R1 | 情报不得单独触发买入；只能解释、有界软调节、或单向风险否决 |
| I-R4 | 进入解释的情报必须带原文链接与发布时间 |

## 工作流
```bash
make install     # 首次
make check       # 每次提交前必跑（lint + mypy + 分层检查 + 测试）
```
- 分支策略：**仅 `main` 一个分支**，trunk-based。完成一组有意义的变更即提交推送。
- 提交信息用 Conventional Commits；`strategy` / `risk` 类改动的提交正文必须附回测对比结论。
- 改架构或规格时，**同一提交内**更新对应文档。

## 常见陷阱
- 新增依赖前先确认是否已有等价能力；重量级依赖需在提交正文说明必要性。
- 写测试时时间必须注入（`FrozenClock`），禁止依赖 `date.today()`。
- 数据源适配器测试用录制 fixture 回放，禁止 CI 打真实网络（标记 `@pytest.mark.network`）。
- 任何涉及真实资金的默认值取最保守一侧。
- 修 bug 先写复现失败的回归测试。
