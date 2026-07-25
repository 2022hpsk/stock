# 自定义情报源插件

把实现了 `NewsSource` Protocol 的 `.py` 文件放在本目录，系统启动时自动发现并注册。
本目录下的 `*.py` 已被 `.gitignore` 忽略（个人渠道不入库），模板文件 `example_source.py.tpl` 除外。

用法：复制模板 → 重命名为 `my_source.py` → 实现 `fetch()` → 在 `config/intel.yaml` 中确认
`external.plugins.enabled: true`。

约束（见 [docs/07-信息情报模块.md](../../docs/07-信息情报模块.md)）：
- `source` 命名必须以 `external:` 开头；
- 必须填 `publish_at`（tz-aware）与 `url`，否则该条目无法进入建议解释（红线 I-R4）；
- 抓取须遵守目标站点条款与限流（红线 I-R6）。
