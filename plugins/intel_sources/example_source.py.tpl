"""自定义情报源插件模板。复制为 my_source.py 后修改。"""

from __future__ import annotations

from datetime import datetime

from quantstock.infra.clock import now
from quantstock.intel.types import IntelDomain, IntelItem, SourceHealth, SourceTier


class MySource:
    """从自有渠道抓取情报。类名任意，需实现 NewsSource Protocol。"""

    name = "external:my_source"          # 必须以 external: 开头
    tier = SourceTier.USER
    domains = (IntelDomain.COMPANY, IntelDomain.INDUSTRY)

    def fetch(self, since: datetime) -> list[IntelItem]:
        """拉取 since 之后的新条目。

        Args:
            since: 上次成功抓取的时间点，tz-aware。

        Returns:
            IntelItem 列表。publish_at 与 url 必填（红线 I-R4）。
        """
        raise NotImplementedError

    def health_check(self) -> SourceHealth:
        """返回本源当前可用性，用于日报'情报健康'小节。"""
        return SourceHealth(name=self.name, ok=True, checked_at=now(), message="")
