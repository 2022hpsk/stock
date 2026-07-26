"""通用 RSS/Atom 情报源。

用户可在 ``config/intel_sources.yaml`` 里声明任意站点。这是**最不容易失效**
的一类源：RSS 是稳定了二十年的标准，不像各家网站的私有接口那样随时改字段。

两个 PIT 相关的细节，都比看起来重要：

1. **发布时间必须来自 feed 本身**（``published_parsed``），不能用抓取时刻顶替。
   用抓取时间冒充发布时间等于把未来消息塞进过去（红线 I-R5）；
2. **没有发布时间的条目直接丢弃**，而不是"就当是现在发的"。少一条情报
   远好过一条时间错乱的情报——后者会在回测里静默地制造未来函数。
"""

from __future__ import annotations

import datetime as dt
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from quantstock.infra.clock import CST, now
from quantstock.infra.errors import IntelError
from quantstock.infra.logging import get_logger
from quantstock.infra.retry import RateLimiter
from quantstock.intel.dedup import content_hash
from quantstock.intel.external import make_item_id
from quantstock.intel.types import IntelDomain, IntelItem, SourceHealth, SourceTier

__all__ = ["RssFeed", "RssSource"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RssFeed:
    """一个 RSS 订阅源的声明。"""

    name: str
    url: str
    domain: IntelDomain = IntelDomain.MARKET
    tier: SourceTier = SourceTier.MEDIA


class RssSource:
    """通用 RSS/Atom 适配器。"""

    def __init__(
        self,
        feeds: Sequence[RssFeed],
        *,
        source_name: str = "rss",
        rate_limit_per_min: int = 30,
        timeout_sec: float = 15.0,
    ) -> None:
        """初始化。

        Args:
            feeds: 订阅源列表。
            source_name: 本源在注册表里的标识。
            rate_limit_per_min: 每分钟请求上限（红线 I-R6 礼貌抓取）。
            timeout_sec: 单次请求超时。
        """
        self._feeds = tuple(feeds)
        self._name = source_name
        self._limiter = RateLimiter(rate_per_min=rate_limit_per_min)
        self._timeout = timeout_sec

    @property
    def name(self) -> str:
        """源标识。"""
        return self._name

    @property
    def domains(self) -> tuple[IntelDomain, ...]:
        """覆盖的情报域。"""
        return tuple(dict.fromkeys(f.domain for f in self._feeds))

    @staticmethod
    def _module() -> Any:  # noqa: ANN401 - 第三方库无类型存根
        """延迟导入 feedparser。

        Returns:
            feedparser 模块。

        Raises:
            IntelError: 未安装。
        """
        try:
            import feedparser  # noqa: PLC0415 - 刻意延迟导入
        except ImportError as exc:
            msg = "未安装 feedparser，请执行：uv pip install feedparser"
            raise IntelError(msg, source="rss") from exc
        return feedparser

    def fetch(self, since: dt.datetime) -> list[IntelItem]:
        """拉取增量情报。

        Args:
            since: 起始时间，tz-aware。

        Returns:
            情报条目。单个 feed 失败只丢那一个，不影响其余。
        """
        module = self._module()
        moment = now()
        out: list[IntelItem] = []

        for feed in self._feeds:
            self._limiter.acquire()
            try:
                parsed = module.parse(feed.url)
            except Exception as exc:
                _log.warning("rss_fetch_failed", feed=feed.name, error=str(exc))
                continue
            out.extend(self._to_items(parsed, feed, since=since, fetched_at=moment))

        _log.info("rss_fetched", feeds=len(self._feeds), items=len(out))
        return out

    def _to_items(
        self,
        parsed: Any,  # noqa: ANN401 - feedparser 返回结构
        feed: RssFeed,
        *,
        since: dt.datetime,
        fetched_at: dt.datetime,
    ) -> list[IntelItem]:
        """把 feed 条目转成 ``IntelItem``。

        Args:
            parsed: feedparser 解析结果。
            feed: 订阅源声明。
            since: 起始时间。
            fetched_at: 抓取时刻。

        Returns:
            情报条目。
        """
        out: list[IntelItem] = []
        for entry in getattr(parsed, "entries", []):
            published = _published_at(entry)
            if published is None:
                # 没有发布时间的条目直接丢弃。用抓取时刻顶替等于把
                # 未来消息塞进过去，在回测里会静默制造未来函数
                _log.warning("rss_entry_without_date", feed=feed.name)
                continue
            if published < since:
                continue

            title = str(getattr(entry, "title", "")).strip()
            body = str(getattr(entry, "summary", "") or getattr(entry, "description", "")).strip()
            if not title and not body:
                continue

            url = str(getattr(entry, "link", "")).strip()
            digest = content_hash(title, body)
            source = f"{self._name}:{feed.name}"
            out.append(
                IntelItem(
                    item_id=make_item_id(source, url, digest),
                    source=source,
                    source_tier=feed.tier,
                    domain=feed.domain,
                    domain_declared=True,
                    publish_at=published,
                    fetched_at=fetched_at,
                    title=title or body[:80],
                    content_hash=digest,
                    body=body,
                    url=url,
                )
            )
        return out

    def health_check(self) -> SourceHealth:
        """探测可用性。

        Returns:
            健康度。**只要有一个 feed 可达就算可用**——订阅源本来就是
            互相独立的，一个挂了不该让整个源被判死。
        """
        if not self._feeds:
            return SourceHealth(source=self._name, ok=False, error="未配置任何订阅源")
        try:
            module = self._module()
        except IntelError as exc:
            return SourceHealth(source=self._name, ok=False, error=str(exc))

        alive = 0
        for feed in self._feeds:
            self._limiter.acquire()
            try:
                parsed = module.parse(feed.url)
            except Exception as exc:
                _log.warning("rss_probe_failed", feed=feed.name, error=str(exc))
                continue
            if getattr(parsed, "entries", None):
                alive += 1

        return SourceHealth(
            source=self._name,
            ok=alive > 0,
            fetched=alive,
            error="" if alive else "全部订阅源均不可达",
        )


def _published_at(entry: Any) -> dt.datetime | None:  # noqa: ANN401 - feedparser 条目
    """取条目的发布时间。

    优先 ``published_parsed``，回退 ``updated_parsed``。两者都没有时返回 None，
    由调用方丢弃该条——这是 PIT 的底线（红线 I-R5）。

    Args:
        entry: feed 条目。

    Returns:
        tz-aware 时间；无法确定时 None。
    """
    for field in ("published_parsed", "updated_parsed"):
        parsed = getattr(entry, field, None)
        if parsed is None:
            continue
        try:
            # feedparser 给的是 UTC 的 struct_time，先按 UTC 解释再转本地时区，
            # 直接当本地时间会整体偏移 8 小时
            stamp = dt.datetime.fromtimestamp(time.mktime(parsed), tz=dt.UTC)
        except (TypeError, ValueError, OverflowError):
            continue
        return stamp.astimezone(CST)
    return None
