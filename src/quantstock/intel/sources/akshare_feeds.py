"""财联社电报与东方财富新闻（经 AkShare）。

财联社电报是 A 股快讯的主力来源：时效性好、覆盖面广。东方财富补个股新闻。

与行情源同样的处理原则：**接口与列名变动频繁**，所以列名映射写得宽容，
升级 AkShare 后某个字段改名退化成"该字段缺失"，而不是整条链路崩掉。

PIT 上的关键点与 RSS 源一致：**发布时间必须来自源方**。这两个接口都给
``发布时间`` / ``发布日期`` 字段；取不到时丢弃该条而不是用抓取时刻顶替。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any

from quantstock.infra.clock import CST, now
from quantstock.infra.errors import IntelError
from quantstock.infra.logging import get_logger
from quantstock.infra.retry import RateLimiter
from quantstock.infra.types import Symbol, split_symbol
from quantstock.intel.dedup import content_hash
from quantstock.intel.external import make_item_id
from quantstock.intel.types import IntelDomain, IntelItem, SourceHealth, SourceTier

__all__ = ["ClsTelegraphSource", "EastMoneySource"]

_log = get_logger(__name__)

_TITLE_KEYS = ("标题", "title", "新闻标题")
_BODY_KEYS = ("内容", "content", "新闻内容", "摘要")
_TIME_KEYS = ("发布时间", "发布日期", "时间", "datetime", "publish_time", "新闻发布时间")
_URL_KEYS = ("链接", "url", "新闻链接")

_CLS_ENDPOINTS = ("stock_info_global_cls", "stock_telegraph_cls")
"""财联社电报的接口候选名，新名在前。

``stock_telegraph_cls`` 在 AkShare 1.18 已并入 ``stock_info_global_cls``，
两个都列上，装了哪个版本都能跑。
"""

_EM_GLOBAL_ENDPOINTS = ("stock_info_global_em",)
_EM_NEWS_ENDPOINTS = ("stock_news_em",)


def _pick(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    """按候选键取值。

    Args:
        row: 数据行。
        keys: 候选键。

    Returns:
        命中的值；无命中时空串。
    """
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() and str(value).strip().lower() != "nan":
            return str(value).strip()
    return ""


def _to_datetime(raw: str) -> dt.datetime | None:
    """解析发布时间。

    Args:
        raw: 原始字符串。

    Returns:
        tz-aware 时间；无法解析时 None——由调用方丢弃该条（红线 I-R5）。
    """
    text = raw.strip().replace("/", "-")
    if not text:
        return None
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%H:%M:%S", "%H:%M"):
        try:
            parsed = dt.datetime.strptime(text, pattern)  # noqa: DTZ007 - 下面立刻补时区
        except ValueError:
            continue
        if pattern.startswith("%H"):
            # 电报接口有时只给"时:分"。补上今天的日期——这是**当日快讯**的语义，
            # 补昨天或留空都会让 PIT 过滤算错
            today = now().date()
            parsed = dt.datetime.combine(today, parsed.time())
        return parsed.replace(tzinfo=CST)
    return None


class _AkShareFeedBase:
    """AkShare 系情报源的公共部分。"""

    def __init__(self, *, rate_limit_per_min: int = 30) -> None:
        """初始化。

        Args:
            rate_limit_per_min: 每分钟请求上限（红线 I-R6）。
        """
        self._limiter = RateLimiter(rate_per_min=rate_limit_per_min)

    @staticmethod
    def _module() -> Any:  # noqa: ANN401 - 第三方库无类型存根
        """延迟导入 akshare。

        Returns:
            akshare 模块。

        Raises:
            IntelError: 未安装。
        """
        try:
            import akshare  # noqa: PLC0415 - 刻意延迟导入
        except ImportError as exc:
            msg = "未安装 akshare，请执行：uv pip install akshare"
            raise IntelError(msg, source="akshare") from exc
        return akshare

    def _build(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        source: str,
        tier: SourceTier,
        domain: IntelDomain,
        since: dt.datetime,
        symbols: tuple[Symbol, ...] = (),
    ) -> list[IntelItem]:
        """把数据行转成情报条目。

        Args:
            rows: 数据行。
            source: 来源标识。
            tier: 来源层级。
            domain: 情报域。
            since: 起始时间。
            symbols: 已知的关联标的（个股新闻接口是按标的查的，所以能直接给）。

        Returns:
            情报条目。
        """
        moment = now()
        out: list[IntelItem] = []
        for row in rows:
            published = _to_datetime(_pick(row, _TIME_KEYS))
            if published is None:
                _log.warning("akshare_news_without_date", source=source)
                continue
            if published < since:
                continue

            title = _pick(row, _TITLE_KEYS)
            body = _pick(row, _BODY_KEYS)
            if not title and not body:
                continue

            url = _pick(row, _URL_KEYS)
            digest = content_hash(title, body)
            out.append(
                IntelItem(
                    item_id=make_item_id(source, url, digest),
                    source=source,
                    source_tier=tier,
                    domain=domain,
                    domain_declared=True,
                    publish_at=published,
                    fetched_at=moment,
                    title=title or body[:80],
                    content_hash=digest,
                    body=body,
                    url=url,
                    symbols=symbols,
                )
            )
        return out

    def _endpoint(self, candidates: Sequence[str]) -> Any:  # noqa: ANN401 - 第三方无存根
        """按候选名取接口。

        AkShare 不只改列名，**接口名本身也会改**（``stock_telegraph_cls``
        在 1.18 已并入 ``stock_info_global_cls``）。按候选顺序取第一个存在的，
        新旧版本都能跑；全都不在时返回 None，由调用方降级成"本次无情报"。

        Args:
            candidates: 候选函数名，新名在前。

        Returns:
            可调用对象；都不存在时 None。
        """
        module = self._module()
        for name in candidates:
            func = getattr(module, name, None)
            if callable(func):
                return func
        _log.warning("akshare_endpoint_missing", candidates=list(candidates))
        return None

    def _probe(self, candidates: Sequence[str], source: str) -> SourceHealth:
        """探测接口是否可达。

        走与 ``fetch`` 相同的候选名解析，否则升级 AkShare 改了接口名时
        健康检查会说"可用"而实际采集一条都拿不到。

        Args:
            candidates: AkShare 函数候选名。
            source: 源标识。

        Returns:
            健康度。
        """
        try:
            endpoint = self._endpoint(candidates)
        except IntelError as exc:
            return SourceHealth(source=source, ok=False, error=str(exc))
        if endpoint is None:
            return SourceHealth(
                source=source, ok=False, error=f"AkShare 无可用接口：{'/'.join(candidates)}"
            )

        try:
            frame = endpoint()
        except Exception as exc:
            return SourceHealth(source=source, ok=False, error=str(exc))
        count = 0 if frame is None else len(frame)
        return SourceHealth(source=source, ok=count > 0, fetched=count)


class ClsTelegraphSource(_AkShareFeedBase):
    """财联社电报。"""

    @property
    def name(self) -> str:
        """源标识。"""
        return "cls"

    @property
    def domains(self) -> tuple[IntelDomain, ...]:
        """覆盖的情报域。"""
        return (IntelDomain.MARKET, IntelDomain.COMPANY, IntelDomain.POLICY)

    def fetch(self, since: dt.datetime) -> list[IntelItem]:
        """拉取电报。

        Args:
            since: 起始时间。

        Returns:
            情报条目。接口失败时返回空列表并记 WARNING——
            情报缺失只降级为"缺少证据"，不阻断建议生成。
        """
        endpoint = self._endpoint(_CLS_ENDPOINTS)
        if endpoint is None:
            return []

        self._limiter.acquire()
        try:
            frame = endpoint(symbol="全部")
        except Exception as exc:
            _log.warning("cls_fetch_failed", error=str(exc))
            return []

        if frame is None or len(frame) == 0:
            return []
        return self._build(
            frame.to_dict("records"),
            source=self.name,
            tier=SourceTier.MEDIA,
            domain=IntelDomain.MARKET,
            since=since,
        )

    def health_check(self) -> SourceHealth:
        """探测可用性。

        Returns:
            健康度。
        """
        return self._probe(_CLS_ENDPOINTS, self.name)


class EastMoneySource(_AkShareFeedBase):
    """东方财富个股新闻与全球快讯。"""

    def __init__(self, *, symbols: Sequence[Symbol] = (), rate_limit_per_min: int = 30) -> None:
        """初始化。

        Args:
            symbols: 要跟踪的个股。为空时只拉全球快讯——
                个股新闻接口是**按标的逐个查**的，全市场遍历会被限流封掉。
            rate_limit_per_min: 每分钟请求上限。
        """
        super().__init__(rate_limit_per_min=rate_limit_per_min)
        self._symbols = tuple(symbols)

    @property
    def name(self) -> str:
        """源标识。"""
        return "eastmoney"

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        """逐只跟踪的标的。空元组表示只拉全球快讯，不采集个股新闻。"""
        return self._symbols

    @property
    def domains(self) -> tuple[IntelDomain, ...]:
        """覆盖的情报域。"""
        return (IntelDomain.COMPANY, IntelDomain.MARKET)

    def fetch(self, since: dt.datetime) -> list[IntelItem]:
        """拉取新闻。

        Args:
            since: 起始时间。

        Returns:
            情报条目。
        """
        out: list[IntelItem] = []

        if (globals_endpoint := self._endpoint(_EM_GLOBAL_ENDPOINTS)) is not None:
            self._limiter.acquire()
            try:
                frame = globals_endpoint()
            except Exception as exc:
                _log.warning("eastmoney_global_failed", error=str(exc))
            else:
                if frame is not None and len(frame) > 0:
                    out.extend(
                        self._build(
                            frame.to_dict("records"),
                            source=self.name,
                            tier=SourceTier.MEDIA,
                            domain=IntelDomain.MARKET,
                            since=since,
                        )
                    )

        news_endpoint = self._endpoint(_EM_NEWS_ENDPOINTS) if self._symbols else None
        for symbol in self._symbols:
            if news_endpoint is None:
                break
            self._limiter.acquire()
            code, _ = split_symbol(symbol)
            try:
                frame = news_endpoint(symbol=code)
            except Exception as exc:
                _log.warning("eastmoney_news_failed", symbol=str(symbol), error=str(exc))
                continue
            if frame is None or len(frame) == 0:
                continue
            out.extend(
                self._build(
                    frame.to_dict("records"),
                    source=self.name,
                    tier=SourceTier.MEDIA,
                    domain=IntelDomain.COMPANY,
                    since=since,
                    symbols=(symbol,),
                )
            )

        _log.info("eastmoney_fetched", symbols=len(self._symbols), items=len(out))
        return out

    def health_check(self) -> SourceHealth:
        """探测可用性。

        Returns:
            健康度。
        """
        return self._probe(_EM_GLOBAL_ENDPOINTS, self.name)
