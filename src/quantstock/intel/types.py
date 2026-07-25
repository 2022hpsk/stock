"""情报层的数据契约。

规范见 docs/07-信息情报模块.md 第四节。

**贯穿全模块的一条约束**：``publish_at`` 是 PIT 关键字段（红线 I-R5）。
回测里"这条消息当时看得到吗"完全由它决定，因此它必须 tz-aware、
必须来自源方而不是抓取时刻，缺失时宁可丢弃该条也不要用 ``fetched_at`` 顶替——
用抓取时间冒充发布时间，等于把未来消息塞进过去。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

from quantstock.infra.types import Symbol, TradeDate

MAX_IMPORTANCE = 100
SENTIMENT_DISPUTE_GAP = 0.5
"""规则分与 LLM 分差超过它就在日报里标注"情绪判定存在分歧"。"""

__all__ = [
    "CalendarEvent",
    "DomainDigest",
    "EventType",
    "IntelDigest",
    "IntelDomain",
    "IntelItem",
    "PortfolioAlert",
    "SourceHealth",
    "SourceTier",
]


class IntelDomain(StrEnum):
    """情报域。每日分域搜集、分域摘要。"""

    MACRO = "macro"
    """宏观：经济数据、央行操作、利率汇率、大宗商品。影响总仓位中枢。"""
    POLICY = "policy"
    """政策监管：证监会/交易所新规、产业政策、财税政策。"""
    INDUSTRY = "industry"
    """行业：景气度、产业链价格、供需、订单。"""
    COMPANY = "company"
    """个股：公告、业绩、并购、减持、诉讼、研报。"""
    MARKET = "market"
    """市场：资金面、涨跌停家数、情绪指标、龙虎榜。"""
    OVERSEAS = "overseas"
    """海外：美股港股、美联储、地缘风险。"""
    CALENDAR = "calendar"
    """财经日历：未来事件时间表，用于日报"未来一周关注"。"""


class SourceTier(IntEnum):
    """来源层级。

    用 ``IntEnum`` 是因为去重时要按层级取主条目、打分时要按层级加权，
    顺序本身有意义：交易所公告永远压过媒体转述。
    """

    SOCIAL = 1
    """社交媒体、论坛。"""
    USER = 2
    """人工导入。可信度由用户自负，但 importance 有上限（防单条人工输入压过全部量化信号）。"""
    RESEARCH = 3
    """券商研报。"""
    MEDIA = 4
    """财经媒体。"""
    OFFICIAL = 5
    """交易所/监管机构官方披露。公告类以此为裁判源。"""


class EventType(StrEnum):
    """事件类型。

    分类以**规则/关键词优先**——确定性高、可回测、可解释；
    LLM 只在规则未命中时兜底，且结果会标注 ``classifier="llm"``。
    """

    # ---- 业绩类 ----
    EARNINGS_FORECAST = "earnings_forecast"
    EARNINGS_REPORT = "earnings_report"
    EARNINGS_SURPRISE = "earnings_surprise"
    # ---- 股权类 ----
    SHAREHOLDER_INCREASE = "shareholder_increase"
    SHAREHOLDER_REDUCE = "shareholder_reduce"
    BUYBACK = "buyback"
    PLACEMENT = "placement"
    UNLOCK = "unlock"
    # ---- 经营类 ----
    MAJOR_CONTRACT = "major_contract"
    MA = "ma"
    ASSET_SALE = "asset_sale"
    CAPACITY = "capacity"
    # ---- 风险类 ----
    REGULATORY_PROBE = "regulatory_probe"
    AUDIT_QUALIFIED = "audit_qualified"
    LITIGATION = "litigation"
    CONTROL_CHANGE = "control_change"
    DELISTING_RISK = "delisting_risk"
    GUARANTEE_RISK = "guarantee_risk"
    PLEDGE_RISK = "pledge_risk"
    # ---- 交易类 ----
    SUSPENSION = "suspension"
    ST_CHANGE = "st_change"
    INDEX_ADJUST = "index_adjust"
    # ---- 分红类 ----
    DIVIDEND = "dividend"
    SPLIT = "split"
    # ---- 行业政策 ----
    POLICY_POSITIVE = "policy_positive"
    POLICY_NEGATIVE = "policy_negative"
    PRICE_MOVE = "price_move"
    # ---- 宏观 ----
    MACRO_DATA = "macro_data"
    MONETARY = "monetary"
    FISCAL = "fiscal"
    GEOPOLITICS = "geopolitics"

    @property
    def is_risk_event(self) -> bool:
        """是否为风险类事件。风险类可单向否决买入（红线 I-R2）。"""
        return self in _RISK_EVENTS


_RISK_EVENTS = frozenset(
    {
        EventType.REGULATORY_PROBE,
        EventType.AUDIT_QUALIFIED,
        EventType.DELISTING_RISK,
        EventType.CONTROL_CHANGE,
        EventType.GUARANTEE_RISK,
    }
)
"""可触发黑名单的风险事件（docs/07 第六节 通路 1）。

刻意**不含** ``LITIGATION`` 与 ``PLEDGE_RISK``：诉讼在 A 股极为常见且多为小额，
质押风险则常年存在，把它们纳入硬否决会让黑名单失去意义。
它们仍会拉低情绪分并进入解释，只是不单独触发禁买。
"""


@dataclass(frozen=True, slots=True)
class IntelItem:
    """一条标准化的情报。

    ``item_id`` 由 ``source + url + content_hash`` 派生，天然幂等——
    同一条消息重复抓取、重复推送都落到同一个 id。
    """

    item_id: str
    source: str
    source_tier: SourceTier
    domain: IntelDomain
    publish_at: dt.datetime
    fetched_at: dt.datetime
    title: str
    content_hash: str
    body: str = ""
    url: str = ""
    symbols: tuple[Symbol, ...] = ()
    industries: tuple[str, ...] = ()
    themes: tuple[str, ...] = ()
    event_type: EventType | None = None
    classifier: str = "rule"
    """分类器来源：``rule`` 或 ``llm:<model>``。LLM 结果必须可识别（红线 I-R3）。"""
    importance: int = 0
    sentiment: float = 0.0
    sentiment_source: str = "rule"
    llm_sentiment: float | None = None
    """LLM 情绪分。与规则分同时保留，分歧 > 0.5 时日报标注"情绪判定存在分歧"。"""
    duplicates: tuple[str, ...] = ()
    """被合并的重复条目 id。多源印证会提高 importance。"""
    match_evidence: tuple[str, ...] = ()
    """实体链接命中的关键词，让匹配结果可解释。"""
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验 PIT 关键字段与出处。

        Raises:
            ValueError: 时间非 tz-aware，或 importance/sentiment 越界。
        """
        if self.publish_at.tzinfo is None:
            msg = "publish_at 必须 tz-aware（红线 R3、I-R5）"
            raise ValueError(msg)
        if self.fetched_at.tzinfo is None:
            msg = "fetched_at 必须 tz-aware（红线 R3）"
            raise ValueError(msg)
        if not 0 <= self.importance <= MAX_IMPORTANCE:
            msg = f"importance 必须在 0~100，收到 {self.importance}"
            raise ValueError(msg)
        if not -1.0 <= self.sentiment <= 1.0:
            msg = f"sentiment 必须在 -1~1，收到 {self.sentiment}"
            raise ValueError(msg)

    @property
    def trade_date(self) -> TradeDate:
        """归属交易日，按发布时间的自然日。分区键。"""
        return self.publish_at.date()

    @property
    def is_llm_generated(self) -> bool:
        """摘要或情绪是否出自 LLM。日报中必须标注（红线 I-R3）。"""
        return self.classifier.startswith("llm") or self.sentiment_source.startswith("llm")

    @property
    def sentiment_disputed(self) -> bool:
        """规则分与 LLM 分是否分歧过大。"""
        if self.llm_sentiment is None:
            return False
        return abs(self.llm_sentiment - self.sentiment) > SENTIMENT_DISPUTE_GAP

    def visible_at(self, as_of: dt.datetime) -> bool:
        """该条情报在给定时点是否可见（红线 I-R5）。

        回测里每一次读取情报都必须过这一关。

        Args:
            as_of: 决策时点，必须 tz-aware。

        Returns:
            发布时间不晚于 ``as_of`` 则 True。

        Raises:
            ValueError: ``as_of`` 非 tz-aware。
        """
        if as_of.tzinfo is None:
            msg = "as_of 必须 tz-aware，否则可见性判断会因时区而错（红线 I-R5）"
            raise ValueError(msg)
        return self.publish_at <= as_of

    def cite(self) -> str:
        """渲染成带出处的引用行（红线 I-R4）。

        Returns:
            形如 ``[domain] 标题 — 来源 · 发布时间 · 链接`` 的字符串。
        """
        stamp = self.publish_at.strftime("%m-%d %H:%M")
        tail = f" · {self.url}" if self.url else ""
        return f"[{self.domain.value}] {self.title} — {self.source} · {stamp}{tail}"


@dataclass(frozen=True, slots=True)
class SourceHealth:
    """单个源的健康度。

    情报缺失**不阻断建议生成**（与行情不同），但必须在日报里说清楚
    "今日哪个域没有情报"——分不清"没查到"和"查了没有"是最糟的状态。
    """

    source: str
    ok: bool
    fetched: int = 0
    error: str = ""
    latency_ms: int = 0
    last_success_at: dt.datetime | None = None

    @property
    def message(self) -> str:
        """人类可读说明。"""
        if self.ok:
            return f"{self.source}：{self.fetched} 条"
        return f"{self.source}：失败（{self.error or '未知原因'}）"


@dataclass(frozen=True, slots=True)
class CalendarEvent:
    """财经日历事件。用于日报"未来一周关注"。"""

    event_date: TradeDate
    title: str
    domain: IntelDomain
    symbols: tuple[Symbol, ...] = ()
    importance: int = 0
    source: str = ""
    url: str = ""


@dataclass(frozen=True, slots=True)
class PortfolioAlert:
    """命中当前持仓的事件。日报里优先级最高。"""

    symbol: Symbol
    item: IntelItem
    severity: str
    """``critical`` / ``warning`` / ``info``。"""
    action_hint: str = ""
    """建议动作提示。仅为提示，不构成下单指令（红线 I-R1）。"""


@dataclass(frozen=True, slots=True)
class DomainDigest:
    """单个域的摘要。"""

    domain: IntelDomain
    highlights: tuple[str, ...]
    items: tuple[IntelItem, ...]
    net_sentiment: float
    symbols: tuple[Symbol, ...] = ()
    llm_generated: bool = False
    """摘要是否由 LLM 生成。为 true 时日报打 🤖 标（红线 I-R3）。"""

    @property
    def count(self) -> int:
        """条目数。"""
        return len(self.items)


@dataclass(frozen=True, slots=True)
class IntelDigest:
    """一次分域摘要（盘前 08:30 / 盘后 18:30）。"""

    trade_date: TradeDate
    generated_at: dt.datetime
    session: str
    """``pre`` 盘前 / ``post`` 盘后。"""
    by_domain: dict[IntelDomain, DomainDigest]
    top_items: tuple[IntelItem, ...] = ()
    portfolio_alerts: tuple[PortfolioAlert, ...] = ()
    watchlist_hits: tuple[IntelItem, ...] = ()
    upcoming: tuple[CalendarEvent, ...] = ()
    coverage: dict[str, SourceHealth] = field(default_factory=dict)

    @property
    def missing_domains(self) -> tuple[IntelDomain, ...]:
        """今日无情报的域。日报"情报健康"小节要列出来。"""
        return tuple(
            d for d in IntelDomain if d not in self.by_domain or not self.by_domain[d].count
        )

    @property
    def failed_sources(self) -> tuple[str, ...]:
        """采集失败的源。"""
        return tuple(name for name, health in self.coverage.items() if not health.ok)

    @property
    def total_items(self) -> int:
        """总条目数。"""
        return sum(d.count for d in self.by_domain.values())
