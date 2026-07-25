"""数据层的数据契约。

规范见 docs/04-数据规格.md。

关键约束：
- 每条 K 线必须显式携带 ``adjust`` 标记（红线 R4），混用口径会抛 ``AdjustMismatchError``。
- 标的状态（ST/停牌/涨跌幅限制）必须**按区间存储**，查询带 ``as_of``——
  只存当前状态就无法正确回测历史（见 docs/04-数据规格.md §2.3）。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal

from quantstock.infra.money import ZERO
from quantstock.infra.types import (
    Adjust,
    AssetType,
    Board,
    Exchange,
    Freq,
    Money,
    Settlement,
    Symbol,
    TradeDate,
)

__all__ = [
    "Bar",
    "Instrument",
    "InstrumentStatus",
    "SourceHealth",
    "UniverseMember",
]


@dataclass(frozen=True, slots=True)
class Bar:
    """一根 K 线。

    ``adjust`` 是必填字段而非可选标注——研究用后复权、下单用不复权，
    混用会让回测收益与真实成交价对不上（红线 R4）。
    """

    symbol: Symbol
    dt: dt.datetime
    """bar 结束时刻，tz-aware。"""
    trade_date: TradeDate
    freq: Freq
    adjust: Adjust
    open: Money
    high: Money
    low: Money
    close: Money
    volume: int
    """成交量（股）。"""
    amount: Money = ZERO
    """成交额（元）。"""
    pre_close: Money = ZERO
    """前收盘价。不复权口径下用于算涨跌幅与涨跌停价。"""
    limit_up: Money | None = None
    limit_down: Money | None = None
    is_suspended: bool = False
    adj_factor: Decimal = Decimal("1")
    source: str = ""

    @property
    def change_pct(self) -> Decimal:
        """相对前收盘的涨跌幅。

        Returns:
            涨跌幅（如 ``0.10`` 表示上涨 10%）；无前收盘价时返回 0。
        """
        if self.pre_close <= 0:
            return ZERO
        return (self.close - self.pre_close) / self.pre_close

    @property
    def is_limit_up(self) -> bool:
        """是否涨停。涨停无法买入（风控规则 A02）。"""
        return self.limit_up is not None and self.close >= self.limit_up

    @property
    def is_limit_down(self) -> bool:
        """是否跌停。跌停无法卖出。"""
        return self.limit_down is not None and self.close <= self.limit_down

    @property
    def is_tradable(self) -> bool:
        """当日是否可交易（未停牌且有成交）。"""
        return not self.is_suspended and self.volume > 0

    def validate(self) -> list[str]:
        """自校验，返回违反的规则编号（DQ01/DQ02）。

        Returns:
            违反的校验项列表；全部通过时为空。
        """
        problems: list[str] = []
        if not (self.low <= min(self.open, self.close) <= max(self.open, self.close) <= self.high):
            problems.append("DQ01")
        if (
            min(self.open, self.high, self.low, self.close) <= 0
            or self.volume < 0
            or self.amount < 0
        ):
            problems.append("DQ02")
        return problems


@dataclass(frozen=True, slots=True)
class Instrument:
    """标的基础信息。

    ``delist_date`` 非空即为已退市。**退市标的必须永久保留在数据湖中**——
    删除它们会造成幸存者偏差，让回测收益系统性高估
    （见 docs/08-差距分析与设计补强.md A1）。
    """

    symbol: Symbol
    name: str
    asset_type: AssetType
    exchange: Exchange
    board: Board
    list_date: TradeDate
    delist_date: TradeDate | None = None
    settlement: Settlement = Settlement.T1
    """交收制度。股票型 ETF 为 T+1，跨境/债券/黄金 ETF 为 T+0（风控 A13）。"""
    sw_l1: str = ""
    """申万一级行业。"""
    sw_l2: str = ""
    lot_size: int = 100
    min_order_qty: int = 100
    """最小申报数量。科创板为 200 股起，之后按 1 股递增。"""
    fund_scale: Money = ZERO
    """基金规模（元）。规模过小的 ETF 有清盘风险且流动性差。"""

    @property
    def is_delisted(self) -> bool:
        """是否已退市。"""
        return self.delist_date is not None

    @property
    def is_fund(self) -> bool:
        """是否为场内基金。"""
        return self.asset_type in {AssetType.ETF, AssetType.LOF}

    def is_listed_on(self, on: TradeDate) -> bool:
        """指定日期时该标的是否在市。

        用于构造 PIT universe——查询历史某日的股票池时，
        必须包含当时在市、如今已退市的标的。

        Args:
            on: 目标日期。

        Returns:
            当日在市则 True。
        """
        if on < self.list_date:
            return False
        return not (self.delist_date is not None and on >= self.delist_date)


@dataclass(frozen=True, slots=True)
class InstrumentStatus:
    """标的状态区间。

    **必须按区间存储**：回测查询历史某日的 ST 状态时，只存当前状态会给出错误答案
    （见 docs/04-数据规格.md §2.3）。
    """

    symbol: Symbol
    start_date: TradeDate
    end_date: TradeDate | None = None
    """区间结束日（不含）。None 表示至今仍然有效。"""
    is_st: bool = False
    is_suspended: bool = False
    is_delisting: bool = False
    """是否处于退市整理期。此期间禁止买入，持仓强制列入清仓建议（风控 A06）。"""

    def covers(self, on: TradeDate) -> bool:
        """该区间是否覆盖指定日期。

        Args:
            on: 目标日期。

        Returns:
            覆盖则 True。
        """
        if on < self.start_date:
            return False
        return self.end_date is None or on < self.end_date


@dataclass(frozen=True, slots=True)
class UniverseMember:
    """股票池成员的生效区间。

    指数成分变动必须记录 ``in_date`` / ``out_date``，
    否则用今天的成分股回测三年前会引入严重的前视偏差。
    """

    universe: str
    symbol: Symbol
    in_date: TradeDate
    out_date: TradeDate | None = None

    def covers(self, on: TradeDate) -> bool:
        """指定日期时该标的是否属于该股票池。

        Args:
            on: 目标日期。

        Returns:
            属于则 True。
        """
        if on < self.in_date:
            return False
        return self.out_date is None or on < self.out_date


@dataclass(frozen=True, slots=True)
class SourceHealth:
    """数据源健康状态。"""

    name: str
    ok: bool
    checked_at: dt.datetime
    message: str = ""
    latency_ms: float = 0.0
    consecutive_failures: int = 0


@dataclass(frozen=True, slots=True)
class FetchRequest:
    """行情拉取请求。"""

    symbols: tuple[Symbol, ...]
    start: TradeDate
    end: TradeDate
    freq: Freq = Freq.D
    adjust: Adjust = Adjust.NONE
    fields: tuple[str, ...] = field(default_factory=tuple)
