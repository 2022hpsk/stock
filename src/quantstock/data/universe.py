"""Point-in-Time 股票池（防幸存者偏差）。

规范见 docs/08-差距分析与设计补强.md A1。

**这是回测失真最常见也最致命的来源**：若数据湖只保存当前在市的标的，
退市股的归零被整体抹掉，回测收益会被系统性高估。

因此：

1. 退市标的**永久保留**，靠 ``delist_date`` 标记而非物理删除；
2. 股票池查询**强制带 ``as_of``**，返回当日实际成分；
3. DQ11 自动检测——任一历史年份的股票池中若已退市标的数为 0，判定存在幸存者偏差。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from quantstock.data.types import Instrument, InstrumentStatus, UniverseMember
from quantstock.infra.errors import DataQualityError
from quantstock.infra.types import Symbol, TradeDate

__all__ = ["SurvivorshipReport", "UniverseRegistry", "check_survivorship_bias"]


@dataclass(frozen=True, slots=True)
class SurvivorshipReport:
    """幸存者偏差检测结果（DQ11）。"""

    as_of: TradeDate
    total: int
    delisted_later: int
    """当日在市、但在检测基准日之前已退市的标的数。"""
    passed: bool
    message: str


class UniverseRegistry:
    """股票池与标的状态的 PIT 查询。

    所有查询方法都**强制要求 ``as_of``** ——没有默认值，
    调用方必须显式说明"站在哪一天看"。
    """

    def __init__(
        self,
        instruments: Iterable[Instrument],
        *,
        members: Iterable[UniverseMember] = (),
        statuses: Iterable[InstrumentStatus] = (),
    ) -> None:
        """初始化。

        Args:
            instruments: 全部标的（**含已退市**）。
            members: 股票池成员区间。
            statuses: 标的状态区间。
        """
        self._instruments: dict[Symbol, Instrument] = {i.symbol: i for i in instruments}
        self._members: list[UniverseMember] = list(members)
        self._statuses: dict[Symbol, list[InstrumentStatus]] = {}
        for status in statuses:
            self._statuses.setdefault(status.symbol, []).append(status)

    # ------------------------------------------------------------------ 标的
    def instrument(self, symbol: Symbol) -> Instrument | None:
        """取标的基础信息（无论是否已退市）。

        Args:
            symbol: 标的。

        Returns:
            标的信息；不存在时返回 None。
        """
        return self._instruments.get(symbol)

    def listed_symbols(self, as_of: TradeDate) -> tuple[Symbol, ...]:
        """指定日期在市的全部标的。

        Args:
            as_of: 基准日。

        Returns:
            当日在市的标的，升序排列。**包含那些如今已退市但当日仍在市的标的**。
        """
        return tuple(
            sorted(sym for sym, inst in self._instruments.items() if inst.is_listed_on(as_of))
        )

    # ------------------------------------------------------------------ 股票池
    def members(self, universe: str, *, as_of: TradeDate) -> tuple[Symbol, ...]:
        """指定日期某股票池的成分。

        Args:
            universe: 股票池名，如 ``hs300``。
            as_of: 基准日。**必填**——用今天的成分股回测三年前会引入前视偏差。

        Returns:
            当日成分，升序排列。
        """
        return tuple(
            sorted(m.symbol for m in self._members if m.universe == universe and m.covers(as_of))
        )

    # ------------------------------------------------------------------ 状态
    def status(self, symbol: Symbol, *, as_of: TradeDate) -> InstrumentStatus | None:
        """指定日期某标的的状态。

        Args:
            symbol: 标的。
            as_of: 基准日。

        Returns:
            覆盖该日期的状态区间；无记录时返回 None。
        """
        for status in self._statuses.get(symbol, ()):
            if status.covers(as_of):
                return status
        return None

    def is_st(self, symbol: Symbol, *, as_of: TradeDate) -> bool:
        """指定日期该标的是否为风险警示股票。

        涨跌幅限制依赖此判断，且 ST 状态**随时间变化**——
        必须按历史区间查询而非用当前状态（见 docs/04-数据规格.md §2.3）。

        Args:
            symbol: 标的。
            as_of: 基准日。

        Returns:
            当日为 ST 则 True。
        """
        status = self.status(symbol, as_of=as_of)
        return status is not None and status.is_st

    def is_tradable(self, symbol: Symbol, *, as_of: TradeDate) -> bool:
        """指定日期该标的是否可交易。

        停牌、退市整理期、尚未上市或已退市均不可交易。

        Args:
            symbol: 标的。
            as_of: 基准日。

        Returns:
            可交易则 True。
        """
        instrument = self._instruments.get(symbol)
        if instrument is None or not instrument.is_listed_on(as_of):
            return False
        status = self.status(symbol, as_of=as_of)
        if status is None:
            return True
        return not (status.is_suspended or status.is_delisting)

    def filter_tradable(self, symbols: Iterable[Symbol], *, as_of: TradeDate) -> tuple[Symbol, ...]:
        """筛出指定日期可交易的标的。

        Args:
            symbols: 候选标的。
            as_of: 基准日。

        Returns:
            可交易的标的，保持输入顺序。
        """
        return tuple(s for s in symbols if self.is_tradable(s, as_of=as_of))

    # ------------------------------------------------------------------ 质量校验
    def survivorship_report(self, *, as_of: TradeDate, today: TradeDate) -> SurvivorshipReport:
        """生成某历史日期的幸存者偏差检测报告（DQ11）。

        Args:
            as_of: 被检测的历史日期。
            today: 当前日期，用于判断"如今是否已退市"。

        Returns:
            检测报告。
        """
        listed = self.listed_symbols(as_of)
        delisted_later = sum(
            1
            for sym in listed
            if (inst := self._instruments[sym]).delist_date is not None
            and inst.delist_date <= today
        )
        passed = delisted_later > 0 or not listed
        message = (
            f"{as_of} 在市 {len(listed)} 只，其中 {delisted_later} 只已于 {today} 前退市"
            if passed
            else (
                f"{as_of} 在市 {len(listed)} 只，但**没有任何一只已退市**——"
                "数据湖很可能只保留了当前在市标的，存在幸存者偏差，回测收益会被系统性高估"
            )
        )
        return SurvivorshipReport(
            as_of=as_of,
            total=len(listed),
            delisted_later=delisted_later,
            passed=passed,
            message=message,
        )


def check_survivorship_bias(
    registry: UniverseRegistry,
    *,
    sample_dates: Sequence[TradeDate],
    today: TradeDate,
    raise_on_fail: bool = True,
) -> list[SurvivorshipReport]:
    """对多个历史日期做幸存者偏差检测（DQ11）。

    应在回测启动前调用。检测不通过时**拒绝运行回测**——
    带幸存者偏差的回测结果毫无意义，跑出来只会误导决策。

    Args:
        registry: 股票池注册表。
        sample_dates: 抽样检测的历史日期，通常取每年一个。
        today: 当前日期。
        raise_on_fail: 检测不通过时是否抛异常。

    Returns:
        各日期的检测报告。

    Raises:
        DataQualityError: ``raise_on_fail=True`` 且存在不通过的日期。
    """
    reports = [registry.survivorship_report(as_of=d, today=today) for d in sample_dates]
    failed = [r for r in reports if not r.passed]
    if failed and raise_on_fail:
        msg = "DQ11 幸存者偏差检测未通过，拒绝运行回测"
        raise DataQualityError(
            msg,
            rule="DQ11",
            failed_dates=[str(r.as_of) for r in failed],
            detail=[r.message for r in failed],
        )
    return reports
