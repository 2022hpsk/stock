"""交易日历与交易时段。

红线 R3：交易日/交易时段判断一律走本模块，**禁止用 ``weekday() < 5`` 近似**——
A 股有法定节假日调休，周末可能开市、工作日可能休市。

日历数据来自交易所，由 ``data`` 层采集后注入（见 docs/04-数据规格.md §2.4）。
本模块只负责日历逻辑，不负责采集。
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from quantstock.infra.clock import CST, ensure_aware
from quantstock.infra.errors import ConfigError

__all__ = [
    "AFTER_HOURS_START_DATE",
    "MarketSession",
    "SessionTimes",
    "TradingCalendar",
    "session_times_for",
]


class MarketSession(StrEnum):
    """盘中时段。

    用于判断当前能做什么：能否申报、能否撤单、是否已产生当日收盘价。
    """

    CLOSED = "closed"
    """非交易日或交易日的非交易时段。"""
    PRE_OPEN = "pre_open"
    """开市前，尚未进入集合竞价。"""
    OPENING_AUCTION = "opening_auction"
    """开盘集合竞价 9:15–9:25。其中 9:20–9:25 不可撤单。"""
    OPENING_AUCTION_NO_CANCEL = "opening_auction_no_cancel"
    """开盘集合竞价不可撤单段 9:20–9:25。"""
    AUCTION_GAP = "auction_gap"
    """9:25–9:30 撮合与静默。"""
    CONTINUOUS_AM = "continuous_am"
    """上午连续竞价 9:30–11:30。"""
    LUNCH_BREAK = "lunch_break"
    """午间休市 11:30–13:00。"""
    CONTINUOUS_PM = "continuous_pm"
    """下午连续竞价 13:00–14:57。"""
    CLOSING_AUCTION = "closing_auction"
    """收盘集合竞价 14:57–15:00。"""
    AFTER_HOURS_QUIET = "after_hours_quiet"
    """盘后静默期 15:00–15:05。可申报盘后固定价格单，但不可撤单。"""
    AFTER_HOURS_FIXED = "after_hours_fixed"
    """盘后固定价格交易 15:05–15:30。成交价固定为当日收盘价。"""
    POST_CLOSE = "post_close"
    """全部交易结束。"""


AFTER_HOURS_START_DATE: Final[dt.date] = dt.date(2026, 7, 6)
"""盘后固定价格交易（15:05–15:30）的施行日。

沪深全市场 A 股、ETF、LOF、REITs 适用；ST/*ST 与退市股不参与。
买单价不得高于收盘价、卖单价不得低于收盘价，无市价单，未成交当日作废不隔夜。
"""


@dataclass(frozen=True, slots=True)
class SessionTimes:
    """某一时期生效的交易时段定义。

    做成按生效日期的配置而非硬编码常量，因为交易时段会随监管调整变化
    （如 2026-07-06 新增盘后固定价格交易）。回测必须使用当时口径。
    """

    opening_auction_start: dt.time
    opening_auction_no_cancel: dt.time
    opening_auction_end: dt.time
    morning_start: dt.time
    morning_end: dt.time
    afternoon_start: dt.time
    closing_auction_start: dt.time
    closing_auction_end: dt.time
    after_hours_quiet_end: dt.time | None
    after_hours_end: dt.time | None

    @property
    def has_after_hours(self) -> bool:
        """该时期是否有盘后固定价格交易。"""
        return self.after_hours_end is not None


_BASE_SESSION: Final = SessionTimes(
    opening_auction_start=dt.time(9, 15),
    opening_auction_no_cancel=dt.time(9, 20),
    opening_auction_end=dt.time(9, 25),
    morning_start=dt.time(9, 30),
    morning_end=dt.time(11, 30),
    afternoon_start=dt.time(13, 0),
    closing_auction_start=dt.time(14, 57),
    closing_auction_end=dt.time(15, 0),
    after_hours_quiet_end=None,
    after_hours_end=None,
)

_AFTER_HOURS_SESSION: Final = SessionTimes(
    opening_auction_start=dt.time(9, 15),
    opening_auction_no_cancel=dt.time(9, 20),
    opening_auction_end=dt.time(9, 25),
    morning_start=dt.time(9, 30),
    morning_end=dt.time(11, 30),
    afternoon_start=dt.time(13, 0),
    closing_auction_start=dt.time(14, 57),
    closing_auction_end=dt.time(15, 0),
    after_hours_quiet_end=dt.time(15, 5),
    after_hours_end=dt.time(15, 30),
)


def session_times_for(on: dt.date) -> SessionTimes:
    """取指定日期生效的交易时段定义。

    Args:
        on: 目标日期。

    Returns:
        该日期适用的时段定义。
    """
    if on >= AFTER_HOURS_START_DATE:
        return _AFTER_HOURS_SESSION
    return _BASE_SESSION


class TradingCalendar:
    """交易日历。

    由交易日集合构造，所有查询都是对该集合的操作。半日市（如春节前最后半天，
    历史上极少）通过 ``half_days`` 单独标注，其收盘时间为 11:30。

    Example:
        >>> cal = TradingCalendar([dt.date(2026, 7, 24), dt.date(2026, 7, 27)])
        >>> cal.is_trading_day(dt.date(2026, 7, 25))  # 周六
        False
        >>> cal.next_trading_day(dt.date(2026, 7, 24))
        datetime.date(2026, 7, 27)
    """

    def __init__(
        self,
        trading_days: Iterable[dt.date],
        *,
        half_days: Iterable[dt.date] = (),
    ) -> None:
        """初始化。

        Args:
            trading_days: 交易日集合，无需预先排序，重复项会被去重。
            half_days: 半日市日期，必须是 ``trading_days`` 的子集。

        Raises:
            ConfigError: 交易日集合为空，或 half_days 含非交易日。
        """
        days = sorted(set(trading_days))
        if not days:
            msg = "交易日历为空。请先执行 `quantstock data update --calendar` 采集日历数据。"
            raise ConfigError(msg)
        self._days: Sequence[dt.date] = days
        self._day_set: frozenset[dt.date] = frozenset(days)

        half = frozenset(half_days)
        if unknown := half - self._day_set:
            msg = "half_days 含非交易日"
            raise ConfigError(msg, unknown=sorted(unknown)[:5])
        self._half_days = half

    # ------------------------------------------------------------------ 基础查询
    @property
    def first_day(self) -> dt.date:
        """日历覆盖的最早交易日。"""
        return self._days[0]

    @property
    def last_day(self) -> dt.date:
        """日历覆盖的最晚交易日。"""
        return self._days[-1]

    def __len__(self) -> int:
        """交易日总数。"""
        return len(self._days)

    def is_trading_day(self, on: dt.date) -> bool:
        """是否为交易日。

        Args:
            on: 待判断日期。

        Returns:
            是交易日则 True。
        """
        return on in self._day_set

    def is_half_day(self, on: dt.date) -> bool:
        """是否为半日市。

        Args:
            on: 待判断日期。

        Returns:
            是半日市则 True。
        """
        return on in self._half_days

    # ------------------------------------------------------------------ 日期推移
    def next_trading_day(self, on: dt.date, *, n: int = 1) -> dt.date:
        """向后第 n 个交易日。

        ``on`` 本身是否为交易日都可以：若 ``on`` 是交易日，返回其之后的第 n 个。

        Args:
            on: 基准日期。
            n: 向后推移的交易日数，必须为正。

        Returns:
            目标交易日。

        Raises:
            ValueError: n 不为正。
            ConfigError: 超出日历覆盖范围。
        """
        if n <= 0:
            msg = f"n 必须为正整数，收到 {n}"
            raise ValueError(msg)
        idx = bisect_right(self._days, on) + n - 1
        if idx >= len(self._days):
            msg = "超出交易日历覆盖范围，请扩充日历数据"
            raise ConfigError(msg, requested_from=on, n=n, calendar_last_day=self.last_day)
        return self._days[idx]

    def prev_trading_day(self, on: dt.date, *, n: int = 1) -> dt.date:
        """向前第 n 个交易日。

        Args:
            on: 基准日期。
            n: 向前推移的交易日数，必须为正。

        Returns:
            目标交易日。

        Raises:
            ValueError: n 不为正。
            ConfigError: 超出日历覆盖范围。
        """
        if n <= 0:
            msg = f"n 必须为正整数，收到 {n}"
            raise ValueError(msg)
        idx = bisect_left(self._days, on) - n
        if idx < 0:
            msg = "超出交易日历覆盖范围，请扩充日历数据"
            raise ConfigError(msg, requested_from=on, n=n, calendar_first_day=self.first_day)
        return self._days[idx]

    def n_trading_days_ago(self, n: int, *, from_date: dt.date) -> dt.date:
        """N 个交易日之前。

        语义与 :meth:`prev_trading_day` 相同，命名更贴近"回看 N 日"的因子场景。

        Args:
            n: 回看的交易日数。
            from_date: 基准日期。

        Returns:
            目标交易日。
        """
        return self.prev_trading_day(from_date, n=n)

    def trading_days_between(
        self, start: dt.date, end: dt.date, *, inclusive: bool = True
    ) -> tuple[dt.date, ...]:
        """区间内的交易日。

        Args:
            start: 起始日期。
            end: 结束日期。
            inclusive: True 时区间为闭区间 ``[start, end]``，False 时为 ``(start, end)``。

        Returns:
            升序排列的交易日元组；区间非法或无交易日时返回空元组。
        """
        if start > end:
            return ()
        if inclusive:
            lo = bisect_left(self._days, start)
            hi = bisect_right(self._days, end)
        else:
            lo = bisect_right(self._days, start)
            hi = bisect_left(self._days, end)
        return tuple(self._days[lo:hi])

    def count_trading_days(self, start: dt.date, end: dt.date) -> int:
        """区间内交易日数量（闭区间）。

        Args:
            start: 起始日期。
            end: 结束日期。

        Returns:
            交易日数量。
        """
        return len(self.trading_days_between(start, end))

    def align_to_trading_day(self, on: dt.date, *, forward: bool = False) -> dt.date:
        """把任意自然日对齐到交易日。

        Args:
            on: 待对齐日期。本身是交易日则原样返回。
            forward: True 时向后找最近交易日，False 时向前找。

        Returns:
            对齐后的交易日。

        Raises:
            ConfigError: 超出日历覆盖范围。
        """
        if on in self._day_set:
            return on
        return self.next_trading_day(on) if forward else self.prev_trading_day(on)

    # ------------------------------------------------------------------ 时段判断
    def session_at(self, moment: dt.datetime) -> MarketSession:  # noqa: C901, PLR0911 - 时段边界是一条有序的时间轴，顺序判断比拆函数更易核对
        """判断某时刻处于哪个交易时段。

        Args:
            moment: 待判断时刻。naive datetime 按 Asia/Shanghai 解释。

        Returns:
            所处时段；非交易日返回 :attr:`MarketSession.CLOSED`。
        """
        aware = ensure_aware(moment)
        on = aware.date()
        if not self.is_trading_day(on):
            return MarketSession.CLOSED

        times = session_times_for(on)
        clock = aware.time()

        if clock < times.opening_auction_start:
            return MarketSession.PRE_OPEN
        if clock < times.opening_auction_no_cancel:
            return MarketSession.OPENING_AUCTION
        if clock < times.opening_auction_end:
            return MarketSession.OPENING_AUCTION_NO_CANCEL
        if clock < times.morning_start:
            return MarketSession.AUCTION_GAP
        if clock < times.morning_end:
            return MarketSession.CONTINUOUS_AM

        # 半日市在 11:30 收市，无下午时段与收盘集合竞价
        if self.is_half_day(on):
            return MarketSession.POST_CLOSE

        if clock < times.afternoon_start:
            return MarketSession.LUNCH_BREAK
        if clock < times.closing_auction_start:
            return MarketSession.CONTINUOUS_PM
        if clock < times.closing_auction_end:
            return MarketSession.CLOSING_AUCTION

        if times.after_hours_quiet_end is not None and clock < times.after_hours_quiet_end:
            return MarketSession.AFTER_HOURS_QUIET
        if times.after_hours_end is not None and clock < times.after_hours_end:
            return MarketSession.AFTER_HOURS_FIXED
        return MarketSession.POST_CLOSE

    def is_continuous_trading(self, moment: dt.datetime) -> bool:
        """是否处于连续竞价时段（可正常限价成交）。

        Args:
            moment: 待判断时刻。

        Returns:
            处于上午或下午连续竞价则 True。
        """
        return self.session_at(moment) in _CONTINUOUS_SESSIONS

    def can_submit_order(self, moment: dt.datetime) -> bool:
        """该时刻是否可以申报委托。

        非交易时段提交的订单应标记为"次日待执行"（风控规则 A05）。

        Args:
            moment: 待判断时刻。

        Returns:
            可申报则 True。
        """
        return self.session_at(moment) in _SUBMITTABLE_SESSIONS

    def can_cancel_order(self, moment: dt.datetime) -> bool:
        """该时刻是否可以撤单。

        9:20–9:25 与盘后静默期不可撤单。

        Args:
            moment: 待判断时刻。

        Returns:
            可撤单则 True。
        """
        return self.session_at(moment) in _CANCELLABLE_SESSIONS

    def is_close_price_final(self, moment: dt.datetime) -> bool:
        """当日收盘价是否已确定。

        PIT 关键：盘中调用时当日日线尚未生成，不得让策略看到"当日收盘价"
        （见 docs/08-差距分析与设计补强.md C5）。

        Args:
            moment: 待判断时刻。

        Returns:
            收盘价已确定则 True。
        """
        aware = ensure_aware(moment)
        on = aware.date()
        if not self.is_trading_day(on):
            # 非交易日：上一交易日的收盘价当然已确定
            return True
        times = session_times_for(on)
        end = times.morning_end if self.is_half_day(on) else times.closing_auction_end
        return aware.time() >= end

    def current_or_prev_trading_day(self, moment: dt.datetime) -> dt.date:
        """取"当前可用"的交易日。

        若当日是交易日且收盘价已确定，返回当日；否则返回上一交易日。
        用于回答"现在能用到哪一天的日线数据"。

        Args:
            moment: 基准时刻。

        Returns:
            可用的交易日。
        """
        aware = ensure_aware(moment)
        on = aware.date()
        if self.is_trading_day(on) and self.is_close_price_final(aware):
            return on
        # prev_trading_day 对交易日返回其前一交易日，对非交易日返回之前最近的交易日，
        # 两种情形都正是这里要的结果。
        return self.prev_trading_day(on)


_CONTINUOUS_SESSIONS: Final[frozenset[MarketSession]] = frozenset(
    {MarketSession.CONTINUOUS_AM, MarketSession.CONTINUOUS_PM}
)

_SUBMITTABLE_SESSIONS: Final[frozenset[MarketSession]] = frozenset(
    {
        MarketSession.OPENING_AUCTION,
        MarketSession.OPENING_AUCTION_NO_CANCEL,
        MarketSession.CONTINUOUS_AM,
        MarketSession.CONTINUOUS_PM,
        MarketSession.CLOSING_AUCTION,
        MarketSession.AFTER_HOURS_QUIET,
        MarketSession.AFTER_HOURS_FIXED,
    }
)

_CANCELLABLE_SESSIONS: Final[frozenset[MarketSession]] = frozenset(
    {
        MarketSession.OPENING_AUCTION,
        MarketSession.CONTINUOUS_AM,
        MarketSession.CONTINUOUS_PM,
        MarketSession.AFTER_HOURS_FIXED,
    }
)


def _utc_offset_sanity() -> None:
    """启动时校验时区数据可用（缺 tzdata 会导致时间全错）。"""
    probe = dt.datetime(2026, 1, 1, tzinfo=CST)
    if probe.utcoffset() != dt.timedelta(hours=8):
        msg = "Asia/Shanghai 时区数据异常，请检查系统 tzdata 或安装 tzdata 包"
        raise ConfigError(msg, actual_offset=str(probe.utcoffset()))


_utc_offset_sanity()
