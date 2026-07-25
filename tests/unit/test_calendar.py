"""交易日历与交易时段测试。

红线 R3：禁止用 weekday() < 5 近似交易日。这里覆盖周末、时段边界与 PIT 语义。
"""

from __future__ import annotations

import datetime as dt

import pytest

from quantstock.infra.calendar import (
    AFTER_HOURS_START_DATE,
    MarketSession,
    TradingCalendar,
    session_times_for,
)
from quantstock.infra.clock import CST
from quantstock.infra.errors import ConfigError
from tests.conftest import at


class TestConstruction:
    def test_empty_calendar__raises_with_actionable_message(self) -> None:
        with pytest.raises(ConfigError, match="交易日历为空"):
            TradingCalendar([])

    def test_half_day_not_in_trading_days__raises(self) -> None:
        with pytest.raises(ConfigError, match="half_days 含非交易日"):
            TradingCalendar([dt.date(2026, 7, 24)], half_days=[dt.date(2026, 7, 25)])

    def test_deduplicates_and_sorts(self) -> None:
        cal = TradingCalendar([dt.date(2026, 7, 27), dt.date(2026, 7, 24), dt.date(2026, 7, 24)])
        assert len(cal) == 2
        assert cal.first_day == dt.date(2026, 7, 24)
        assert cal.last_day == dt.date(2026, 7, 27)


class TestTradingDay:
    def test_is_trading_day__weekend__false(self, calendar: TradingCalendar) -> None:
        assert not calendar.is_trading_day(dt.date(2026, 7, 25))  # 周六
        assert not calendar.is_trading_day(dt.date(2026, 7, 26))  # 周日

    def test_is_trading_day__weekday__true(self, calendar: TradingCalendar) -> None:
        assert calendar.is_trading_day(dt.date(2026, 7, 24))

    def test_next_trading_day__skips_weekend(self, calendar: TradingCalendar) -> None:
        assert calendar.next_trading_day(dt.date(2026, 7, 24)) == dt.date(2026, 7, 27)

    def test_next_trading_day__from_weekend(self, calendar: TradingCalendar) -> None:
        assert calendar.next_trading_day(dt.date(2026, 7, 25)) == dt.date(2026, 7, 27)

    def test_prev_trading_day__skips_weekend(self, calendar: TradingCalendar) -> None:
        assert calendar.prev_trading_day(dt.date(2026, 7, 27)) == dt.date(2026, 7, 24)

    def test_next_trading_day__n_steps(self, calendar: TradingCalendar) -> None:
        assert calendar.next_trading_day(dt.date(2026, 7, 24), n=3) == dt.date(2026, 7, 29)

    @pytest.mark.parametrize("n", [0, -1])
    def test_next_trading_day__non_positive_n__raises(
        self, calendar: TradingCalendar, n: int
    ) -> None:
        with pytest.raises(ValueError, match="n 必须为正整数"):
            calendar.next_trading_day(dt.date(2026, 7, 24), n=n)

    def test_next_trading_day__beyond_calendar__raises(self, calendar: TradingCalendar) -> None:
        with pytest.raises(ConfigError, match="超出交易日历覆盖范围"):
            calendar.next_trading_day(dt.date(2026, 7, 31))

    def test_trading_days_between__inclusive(self, calendar: TradingCalendar) -> None:
        days = calendar.trading_days_between(dt.date(2026, 7, 24), dt.date(2026, 7, 28))
        assert days == (dt.date(2026, 7, 24), dt.date(2026, 7, 27), dt.date(2026, 7, 28))

    def test_trading_days_between__exclusive(self, calendar: TradingCalendar) -> None:
        days = calendar.trading_days_between(
            dt.date(2026, 7, 24), dt.date(2026, 7, 28), inclusive=False
        )
        assert days == (dt.date(2026, 7, 27),)

    def test_trading_days_between__reversed_range__empty(self, calendar: TradingCalendar) -> None:
        assert calendar.trading_days_between(dt.date(2026, 7, 28), dt.date(2026, 7, 24)) == ()

    def test_align_to_trading_day__backward_and_forward(self, calendar: TradingCalendar) -> None:
        saturday = dt.date(2026, 7, 25)
        assert calendar.align_to_trading_day(saturday) == dt.date(2026, 7, 24)
        assert calendar.align_to_trading_day(saturday, forward=True) == dt.date(2026, 7, 27)

    def test_align_to_trading_day__already_trading_day__unchanged(
        self, calendar: TradingCalendar
    ) -> None:
        assert calendar.align_to_trading_day(dt.date(2026, 7, 24)) == dt.date(2026, 7, 24)


class TestSessions:
    @pytest.mark.parametrize(
        ("hour", "minute", "expected"),
        [
            (9, 0, MarketSession.PRE_OPEN),
            (9, 15, MarketSession.OPENING_AUCTION),
            (9, 19, MarketSession.OPENING_AUCTION),
            (9, 20, MarketSession.OPENING_AUCTION_NO_CANCEL),
            (9, 24, MarketSession.OPENING_AUCTION_NO_CANCEL),
            (9, 25, MarketSession.AUCTION_GAP),
            (9, 30, MarketSession.CONTINUOUS_AM),
            (11, 29, MarketSession.CONTINUOUS_AM),
            (11, 30, MarketSession.LUNCH_BREAK),
            (12, 59, MarketSession.LUNCH_BREAK),
            (13, 0, MarketSession.CONTINUOUS_PM),
            (14, 56, MarketSession.CONTINUOUS_PM),
            (14, 57, MarketSession.CLOSING_AUCTION),
            (14, 59, MarketSession.CLOSING_AUCTION),
        ],
    )
    def test_session_at__boundaries(
        self, calendar: TradingCalendar, hour: int, minute: int, expected: MarketSession
    ) -> None:
        assert calendar.session_at(at(hour, minute)) == expected

    def test_session_at__non_trading_day__closed(self, calendar: TradingCalendar) -> None:
        saturday = dt.datetime(2026, 7, 25, 10, 0, tzinfo=CST)
        assert calendar.session_at(saturday) == MarketSession.CLOSED

    def test_session_at__naive_datetime__treated_as_shanghai(
        self, calendar: TradingCalendar
    ) -> None:
        naive = dt.datetime(2026, 7, 24, 10, 0)  # noqa: DTZ001 - 刻意构造 naive 输入
        assert calendar.session_at(naive) == MarketSession.CONTINUOUS_AM

    def test_is_continuous_trading(self, calendar: TradingCalendar) -> None:
        assert calendar.is_continuous_trading(at(10, 0))
        assert not calendar.is_continuous_trading(at(12, 0))

    def test_can_cancel__opening_auction_no_cancel_window(self, calendar: TradingCalendar) -> None:
        """9:20–9:25 不可撤单。"""
        assert calendar.can_cancel_order(at(9, 19))
        assert not calendar.can_cancel_order(at(9, 22))

    def test_can_submit__outside_session__false(self, calendar: TradingCalendar) -> None:
        assert not calendar.can_submit_order(at(8, 0))
        assert not calendar.can_submit_order(at(12, 0))


class TestAfterHoursTrading:
    """2026-07-06 起新增 15:05–15:30 盘后固定价格交易。"""

    def test_session_times__before_effective_date__no_after_hours(self) -> None:
        times = session_times_for(AFTER_HOURS_START_DATE - dt.timedelta(days=1))
        assert not times.has_after_hours

    def test_session_times__on_effective_date__has_after_hours(self) -> None:
        assert session_times_for(AFTER_HOURS_START_DATE).has_after_hours

    @pytest.mark.parametrize(
        ("hour", "minute", "expected"),
        [
            (15, 0, MarketSession.AFTER_HOURS_QUIET),
            (15, 4, MarketSession.AFTER_HOURS_QUIET),
            (15, 5, MarketSession.AFTER_HOURS_FIXED),
            (15, 29, MarketSession.AFTER_HOURS_FIXED),
            (15, 30, MarketSession.POST_CLOSE),
        ],
    )
    def test_session_at__after_hours_window(
        self, calendar: TradingCalendar, hour: int, minute: int, expected: MarketSession
    ) -> None:
        assert calendar.session_at(at(hour, minute)) == expected

    def test_quiet_period__can_submit_but_not_cancel(self, calendar: TradingCalendar) -> None:
        """15:00–15:05 可挂盘后单但不可撤单。"""
        moment = at(15, 2)
        assert calendar.can_submit_order(moment)
        assert not calendar.can_cancel_order(moment)

    def test_legacy_calendar__no_after_hours_session(self) -> None:
        """施行日之前的交易日 15:10 应已收市。"""
        legacy = TradingCalendar([dt.date(2026, 6, 30)])
        moment = dt.datetime(2026, 6, 30, 15, 10, tzinfo=CST)
        assert legacy.session_at(moment) == MarketSession.POST_CLOSE


class TestHalfDay:
    def test_half_day__closes_at_1130(self) -> None:
        cal = TradingCalendar([dt.date(2026, 7, 24)], half_days=[dt.date(2026, 7, 24)])
        assert cal.session_at(at(11, 0)) == MarketSession.CONTINUOUS_AM
        assert cal.session_at(at(14, 0)) == MarketSession.POST_CLOSE

    def test_half_day__close_price_final_after_1130(self) -> None:
        cal = TradingCalendar([dt.date(2026, 7, 24)], half_days=[dt.date(2026, 7, 24)])
        assert not cal.is_close_price_final(at(11, 29))
        assert cal.is_close_price_final(at(11, 30))


class TestPointInTime:
    """PIT 语义：盘中不得看到当日收盘价（见 08 号文档 C5）。"""

    @pytest.mark.parametrize(
        ("hour", "minute", "expected"),
        [(10, 0, False), (14, 58, False), (15, 0, True), (15, 30, True)],
    )
    def test_is_close_price_final(
        self, calendar: TradingCalendar, hour: int, minute: int, expected: bool
    ) -> None:
        assert calendar.is_close_price_final(at(hour, minute)) is expected

    def test_current_or_prev__intraday__returns_previous_day(
        self, calendar: TradingCalendar
    ) -> None:
        """盘中调用时当日日线尚未生成，只能用上一交易日。"""
        assert calendar.current_or_prev_trading_day(at(10, 0)) == dt.date(2026, 7, 23)

    def test_current_or_prev__after_close__returns_today(self, calendar: TradingCalendar) -> None:
        assert calendar.current_or_prev_trading_day(at(15, 30)) == dt.date(2026, 7, 24)

    def test_current_or_prev__weekend__returns_last_trading_day(
        self, calendar: TradingCalendar
    ) -> None:
        saturday = dt.datetime(2026, 7, 25, 10, 0, tzinfo=CST)
        assert calendar.current_or_prev_trading_day(saturday) == dt.date(2026, 7, 24)
