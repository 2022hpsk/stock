"""时钟测试。

红线 R3：时间必须 tz-aware 且统一 Asia/Shanghai；测试中时间必须可注入。
"""

from __future__ import annotations

import datetime as dt

from quantstock.infra.clock import (
    CST,
    Clock,
    FrozenClock,
    SystemClock,
    ensure_aware,
    get_clock,
    now,
    set_clock,
    today,
)


class TestSystemClock:
    def test_now__is_timezone_aware_shanghai(self) -> None:
        moment = SystemClock().now()
        assert moment.tzinfo is not None
        assert moment.utcoffset() == dt.timedelta(hours=8)

    def test_satisfies_protocol(self) -> None:
        assert isinstance(SystemClock(), Clock)


class TestFrozenClock:
    def test_now__returns_fixed_moment(self) -> None:
        fixed = dt.datetime(2026, 7, 24, 15, 30, tzinfo=CST)
        clock = FrozenClock(fixed)
        assert clock.now() == fixed
        assert clock.now() == fixed  # 多次调用结果一致

    def test_naive_input__interpreted_as_shanghai(self) -> None:
        clock = FrozenClock(dt.datetime(2026, 7, 24, 15, 30))  # noqa: DTZ001 - 刻意构造
        assert clock.now().utcoffset() == dt.timedelta(hours=8)

    def test_advance(self) -> None:
        clock = FrozenClock(dt.datetime(2026, 7, 24, 9, 30, tzinfo=CST))
        clock.advance(dt.timedelta(hours=2))
        assert clock.now() == dt.datetime(2026, 7, 24, 11, 30, tzinfo=CST)

    def test_set(self) -> None:
        clock = FrozenClock(dt.datetime(2026, 7, 24, 9, 30, tzinfo=CST))
        clock.set(dt.datetime(2026, 7, 27, 9, 30, tzinfo=CST))
        assert clock.now().date() == dt.date(2026, 7, 27)

    def test_satisfies_protocol(self) -> None:
        assert isinstance(FrozenClock(dt.datetime(2026, 1, 1, tzinfo=CST)), Clock)


class TestGlobalClock:
    def test_set_clock__now_and_today_follow(self, frozen_clock: FrozenClock) -> None:
        assert now() == frozen_clock.now()
        assert today() == dt.date(2026, 7, 24)
        assert get_clock() is frozen_clock

    def test_set_clock__restores(self) -> None:
        original = get_clock()
        try:
            set_clock(FrozenClock(dt.datetime(2020, 1, 1, tzinfo=CST)))
            assert today() == dt.date(2020, 1, 1)
        finally:
            set_clock(original)


class TestEnsureAware:
    def test_naive__gets_shanghai(self) -> None:
        naive = dt.datetime(2026, 7, 24, 10, 0)  # noqa: DTZ001 - 刻意构造
        assert ensure_aware(naive).utcoffset() == dt.timedelta(hours=8)

    def test_other_timezone__converted_to_shanghai(self) -> None:
        utc_moment = dt.datetime(2026, 7, 24, 2, 0, tzinfo=dt.UTC)
        converted = ensure_aware(utc_moment)
        assert converted.hour == 10  # UTC+8
        assert converted.utcoffset() == dt.timedelta(hours=8)

    def test_already_shanghai__unchanged(self) -> None:
        moment = dt.datetime(2026, 7, 24, 10, 0, tzinfo=CST)
        assert ensure_aware(moment) == moment
