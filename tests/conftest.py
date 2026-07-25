"""共享测试夹具。

规范：测试必须确定性——时间通过 FrozenClock 注入，禁止依赖当前日期
（见 docs/01-开发规范.md 第九条）。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import pytest

from quantstock.infra.calendar import TradingCalendar
from quantstock.infra.clock import CST, FrozenClock, set_clock

# 2026-07 的真实交易日：7/1(三) 起至 7/31(五)，剔除周末。
# 该月无法定节假日，因此等于全部工作日。
_JULY_2026_TRADING_DAYS = [
    dt.date(2026, 7, d)
    for d in (
        1, 2, 3,
        6, 7, 8, 9, 10,
        13, 14, 15, 16, 17,
        20, 21, 22, 23, 24,
        27, 28, 29, 30, 31,
    )
]  # fmt: skip


@pytest.fixture
def trading_days() -> list[dt.date]:
    """2026 年 7 月的交易日列表。"""
    return list(_JULY_2026_TRADING_DAYS)


@pytest.fixture
def calendar(trading_days: list[dt.date]) -> TradingCalendar:
    """基于 2026 年 7 月交易日的日历。"""
    return TradingCalendar(trading_days)


@pytest.fixture
def frozen_clock() -> Iterator[FrozenClock]:
    """固定在 2026-07-24（周五）15:30 的时钟，并设为全局时钟。

    收盘后时点，当日日线已可用。
    """
    clock = FrozenClock(dt.datetime(2026, 7, 24, 15, 30, tzinfo=CST))
    set_clock(clock)
    yield clock
    set_clock(FrozenClock(dt.datetime(2026, 7, 24, 15, 30, tzinfo=CST)))


def at(hour: int, minute: int, *, day: int = 24) -> dt.datetime:
    """构造 2026-07-<day> 的指定时刻。

    Args:
        hour: 小时。
        minute: 分钟。
        day: 日期，默认 24（周五，交易日）。

    Returns:
        tz-aware 的时刻。
    """
    return dt.datetime(2026, 7, day, hour, minute, tzinfo=CST)
