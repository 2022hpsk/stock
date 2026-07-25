"""时钟抽象。

红线 R3：时间必须 tz-aware 且统一 ``Asia/Shanghai``；禁止直接调用 ``datetime.now()``，
一律走本模块，以便测试用 :class:`FrozenClock` 注入固定时间。
"""

from __future__ import annotations

import datetime as dt
from typing import Final, Protocol, runtime_checkable
from zoneinfo import ZoneInfo

__all__ = [
    "CST",
    "Clock",
    "FrozenClock",
    "SystemClock",
    "ensure_aware",
    "get_clock",
    "now",
    "set_clock",
    "today",
]

CST: Final[ZoneInfo] = ZoneInfo("Asia/Shanghai")
"""交易所所在时区。全系统唯一时区，存储与计算都用它。"""


@runtime_checkable
class Clock(Protocol):
    """时钟接口。业务代码依赖本协议而非具体实现。"""

    def now(self) -> dt.datetime:
        """返回当前时刻（tz-aware，Asia/Shanghai）。"""
        ...


class SystemClock:
    """真实系统时钟。"""

    def now(self) -> dt.datetime:
        """返回当前系统时刻。

        Returns:
            tz-aware 的当前时刻。
        """
        return dt.datetime.now(tz=CST)


class FrozenClock:
    """固定时钟，用于测试。

    测试中的时间必须注入，禁止依赖 ``date.today()``
    （见 docs/01-开发规范.md 第九条）。
    """

    def __init__(self, moment: dt.datetime) -> None:
        """初始化。

        Args:
            moment: 固定的时刻。naive datetime 会被视为 Asia/Shanghai。
        """
        self._moment = ensure_aware(moment)

    def now(self) -> dt.datetime:
        """返回固定时刻。

        Returns:
            构造时给定的时刻。
        """
        return self._moment

    def advance(self, delta: dt.timedelta) -> None:
        """推进时钟。

        Args:
            delta: 推进的时间量。
        """
        self._moment += delta

    def set(self, moment: dt.datetime) -> None:
        """重设时刻。

        Args:
            moment: 新的时刻。
        """
        self._moment = ensure_aware(moment)


_clock: Clock = SystemClock()


def get_clock() -> Clock:
    """取当前进程使用的时钟。

    Returns:
        当前时钟实例。
    """
    return _clock


def set_clock(clock: Clock) -> None:
    """替换全局时钟。

    仅供测试与回测使用。生产代码不应调用。

    Args:
        clock: 新的时钟实例。
    """
    global _clock  # noqa: PLW0603 - 进程级单例，替换是本函数的唯一目的
    _clock = clock


def now() -> dt.datetime:
    """当前时刻（tz-aware，Asia/Shanghai）。

    Returns:
        当前时刻。
    """
    return _clock.now()


def today() -> dt.date:
    """当前自然日。

    注意这是**自然日**而非交易日。判断交易日请用 ``infra.calendar``。

    Returns:
        当前日期。
    """
    return _clock.now().date()


def ensure_aware(moment: dt.datetime) -> dt.datetime:
    """确保 datetime 带时区。

    naive datetime 一律按 Asia/Shanghai 解释——本系统只处理 A 股，
    出现 naive 时间通常是外部数据源没带时区，按交易所时区解释是正确的默认。

    Args:
        moment: 待处理时刻。

    Returns:
        带时区的时刻，已转换到 Asia/Shanghai。
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=CST)
    return moment.astimezone(CST)
