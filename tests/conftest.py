"""共享测试夹具。

规范：测试必须确定性——时间通过 FrozenClock 注入，禁止依赖当前日期
（见 docs/01-开发规范.md 第九条）。

同一条规范还要求**测试禁止打真实网络**。光靠自觉不够：只要哪个测试走到了
默认装配的适配器（而不是显式注入的假源），它就会真的去请求外网。这类错误
不会表现成"测试失败"，而是表现成**测试变慢**——限速、超时、重试层层叠加，
一次 ``make check`` 从半分钟拖到十几分钟，还会随着别人的网络状况随机失败。
所以这里直接在 socket 层拦死，未标 ``network`` 的用例一联网就当场报错。
"""

from __future__ import annotations

import datetime as dt
import os
import socket
from collections.abc import Iterator
from urllib.parse import urlparse

import pytest

from quantstock.infra.calendar import TradingCalendar
from quantstock.infra.clock import CST, FrozenClock, set_clock

_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", ""})

_PROXY_ENV_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY")


class NetworkAccessInTestError(RuntimeError):
    """测试期间发生了真实网络访问。"""


def _proxy_endpoints() -> frozenset[tuple[str, int]]:
    """从环境变量解析出出网代理的地址。

    **不解析代理就等于没设防**：容器里代理常常挂在 ``127.0.0.1``，
    此时所有外网请求在 socket 层看到的目标都是本机地址，
    只放行 loopback 的守卫会全部漏过去。

    Returns:
        代理的 (host, port) 集合。
    """
    out: set[tuple[str, int]] = set()
    for var in _PROXY_ENV_VARS:
        raw = os.environ.get(var, "").strip()
        if not raw:
            continue
        parsed = urlparse(raw if "://" in raw else f"http://{raw}")
        if parsed.hostname and parsed.port:
            out.add((parsed.hostname, parsed.port))
    return frozenset(out)


def _blocked(address: object) -> NetworkAccessInTestError:
    """构造拦截异常。

    Args:
        address: 目标地址。

    Returns:
        异常实例。
    """
    return NetworkAccessInTestError(
        f"测试禁止访问真实网络（目标 {address!r}）。\n"
        "适配器测试请注入假模块或录制 fixture；确实需要联网的用例请标记 "
        "@pytest.mark.network（CI 不执行）。"
    )


def _is_allowed(address: object, proxies: frozenset[tuple[str, int]]) -> bool:
    """判断该地址是否允许连接。

    Args:
        address: socket 目标地址。
        proxies: 出网代理地址集合。

    Returns:
        允许则 True。
    """
    if not isinstance(address, tuple) or len(address) < 2:
        return False
    host, port = str(address[0]), address[1]
    if isinstance(port, int) and (host, port) in proxies:
        return False  # 走代理就是在出网，哪怕代理挂在 127.0.0.1
    return host in _LOCAL_HOSTS


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """拦截一切真实网络连接。

    本机地址放行（``TestClient`` 与本地 uvicorn 用例需要），但**出网代理
    的地址不放行**，哪怕它就在 ``127.0.0.1`` 上。

    Args:
        request: pytest 请求对象，用于识别 ``network`` 标记。
        monkeypatch: 打补丁工具，测试结束自动还原。
    """
    if request.node.get_closest_marker("network") is not None:
        return

    proxies = _proxy_endpoints()
    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def guard_connect(self: socket.socket, address: object) -> object:
        if not _is_allowed(address, proxies):
            raise _blocked(address)
        return real_connect(self, address)  # type: ignore[arg-type]

    def guard_create(address: tuple[str, int], *args: object, **kwargs: object) -> object:
        if not _is_allowed(address, proxies):
            raise _blocked(address)
        return real_create(address, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(socket.socket, "connect", guard_connect)
    monkeypatch.setattr(socket, "create_connection", guard_create)


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
