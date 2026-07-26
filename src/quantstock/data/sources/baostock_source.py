"""BaoStock 行情数据源。

历史批量初始化的主力：免费、无需 token、**含已退市标的的完整历史**——
最后一条尤其关键，退市标的拿不到就必然产生幸存者偏差（docs/08 A1）。

**依赖是延迟导入的**。``baostock`` 没装时，其它数据源与全部离线功能照常可用，
只有真正调用本源时才报错并说清楚怎么装。把它做成模块级导入会让
"没装某个可选依赖"变成"整个程序起不来"。

BaoStock 的两个坑，都在本模块处理掉：

1. **它不抛异常**——所有错误通过返回对象的 ``error_code`` 传递，
   不检查就会拿到一个空结果集当成"这只股票没数据"；
2. **登录是全局状态**——``bs.login()`` 影响整个进程。这里用上下文管理器
   保证成对出现，避免残留的会话在下次调用时神秘失效。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from typing import Any

from quantstock.data.types import Bar, Instrument, SourceHealth
from quantstock.infra.clock import CST, now
from quantstock.infra.errors import DataSourceError
from quantstock.infra.logging import get_logger
from quantstock.infra.money import ZERO
from quantstock.infra.retry import RateLimiter
from quantstock.infra.types import (
    Adjust,
    AssetType,
    Board,
    Exchange,
    Freq,
    Symbol,
    TradeDate,
    make_symbol,
    split_symbol,
)

__all__ = ["BaoStockSource", "to_baostock_code", "to_symbol"]

_log = get_logger(__name__)

_SUCCESS = "0"

_ADJUST_FLAG = {
    # BaoStock 的 adjustflag：1=后复权 2=前复权 3=不复权
    Adjust.HFQ: "1",
    Adjust.QFQ: "2",
    Adjust.NONE: "3",
}

_DAILY_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,tradestatus"


def to_baostock_code(symbol: Symbol) -> str:
    """标准 Symbol → BaoStock 代码。

    ``600519.SH`` → ``sh.600519``。

    Args:
        symbol: 标准 Symbol。

    Returns:
        BaoStock 代码。

    Raises:
        DataSourceError: 北交所标的，BaoStock 不覆盖。
    """
    code, exchange = split_symbol(symbol)
    if exchange is Exchange.BJ:
        msg = "BaoStock 不覆盖北交所标的，请改用 AkShare"
        raise DataSourceError(msg, source="baostock", symbol=str(symbol))
    return f"{exchange.value.lower()}.{code}"


def to_symbol(raw: str) -> Symbol:
    """BaoStock 代码 → 标准 Symbol。

    Args:
        raw: 形如 ``sh.600519``。

    Returns:
        标准 Symbol。

    Raises:
        DataSourceError: 格式非法。
    """
    parts = raw.strip().split(".")
    expected_parts = 2
    if len(parts) != expected_parts:
        msg = f"无法解析 BaoStock 代码：{raw!r}"
        raise DataSourceError(msg, source="baostock")
    prefix, code = parts
    try:
        return make_symbol(code, Exchange(prefix.upper()))
    except ValueError as exc:
        msg = f"无法解析 BaoStock 代码：{raw!r}"
        raise DataSourceError(msg, source="baostock") from exc


def _decimal(raw: str, *, default: Decimal = ZERO) -> Decimal:
    """把 BaoStock 的字符串字段转成 Decimal。

    停牌日的价格字段是空串而不是 0，直接 ``Decimal("")`` 会抛异常。

    Args:
        raw: 原始字符串。
        default: 缺省值。

    Returns:
        Decimal 值。
    """
    text = (raw or "").strip()
    if not text:
        return default
    try:
        return Decimal(text)
    except InvalidOperation:
        return default


class BaoStockSource:
    """BaoStock 适配器。"""

    name = "baostock"

    def __init__(self, *, rate_limit_per_min: int = 120) -> None:
        """初始化。

        Args:
            rate_limit_per_min: 每分钟请求上限，礼貌抓取。
        """
        self._limiter = RateLimiter(rate_per_min=rate_limit_per_min)

    @staticmethod
    def _module() -> Any:  # noqa: ANN401 - 第三方库无类型存根
        """延迟导入 baostock。

        Returns:
            baostock 模块。

        Raises:
            DataSourceError: 未安装。
        """
        try:
            import baostock  # noqa: PLC0415 - 刻意延迟：没装它时其它源仍可用
        except ImportError as exc:
            msg = "未安装 baostock，请执行：uv pip install baostock"
            raise DataSourceError(msg, source="baostock") from exc
        return baostock

    @contextmanager
    def _session(self) -> Iterator[Any]:
        """登录/登出的成对保证。

        BaoStock 的登录是**进程级全局状态**，漏掉登出会让后续调用
        以难以定位的方式失败。

        Yields:
            baostock 模块。

        Raises:
            DataSourceError: 登录失败。
        """
        module = self._module()
        result = module.login()
        if result.error_code != _SUCCESS:
            msg = "BaoStock 登录失败"
            raise DataSourceError(
                msg, source="baostock", code=result.error_code, detail=result.error_msg
            )
        try:
            yield module
        finally:
            module.logout()

    @staticmethod
    def _rows(result: Any) -> Iterator[list[str]]:  # noqa: ANN401 - 第三方返回对象
        """遍历结果集并检查错误码。

        **必须检查 error_code**：BaoStock 出错时不抛异常，只返回一个
        error_code 非 0 的空结果集。不检查就会把"接口挂了"
        误读成"这只股票没数据"，然后静默地少算一大段历史。

        Args:
            result: BaoStock 返回对象。

        Yields:
            数据行。

        Raises:
            DataSourceError: 接口返回错误。
        """
        if result.error_code != _SUCCESS:
            msg = "BaoStock 接口返回错误"
            raise DataSourceError(
                msg, source="baostock", code=result.error_code, detail=result.error_msg
            )
        while result.next():
            yield result.get_row_data()

    def fetch_daily_bars(
        self,
        symbols: Sequence[Symbol],
        *,
        start: TradeDate,
        end: TradeDate,
        adjust: Adjust,
    ) -> list[Bar]:
        """拉取日线。

        Args:
            symbols: 标的列表。
            start: 起始日（含）。
            end: 结束日（含）。
            adjust: 复权口径。

        Returns:
            K 线列表。

        Raises:
            DataSourceError: 接口不可用或返回错误。
        """
        out: list[Bar] = []
        with self._session() as module:
            for symbol in symbols:
                self._limiter.acquire()
                result = module.query_history_k_data_plus(
                    to_baostock_code(symbol),
                    _DAILY_FIELDS,
                    start_date=start.isoformat(),
                    end_date=end.isoformat(),
                    frequency="d",
                    adjustflag=_ADJUST_FLAG[adjust],
                )
                out.extend(self._to_bars(result, symbol, adjust))
        _log.info("baostock_bars_fetched", symbols=len(symbols), bars=len(out))
        return out

    def _to_bars(self, result: Any, symbol: Symbol, adjust: Adjust) -> list[Bar]:  # noqa: ANN401
        """结果集 → Bar 列表。

        Args:
            result: BaoStock 返回对象。
            symbol: 标的。
            adjust: 复权口径。

        Returns:
            K 线列表。
        """
        out: list[Bar] = []
        for row in self._rows(result):
            trade_date = dt.date.fromisoformat(row[0])
            close = _decimal(row[5])
            if close <= 0:
                # 停牌日 BaoStock 返回空价格。**跳过而不是补 0**——
                # 0 会被因子计算读成"跌到 0"，制造出巨大的假跌幅
                continue
            out.append(
                Bar(
                    symbol=symbol,
                    dt=dt.datetime.combine(trade_date, dt.time(15, 0), tzinfo=CST),
                    trade_date=trade_date,
                    freq=Freq.D,
                    adjust=adjust,
                    open=_decimal(row[2], default=close),
                    high=_decimal(row[3], default=close),
                    low=_decimal(row[4], default=close),
                    close=close,
                    volume=int(_decimal(row[7])),
                    amount=_decimal(row[8]),
                    pre_close=_decimal(row[6]),
                    is_suspended=row[10].strip() == "0",
                )
            )
        return out

    def fetch_instruments(self) -> list[Instrument]:
        """拉取标的列表。

        用 ``query_stock_basic`` 而不是某一天的成分快照——前者**含已退市标的**，
        后者只有当日在市的，用它会直接制造幸存者偏差。

        Returns:
            标的列表。

        Raises:
            DataSourceError: 接口不可用。
        """
        out: list[Instrument] = []
        with self._session() as module:
            self._limiter.acquire()
            for row in self._rows(module.query_stock_basic()):
                try:
                    symbol = to_symbol(row[0])
                except DataSourceError:
                    continue  # 指数等非标的代码跳过

                out.append(
                    Instrument(
                        symbol=symbol,
                        name=row[1].strip(),
                        asset_type=AssetType.INDEX if row[4] == "2" else AssetType.STOCK,
                        exchange=split_symbol(symbol)[1],
                        board=_infer_board(symbol),
                        list_date=dt.date.fromisoformat(row[2])
                        if row[2]
                        else dt.date(1990, 12, 19),
                        # status=0 表示已退市。退市标的必须永久保留（docs/08 A1）
                        delist_date=dt.date.fromisoformat(row[3]) if row[3].strip() else None,
                    )
                )
        _log.info("baostock_instruments_fetched", count=len(out))
        return out

    def fetch_trading_days(self, *, start: TradeDate, end: TradeDate) -> list[TradeDate]:
        """拉取交易日历。

        Args:
            start: 起始日。
            end: 结束日。

        Returns:
            交易日列表。

        Raises:
            DataSourceError: 接口不可用。
        """
        with self._session() as module:
            self._limiter.acquire()
            result = module.query_trade_dates(
                start_date=start.isoformat(), end_date=end.isoformat()
            )
            return [dt.date.fromisoformat(row[0]) for row in self._rows(result) if row[1] == "1"]

    def health_check(self) -> SourceHealth:
        """探测可用性。

        Returns:
            健康度。失败时**不抛异常**——健康检查的用途就是报告故障，
            让它自己也炸掉就没法用于降级判断了。
        """
        started = now()
        try:
            with self._session():
                pass
        except DataSourceError as exc:
            return SourceHealth(name=self.name, ok=False, checked_at=now(), message=str(exc))
        latency = (now() - started).total_seconds() * 1000
        return SourceHealth(
            name=self.name, ok=True, checked_at=now(), message="登录成功", latency_ms=latency
        )


def _infer_board(symbol: Symbol) -> Board:
    """由代码前缀推断板块。

    Args:
        symbol: 标的。

    Returns:
        板块。
    """
    code, exchange = split_symbol(symbol)
    if code.startswith("688"):
        return Board.STAR
    if code.startswith("300"):
        return Board.GEM
    if exchange is Exchange.BJ:
        return Board.BSE
    return Board.MAIN
