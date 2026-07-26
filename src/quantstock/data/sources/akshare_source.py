"""AkShare 行情数据源。

用于增量更新与补缺：接口覆盖面广（含 ETF、北交所），但**接口与列名变动频繁**——
``pyproject.toml`` 里锁了次版本（``akshare>=1.14,<1.20``）就是为了这个
（docs/08 C3）。

因此本模块的列名映射刻意写得宽容：同一个字段准备多个候选名，
命中任一即可。升级 AkShare 后若某个字段改名，退化成"该字段缺失"
而不是整条链路崩掉。

依赖同样是延迟导入。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
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
    parse_symbol,
    split_symbol,
)

__all__ = ["AkShareSource"]

_log = get_logger(__name__)

_ADJUST_ARG = {Adjust.NONE: "", Adjust.QFQ: "qfq", Adjust.HFQ: "hfq"}

# 一个字段准备多个候选列名：AkShare 改列名是常态，宽容映射让升级不至于全盘崩
_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "date": ("日期", "date", "时间"),
    "open": ("开盘", "open"),
    "high": ("最高", "high"),
    "low": ("最低", "low"),
    "close": ("收盘", "close"),
    "volume": ("成交量", "volume", "vol"),
    "amount": ("成交额", "amount"),
}

ETF_PREFIXES_SH = ("510", "511", "512", "513", "515", "516", "517", "518", "588")
ETF_PREFIXES_SZ = ("159",)


def _cell(row: Mapping[str, Any], field: str) -> Any:  # noqa: ANN401 - pandas 行元素
    """按候选列名取值。

    Args:
        row: 数据行。
        field: 逻辑字段名。

    Returns:
        命中的值；无命中时 None。
    """
    for name in _COLUMNS.get(field, (field,)):
        if name in row:
            return row[name]
    return None


def _decimal(value: Any, *, default: Decimal = ZERO) -> Decimal:  # noqa: ANN401 - pandas 标量
    """转 Decimal。

    经 ``str()`` 而不是直接 ``Decimal(float)``——后者会把 pandas 的
    float64 精度噪声原样带进金额（红线 R1）。

    Args:
        value: 原始值。
        default: 缺省值。

    Returns:
        Decimal 值。
    """
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-"}:
        return default
    try:
        return Decimal(text)
    except InvalidOperation:
        return default


def _to_date(value: Any) -> TradeDate:  # noqa: ANN401 - pandas 可能给 str/Timestamp/date
    """转日期。

    Args:
        value: 原始值。

    Returns:
        日期。

    Raises:
        DataSourceError: 无法解析。
    """
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip().replace("/", "-")
    if text.isdigit() and len(text) == 8:  # noqa: PLR2004 - YYYYMMDD
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError as exc:
        msg = f"无法解析日期：{value!r}"
        raise DataSourceError(msg, source="akshare") from exc


class AkShareSource:
    """AkShare 适配器。"""

    name = "akshare"

    def __init__(self, *, rate_limit_per_min: int = 60) -> None:
        """初始化。

        Args:
            rate_limit_per_min: 每分钟请求上限。AkShare 背后是各家网站的公开接口，
                抓太快会被封（红线 I-R6 同理）。
        """
        self._limiter = RateLimiter(rate_per_min=rate_limit_per_min)

    @staticmethod
    def _module() -> Any:  # noqa: ANN401 - 第三方库无类型存根
        """延迟导入 akshare。

        Returns:
            akshare 模块。

        Raises:
            DataSourceError: 未安装。
        """
        try:
            import akshare  # noqa: PLC0415 - 刻意延迟导入
        except ImportError as exc:
            msg = "未安装 akshare，请执行：uv pip install akshare"
            raise DataSourceError(msg, source="akshare") from exc
        return akshare

    def fetch_daily_bars(
        self,
        symbols: Sequence[Symbol],
        *,
        start: TradeDate,
        end: TradeDate,
        adjust: Adjust,
    ) -> list[Bar]:
        """拉取日线。

        股票走 ``stock_zh_a_hist``，ETF 走 ``fund_etf_hist_em``——
        两者的接口与列名不同，混用会拿到空结果。

        Args:
            symbols: 标的列表。
            start: 起始日（含）。
            end: 结束日（含）。
            adjust: 复权口径。

        Returns:
            K 线列表。

        Raises:
            DataSourceError: 接口调用失败。
        """
        module = self._module()
        out: list[Bar] = []
        for symbol in symbols:
            self._limiter.acquire()
            code, _ = split_symbol(symbol)
            is_etf = _is_etf(symbol)
            try:
                frame = (
                    module.fund_etf_hist_em(
                        symbol=code,
                        period="daily",
                        start_date=start.strftime("%Y%m%d"),
                        end_date=end.strftime("%Y%m%d"),
                        adjust=_ADJUST_ARG[adjust],
                    )
                    if is_etf
                    else module.stock_zh_a_hist(
                        symbol=code,
                        period="daily",
                        start_date=start.strftime("%Y%m%d"),
                        end_date=end.strftime("%Y%m%d"),
                        adjust=_ADJUST_ARG[adjust],
                    )
                )
            except Exception as exc:
                # 单只失败不该拖垮整批：记下来，缺口交给 DQ 校验统一报告
                _log.warning("akshare_fetch_failed", symbol=str(symbol), error=str(exc))
                continue

            out.extend(self._to_bars(frame, symbol, adjust))
        _log.info("akshare_bars_fetched", symbols=len(symbols), bars=len(out))
        return out

    @staticmethod
    def _to_bars(frame: Any, symbol: Symbol, adjust: Adjust) -> list[Bar]:  # noqa: ANN401
        """DataFrame → Bar 列表。

        Args:
            frame: AkShare 返回的 DataFrame。
            symbol: 标的。
            adjust: 复权口径。

        Returns:
            K 线列表。
        """
        if frame is None or len(frame) == 0:
            return []

        out: list[Bar] = []
        previous: Decimal = ZERO
        for record in frame.to_dict("records"):
            close = _decimal(_cell(record, "close"))
            if close <= 0:
                continue
            trade_date = _to_date(_cell(record, "date"))
            out.append(
                Bar(
                    symbol=symbol,
                    dt=dt.datetime.combine(trade_date, dt.time(15, 0), tzinfo=CST),
                    trade_date=trade_date,
                    freq=Freq.D,
                    adjust=adjust,
                    open=_decimal(_cell(record, "open"), default=close),
                    high=_decimal(_cell(record, "high"), default=close),
                    low=_decimal(_cell(record, "low"), default=close),
                    close=close,
                    volume=int(_decimal(_cell(record, "volume"))),
                    amount=_decimal(_cell(record, "amount")),
                    # AkShare 不给前收盘，用上一行的收盘补。首行留 0 而不是
                    # 拿自己当前收——那会让首日涨跌幅恒等于 0，是个假数据
                    pre_close=previous,
                )
            )
            previous = close
        return out

    def fetch_instruments(self) -> list[Instrument]:
        """拉取标的列表（股票 + ETF）。

        **注意**：AkShare 的实时列表接口只含**在市**标的，不含退市的。
        因此历史初始化应以 BaoStock 为主力，本源用于补 ETF 与北交所。
        单独用它会造成幸存者偏差（docs/08 A1）。

        Returns:
            标的列表。

        Raises:
            DataSourceError: 接口调用失败。
        """
        module = self._module()
        out: list[Instrument] = []

        # 两个接口分别取股票与 ETF；任一失败只丢那一类，不拖垮另一类
        for endpoint, asset_type in (
            ("stock_info_a_code_name", AssetType.STOCK),
            ("fund_etf_spot_em", AssetType.ETF),
        ):
            self._limiter.acquire()
            try:
                frame = getattr(module, endpoint)()
            except Exception as exc:
                _log.warning(
                    "akshare_instruments_failed",
                    endpoint=endpoint,
                    asset=asset_type.value,
                    error=str(exc),
                )
                continue
            out.extend(self._to_instruments(frame, asset_type))

        if not out:
            msg = "AkShare 未返回任何标的"
            raise DataSourceError(msg, source="akshare")
        _log.info("akshare_instruments_fetched", count=len(out))
        return out

    @staticmethod
    def _to_instruments(frame: Any, asset_type: AssetType) -> list[Instrument]:  # noqa: ANN401
        """DataFrame → Instrument 列表。

        Args:
            frame: AkShare 返回的 DataFrame。
            asset_type: 资产类型。

        Returns:
            标的列表。
        """
        if frame is None or len(frame) == 0:
            return []

        out: list[Instrument] = []
        for record in frame.to_dict("records"):
            raw = record.get("code") or record.get("代码")
            if raw is None:
                continue
            try:
                symbol = parse_symbol(str(raw))
            except ValueError:
                continue
            out.append(
                Instrument(
                    symbol=symbol,
                    name=str(record.get("name") or record.get("名称") or symbol),
                    asset_type=asset_type,
                    exchange=split_symbol(symbol)[1],
                    board=Board.ETF if asset_type is AssetType.ETF else _infer_board(symbol),
                    list_date=dt.date(1990, 12, 19),
                )
            )
        return out

    def fetch_trading_days(self, *, start: TradeDate, end: TradeDate) -> list[TradeDate]:
        """拉取交易日历。

        Args:
            start: 起始日。
            end: 结束日。

        Returns:
            交易日列表。

        Raises:
            DataSourceError: 接口调用失败。
        """
        module = self._module()
        self._limiter.acquire()
        try:
            frame = module.tool_trade_date_hist_sina()
        except Exception as exc:
            msg = "AkShare 交易日历接口失败"
            raise DataSourceError(msg, source="akshare", error=str(exc)) from exc

        days = {_to_date(record["trade_date"]) for record in frame.to_dict("records")}
        return sorted(d for d in days if start <= d <= end)

    def health_check(self) -> SourceHealth:
        """探测可用性。

        Returns:
            健康度。
        """
        started = now()
        try:
            module = self._module()
            module.tool_trade_date_hist_sina()
        except Exception as exc:
            return SourceHealth(name=self.name, ok=False, checked_at=now(), message=str(exc))
        latency = (now() - started).total_seconds() * 1000
        return SourceHealth(
            name=self.name, ok=True, checked_at=now(), message="接口可达", latency_ms=latency
        )


def _is_etf(symbol: Symbol) -> bool:
    """是否为场内基金。

    Args:
        symbol: 标的。

    Returns:
        是 ETF 则 True。
    """
    code, exchange = split_symbol(symbol)
    if exchange is Exchange.SH:
        return code.startswith(ETF_PREFIXES_SH)
    if exchange is Exchange.SZ:
        return code.startswith(ETF_PREFIXES_SZ)
    return False


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
