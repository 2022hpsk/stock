"""CSV 行情数据源。

**这不是测试桩，是一个正经入口。** 两类用户会用到它：

1. 已经有一份数据的人——券商导出、同花顺/通达信导出、自己爬的历史库，
   放成 CSV 就能直接喂进来，不必等在线源接通；
2. 网络受限或不想依赖第三方接口的人——整条链路（因子 → 信号 → 组合 →
   风控 → 建议 → 回测）在纯离线环境下完整可跑。

目录布局::

    <root>/
    ├── instruments.csv          标的列表（含退市标的）
    ├── trading_days.csv         交易日历
    └── bars/
        ├── 600519.SH.csv
        └── 510300.SH.csv

列名大小写不敏感，且接受常见中文表头（``日期``/``开盘``/……）——
用户从各家软件导出的表头五花八门，为此让用户手工改列名是没必要的摩擦。
"""

from __future__ import annotations

import csv
import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path

from quantstock.data.types import Bar, Instrument, SourceHealth
from quantstock.infra.clock import CST, now
from quantstock.infra.errors import DataSourceError
from quantstock.infra.logging import get_logger
from quantstock.infra.money import ZERO
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

__all__ = ["CsvSource"]

_log = get_logger(__name__)

_DATE_ALIASES = ("date", "trade_date", "日期", "交易日期", "时间")
_OPEN_ALIASES = ("open", "开盘", "开盘价")
_HIGH_ALIASES = ("high", "最高", "最高价")
_LOW_ALIASES = ("low", "最低", "最低价")
_CLOSE_ALIASES = ("close", "收盘", "收盘价")
_VOLUME_ALIASES = ("volume", "vol", "成交量")
_AMOUNT_ALIASES = ("amount", "turnover", "成交额", "成交金额")
_PRECLOSE_ALIASES = ("pre_close", "preclose", "前收", "前收盘", "昨收")

_NAME_ALIASES = ("name", "名称", "证券简称")
_SYMBOL_ALIASES = ("symbol", "code", "代码", "证券代码")
_LIST_DATE_ALIASES = ("list_date", "上市日期", "ipo_date")
_DELIST_DATE_ALIASES = ("delist_date", "退市日期")
_INDUSTRY_ALIASES = ("sw_l1", "industry", "行业", "申万一级")


def _pick(row: Mapping[str, str], aliases: Sequence[str]) -> str | None:
    """按别名取列。

    Args:
        row: CSV 行。
        aliases: 候选列名。

    Returns:
        命中的值；无命中时 None。
    """
    lowered = {k.strip().lower(): v for k, v in row.items() if k}
    for alias in aliases:
        value = lowered.get(alias.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _to_date(raw: str) -> TradeDate:
    """解析日期，容忍 ``2026-07-25`` / ``20260725`` / ``2026/07/25``。

    Args:
        raw: 原始字符串。

    Returns:
        日期。

    Raises:
        DataSourceError: 无法解析。
    """
    text = raw.strip().replace("/", "-")
    if text.isdigit() and len(text) == 8:  # noqa: PLR2004 - YYYYMMDD 就是 8 位
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError as exc:
        msg = f"无法解析日期：{raw!r}"
        raise DataSourceError(msg, source="csv") from exc


def _to_decimal(raw: str | None, *, default: Decimal = ZERO) -> Decimal:
    """解析金额。

    走 ``Decimal(str)`` 而非 ``float``——CSV 里的价格是精确的十进制文本，
    经过 float 就再也回不去了（红线 R1）。

    Args:
        raw: 原始字符串。
        default: 缺省值。

    Returns:
        Decimal 值。
    """
    if raw is None or not str(raw).strip():
        return default
    cleaned = str(raw).strip().replace(",", "")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return default


class CsvSource:
    """从本地 CSV 读取行情。"""

    name = "csv"

    def __init__(self, root: Path) -> None:
        """初始化。

        Args:
            root: 数据根目录。
        """
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """数据根目录。"""
        return self._root

    @property
    def bars_dir(self) -> Path:
        """K 线目录。"""
        return self._root / "bars"

    def _bar_file(self, symbol: Symbol) -> Path | None:
        """找某标的的 K 线文件。

        同时接受 ``600519.SH.csv`` 与 ``600519.csv`` 两种命名。

        Args:
            symbol: 标的。

        Returns:
            文件路径；不存在时 None。
        """
        code, _ = split_symbol(symbol)
        for candidate in (f"{symbol}.csv", f"{code}.csv"):
            path = self.bars_dir / candidate
            if path.exists():
                return path
        return None

    def fetch_daily_bars(
        self,
        symbols: Sequence[Symbol],
        *,
        start: TradeDate,
        end: TradeDate,
        adjust: Adjust,
    ) -> list[Bar]:
        """读取日线。

        Args:
            symbols: 标的列表。
            start: 起始日（含）。
            end: 结束日（含）。
            adjust: 复权口径。**按调用方声明如实标注**——
                CSV 里是什么口径由用户负责，这里不做转换也不猜测。

        Returns:
            K 线列表，按 ``(symbol, trade_date)`` 排序。

        Raises:
            DataSourceError: 文件存在但格式非法。
        """
        out: list[Bar] = []
        for symbol in symbols:
            path = self._bar_file(symbol)
            if path is None:
                # 缺文件按"这只没数据"处理而不是抛错：批量拉取时
                # 一只缺失不该让整批失败，缺口由 DQ 校验统一报告
                _log.warning("csv_bars_missing", symbol=str(symbol))
                continue
            out.extend(self._read_bars(path, symbol, start=start, end=end, adjust=adjust))
        return sorted(out, key=lambda b: (b.symbol, b.trade_date))

    def _read_bars(
        self,
        path: Path,
        symbol: Symbol,
        *,
        start: TradeDate,
        end: TradeDate,
        adjust: Adjust,
    ) -> list[Bar]:
        """读单个文件。

        Args:
            path: 文件路径。
            symbol: 标的。
            start: 起始日。
            end: 结束日。
            adjust: 复权口径。

        Returns:
            K 线列表。

        Raises:
            DataSourceError: 缺少必需列。
        """
        out: list[Bar] = []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for line_no, row in enumerate(csv.DictReader(handle), start=2):
                raw_date = _pick(row, _DATE_ALIASES)
                raw_close = _pick(row, _CLOSE_ALIASES)
                if raw_date is None or raw_close is None:
                    msg = "CSV 缺少日期或收盘价列"
                    raise DataSourceError(
                        msg, source="csv", path=str(path), line=line_no, columns=list(row)
                    )

                trade_date = _to_date(raw_date)
                if not start <= trade_date <= end:
                    continue

                close = _to_decimal(raw_close)
                out.append(
                    Bar(
                        symbol=symbol,
                        dt=dt.datetime.combine(trade_date, dt.time(15, 0), tzinfo=CST),
                        trade_date=trade_date,
                        freq=Freq.D,
                        adjust=adjust,
                        open=_to_decimal(_pick(row, _OPEN_ALIASES), default=close),
                        high=_to_decimal(_pick(row, _HIGH_ALIASES), default=close),
                        low=_to_decimal(_pick(row, _LOW_ALIASES), default=close),
                        close=close,
                        volume=int(_to_decimal(_pick(row, _VOLUME_ALIASES))),
                        amount=_to_decimal(_pick(row, _AMOUNT_ALIASES)),
                        pre_close=_to_decimal(_pick(row, _PRECLOSE_ALIASES)),
                    )
                )
        return out

    def fetch_instruments(self) -> list[Instrument]:
        """读取标的列表。

        Returns:
            标的列表。文件不存在时由 K 线目录反推一份最小可用的清单——
            用户只丢了 bars 目录进来也应该能跑。

        Raises:
            DataSourceError: 文件存在但缺少代码列。
        """
        path = self._root / "instruments.csv"
        if not path.exists():
            return self._infer_instruments()

        out: list[Instrument] = []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for line_no, row in enumerate(csv.DictReader(handle), start=2):
                raw_symbol = _pick(row, _SYMBOL_ALIASES)
                if raw_symbol is None:
                    msg = "instruments.csv 缺少代码列"
                    raise DataSourceError(msg, source="csv", line=line_no, columns=list(row))

                symbol = parse_symbol(raw_symbol)
                delist = _pick(row, _DELIST_DATE_ALIASES)
                out.append(
                    Instrument(
                        symbol=symbol,
                        name=_pick(row, _NAME_ALIASES) or str(symbol),
                        asset_type=_infer_asset_type(symbol),
                        exchange=split_symbol(symbol)[1],
                        board=_infer_board(symbol),
                        list_date=_to_date(_pick(row, _LIST_DATE_ALIASES) or "1990-12-19"),
                        delist_date=_to_date(delist) if delist else None,
                        sw_l1=_pick(row, _INDUSTRY_ALIASES) or "",
                    )
                )
        return out

    def _infer_instruments(self) -> list[Instrument]:
        """由 bars 目录反推标的列表。

        Returns:
            最小可用的标的清单。
        """
        if not self.bars_dir.is_dir():
            return []
        out: list[Instrument] = []
        for path in sorted(self.bars_dir.glob("*.csv")):
            try:
                symbol = parse_symbol(path.stem)
            except ValueError:
                _log.warning("csv_symbol_unparseable", file=path.name)
                continue
            out.append(
                Instrument(
                    symbol=symbol,
                    name=str(symbol),
                    asset_type=_infer_asset_type(symbol),
                    exchange=split_symbol(symbol)[1],
                    board=_infer_board(symbol),
                    list_date=dt.date(1990, 12, 19),
                )
            )
        return out

    def fetch_trading_days(self, *, start: TradeDate, end: TradeDate) -> list[TradeDate]:
        """读取交易日历。

        Args:
            start: 起始日。
            end: 结束日。

        Returns:
            交易日列表。无日历文件时由已有 K 线的日期并集推导——
            这对回测足够，且比"假设周一到周五都是交易日"准确得多
            （后者会把春节长假算成交易日）。
        """
        path = self._root / "trading_days.csv"
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as handle:
                days = {
                    _to_date(value)
                    for row in csv.DictReader(handle)
                    if (value := _pick(row, _DATE_ALIASES)) is not None
                }
            return sorted(d for d in days if start <= d <= end)
        return self._infer_trading_days(start=start, end=end)

    def _infer_trading_days(self, *, start: TradeDate, end: TradeDate) -> list[TradeDate]:
        """由已有 K 线推导交易日。

        Args:
            start: 起始日。
            end: 结束日。

        Returns:
            交易日列表。
        """
        if not self.bars_dir.is_dir():
            return []
        days: set[TradeDate] = set()
        for path in self.bars_dir.glob("*.csv"):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if (value := _pick(row, _DATE_ALIASES)) is not None:
                        days.add(_to_date(value))
        return sorted(d for d in days if start <= d <= end)

    def health_check(self) -> SourceHealth:
        """探测可用性。

        Returns:
            健康度。
        """
        if not self._root.is_dir():
            return SourceHealth(
                name=self.name, ok=False, checked_at=now(), message=f"目录不存在：{self._root}"
            )
        count = len(list(self.bars_dir.glob("*.csv"))) if self.bars_dir.is_dir() else 0
        if count == 0:
            return SourceHealth(
                name=self.name, ok=False, checked_at=now(), message=f"{self.bars_dir} 下没有 CSV"
            )
        return SourceHealth(name=self.name, ok=True, checked_at=now(), message=f"{count} 只标的")

    def write_bars(self, bars: Iterable[Bar]) -> int:
        """把 K 线写成 CSV，便于用户导出/交换数据。

        Args:
            bars: K 线。

        Returns:
            写入的条数。
        """
        grouped: dict[Symbol, list[Bar]] = {}
        for bar in bars:
            grouped.setdefault(bar.symbol, []).append(bar)

        self.bars_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for symbol, rows in grouped.items():
            path = self.bars_dir / f"{symbol}.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    ["date", "open", "high", "low", "close", "volume", "amount", "pre_close"]
                )
                for bar in sorted(rows, key=lambda b: b.trade_date):
                    writer.writerow(
                        [
                            bar.trade_date.isoformat(),
                            bar.open,
                            bar.high,
                            bar.low,
                            bar.close,
                            bar.volume,
                            bar.amount,
                            bar.pre_close,
                        ]
                    )
                    written += 1
        return written


def _infer_asset_type(symbol: Symbol) -> AssetType:
    """由代码前缀推断资产类型。

    Args:
        symbol: 标的。

    Returns:
        资产类型。
    """
    code, exchange = split_symbol(symbol)
    if exchange is Exchange.SH and code.startswith(("510", "511", "512", "513", "515", "588")):
        return AssetType.ETF
    if exchange is Exchange.SZ and code.startswith(("159",)):
        return AssetType.ETF
    return AssetType.STOCK


def _infer_board(symbol: Symbol) -> Board:
    """由代码前缀推断板块。

    **ETF 的板块是 ETF，不是它所跟踪指数的板块。** 588000 跟踪科创50，
    但它本身是一只在上交所主板交易的基金——按 STAR 处理会让它错误地
    继承科创板的 200 股起、±20% 涨跌幅等规则。与 AkShareSource 保持一致。

    Args:
        symbol: 标的。

    Returns:
        板块。
    """
    if _infer_asset_type(symbol) is AssetType.ETF:
        return Board.ETF

    code, exchange = split_symbol(symbol)
    if code.startswith("688"):
        return Board.STAR
    if code.startswith("300"):
        return Board.GEM
    if exchange is Exchange.BJ:
        return Board.BSE
    return Board.MAIN
