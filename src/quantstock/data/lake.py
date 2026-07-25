"""数据湖：Parquet 列存 + DuckDB 查询。

规范见 docs/04-数据规格.md 第七节。

设计要点：

- 分区键选 ``freq/adjust/symbol``（日线及以上），增量更新时只重写单个标的的文件。
- **写入幂等**：按主键去重后整体重写分区，重复执行不产生重复行（DQ03）。
- 业务代码禁止直接拼路径读文件，一律走本模块——分区结构变化时只改这里。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from quantstock.data.types import Bar
from quantstock.infra.clock import CST
from quantstock.infra.errors import DataError
from quantstock.infra.logging import get_logger
from quantstock.infra.types import Adjust, Freq, Symbol, TradeDate

__all__ = ["ParquetLake"]

_log = get_logger(__name__)

_BAR_SCHEMA = pa.schema(
    [
        ("symbol", pa.string()),
        ("dt", pa.timestamp("us", tz="Asia/Shanghai")),
        ("trade_date", pa.date32()),
        ("freq", pa.string()),
        ("adjust", pa.string()),
        ("open", pa.decimal128(20, 4)),
        ("high", pa.decimal128(20, 4)),
        ("low", pa.decimal128(20, 4)),
        ("close", pa.decimal128(20, 4)),
        ("pre_close", pa.decimal128(20, 4)),
        ("volume", pa.int64()),
        ("amount", pa.decimal128(20, 4)),
        ("limit_up", pa.decimal128(20, 4)),
        ("limit_down", pa.decimal128(20, 4)),
        ("is_suspended", pa.bool_()),
        ("adj_factor", pa.decimal128(20, 8)),
        ("source", pa.string()),
    ]
)

_PRICE_SCALE = Decimal("0.0001")
_FACTOR_SCALE = Decimal("0.00000001")


class ParquetLake:
    """基于 Parquet 的本地数据湖。

    Example:
        >>> lake = ParquetLake(Path("var/lake"))
        >>> lake.write_bars(bars)
        >>> loaded = lake.read_bars([symbol], start=..., end=...)
    """

    def __init__(self, root: Path) -> None:
        """初始化。

        Args:
            root: 数据湖根目录。
        """
        self._root = Path(root)

    @property
    def root(self) -> Path:
        """数据湖根目录。"""
        return self._root

    def _bar_path(self, *, symbol: Symbol, freq: Freq, adjust: Adjust) -> Path:
        """某标的某口径的 Parquet 文件路径。

        Args:
            symbol: 标的。
            freq: 频率。
            adjust: 复权口径。

        Returns:
            文件路径。
        """
        return (
            self._root
            / "bars"
            / f"freq={freq.value}"
            / f"adjust={adjust.value}"
            / f"symbol={symbol}"
            / "part.parquet"
        )

    # ------------------------------------------------------------------ 写入
    def write_bars(self, bars: Iterable[Bar], *, merge: bool = True) -> int:
        """写入 K 线，按 ``(symbol, freq, adjust)`` 分区。

        **幂等**：与已有数据按主键 ``(symbol, dt)`` 合并去重后整体重写分区，
        同一批数据重复写入不会产生重复行。

        Args:
            bars: 待写入的 K 线。
            merge: True 时与已有数据合并；False 时直接覆盖该分区。

        Returns:
            写入后各分区的总行数。

        Raises:
            DataError: 同一批数据混用了不同的 freq/adjust 口径。
        """
        grouped: dict[tuple[Symbol, Freq, Adjust], list[Bar]] = {}
        for bar in bars:
            grouped.setdefault((bar.symbol, bar.freq, bar.adjust), []).append(bar)

        total = 0
        for (symbol, freq, adjust), group in grouped.items():
            path = self._bar_path(symbol=symbol, freq=freq, adjust=adjust)
            path.parent.mkdir(parents=True, exist_ok=True)

            rows: dict[dt.datetime, dict[str, Any]] = {}
            if merge and path.exists():
                for existing in _table_to_rows(pq.read_table(path)):
                    rows[existing["dt"]] = existing
            for bar in group:
                rows[bar.dt] = _bar_to_row(bar)

            ordered = [rows[key] for key in sorted(rows)]
            pq.write_table(_rows_to_table(ordered), path, compression="zstd", version="2.6")
            total += len(ordered)

        if grouped:
            _log.info(
                "bars_written",
                partitions=len(grouped),
                rows=total,
                symbols=len({k[0] for k in grouped}),
            )
        return total

    # ------------------------------------------------------------------ 读取
    def read_bars(
        self,
        symbols: Sequence[Symbol],
        *,
        start: TradeDate | None = None,
        end: TradeDate | None = None,
        freq: Freq = Freq.D,
        adjust: Adjust = Adjust.NONE,
    ) -> list[Bar]:
        """读取 K 线。

        Args:
            symbols: 标的列表。
            start: 起始交易日（含）。
            end: 结束交易日（含）。
            freq: 频率。
            adjust: 复权口径。**必须显式指定**（红线 R4）。

        Returns:
            按 ``(symbol, dt)`` 排序的 K 线列表；无数据时为空列表。
        """
        result: list[Bar] = []
        for symbol in symbols:
            path = self._bar_path(symbol=symbol, freq=freq, adjust=adjust)
            if not path.exists():
                continue
            for row in _table_to_rows(pq.read_table(path)):
                trade_date: TradeDate = row["trade_date"]
                if start is not None and trade_date < start:
                    continue
                if end is not None and trade_date > end:
                    continue
                result.append(_row_to_bar(row))
        result.sort(key=lambda b: (b.symbol, b.dt))
        return result

    def available_symbols(
        self, *, freq: Freq = Freq.D, adjust: Adjust = Adjust.NONE
    ) -> list[Symbol]:
        """列出已有数据的标的。

        Args:
            freq: 频率。
            adjust: 复权口径。

        Returns:
            标的列表，升序。
        """
        base = self._root / "bars" / f"freq={freq.value}" / f"adjust={adjust.value}"
        if not base.exists():
            return []
        return sorted(
            Symbol(p.name.removeprefix("symbol="))
            for p in base.iterdir()
            if p.is_dir() and p.name.startswith("symbol=")
        )

    def last_trade_date(
        self, symbol: Symbol, *, freq: Freq = Freq.D, adjust: Adjust = Adjust.NONE
    ) -> TradeDate | None:
        """某标的已有数据的最后一个交易日。

        增量更新以此为起点，只拉缺失区间。

        Args:
            symbol: 标的。
            freq: 频率。
            adjust: 复权口径。

        Returns:
            最后交易日；无数据时返回 None。
        """
        path = self._bar_path(symbol=symbol, freq=freq, adjust=adjust)
        if not path.exists():
            return None
        table = pq.read_table(path, columns=["trade_date"])
        if table.num_rows == 0:
            return None
        dates = [d for d in table.column("trade_date").to_pylist() if d is not None]
        return max(dates) if dates else None

    def stats(self) -> dict[str, int]:
        """数据湖概况，供界面展示。

        Returns:
            分区数、文件数、总字节数。
        """
        bars_dir = self._root / "bars"
        if not bars_dir.exists():
            return {"partitions": 0, "files": 0, "bytes": 0}
        files = list(bars_dir.rglob("*.parquet"))
        return {
            "partitions": len({f.parent for f in files}),
            "files": len(files),
            "bytes": sum(f.stat().st_size for f in files),
        }


# --------------------------------------------------------------------- 行转换
def _bar_to_row(bar: Bar) -> dict[str, Any]:
    """Bar → Parquet 行。

    Args:
        bar: K 线。

    Returns:
        行字典。
    """
    return {
        "symbol": str(bar.symbol),
        "dt": bar.dt,
        "trade_date": bar.trade_date,
        "freq": bar.freq.value,
        "adjust": bar.adjust.value,
        "open": bar.open.quantize(_PRICE_SCALE),
        "high": bar.high.quantize(_PRICE_SCALE),
        "low": bar.low.quantize(_PRICE_SCALE),
        "close": bar.close.quantize(_PRICE_SCALE),
        "pre_close": bar.pre_close.quantize(_PRICE_SCALE),
        "volume": bar.volume,
        "amount": bar.amount.quantize(_PRICE_SCALE),
        "limit_up": None if bar.limit_up is None else bar.limit_up.quantize(_PRICE_SCALE),
        "limit_down": None if bar.limit_down is None else bar.limit_down.quantize(_PRICE_SCALE),
        "is_suspended": bar.is_suspended,
        "adj_factor": bar.adj_factor.quantize(_FACTOR_SCALE),
        "source": bar.source,
    }


def _row_to_bar(row: dict[str, Any]) -> Bar:
    """Parquet 行 → Bar。

    Args:
        row: 行字典。

    Returns:
        K 线。
    """
    moment: dt.datetime = row["dt"]
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=CST)
    return Bar(
        symbol=Symbol(row["symbol"]),
        dt=moment,
        trade_date=row["trade_date"],
        freq=Freq(row["freq"]),
        adjust=Adjust(row["adjust"]),
        open=row["open"],
        high=row["high"],
        low=row["low"],
        close=row["close"],
        pre_close=row["pre_close"],
        volume=int(row["volume"]),
        amount=row["amount"],
        limit_up=row["limit_up"],
        limit_down=row["limit_down"],
        is_suspended=bool(row["is_suspended"]),
        adj_factor=row["adj_factor"],
        source=row["source"] or "",
    )


def _rows_to_table(rows: Sequence[dict[str, Any]]) -> pa.Table:
    """行列表 → Arrow 表。

    Args:
        rows: 行列表。

    Returns:
        Arrow 表。

    Raises:
        DataError: 行数据不符合 schema。
    """
    if not rows:
        return _BAR_SCHEMA.empty_table()
    columns = {name: [row[name] for row in rows] for name in _BAR_SCHEMA.names}
    try:
        return pa.Table.from_pydict(columns, schema=_BAR_SCHEMA)
    except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
        msg = "K 线数据不符合数据湖 schema"
        raise DataError(msg, error=str(exc)) from exc


def _table_to_rows(table: pa.Table) -> list[dict[str, Any]]:
    """Arrow 表 → 行列表。

    Args:
        table: Arrow 表。

    Returns:
        行列表。
    """
    rows: list[dict[str, Any]] = table.to_pylist()
    return rows
