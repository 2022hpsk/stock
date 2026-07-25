"""数据湖与数据源降级链测试。

重点：写入幂等（DQ03 的第一道防线）、复权口径隔离、全源失败必须抛异常。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

import pytest

from quantstock.data.lake import ParquetLake
from quantstock.data.protocols import FallbackChain, MarketDataSource
from quantstock.data.types import Bar, Instrument, SourceHealth
from quantstock.infra.clock import CST, now
from quantstock.infra.errors import DataSourceError
from quantstock.infra.types import (
    Adjust,
    AssetType,
    Board,
    Exchange,
    Freq,
    Symbol,
    TradeDate,
)

MAOTAI = Symbol("600519.SH")
CATL = Symbol("300750.SZ")
_DAY = dt.date(2026, 7, 24)


def make_bar(
    symbol: Symbol = MAOTAI,
    *,
    day: dt.date = _DAY,
    close: str = "105",
    adjust: Adjust = Adjust.NONE,
    source: str = "test",
) -> Bar:
    """构造一根合法 K 线。"""
    return Bar(
        symbol=symbol,
        dt=dt.datetime.combine(day, dt.time(15, 0), tzinfo=CST),
        trade_date=day,
        freq=Freq.D,
        adjust=adjust,
        open=Decimal("100"),
        high=Decimal("110"),
        low=Decimal("95"),
        close=Decimal(close),
        pre_close=Decimal("100"),
        volume=1000,
        amount=Decimal("105000"),
        adj_factor=Decimal("1.5"),
        source=source,
    )


class TestParquetLake:
    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        lake = ParquetLake(tmp_path)
        lake.write_bars([make_bar()])
        loaded = lake.read_bars([MAOTAI])
        assert len(loaded) == 1
        assert loaded[0].symbol == MAOTAI
        assert loaded[0].close == Decimal("105.0000")
        assert loaded[0].adj_factor == Decimal("1.50000000")
        assert loaded[0].dt.tzinfo is not None

    def test_write_is_idempotent(self, tmp_path: Path) -> None:
        """重复写同一批数据不得产生重复行——这是 DQ03 的第一道防线。"""
        lake = ParquetLake(tmp_path)
        bars = [make_bar()]
        lake.write_bars(bars)
        lake.write_bars(bars)
        assert len(lake.read_bars([MAOTAI])) == 1

    def test_rewrite_same_key__updates_in_place(self, tmp_path: Path) -> None:
        """同一主键重写应覆盖而非追加——数据修正走这条路径。"""
        lake = ParquetLake(tmp_path)
        lake.write_bars([make_bar(close="105")])
        lake.write_bars([make_bar(close="108")])
        loaded = lake.read_bars([MAOTAI])
        assert len(loaded) == 1
        assert loaded[0].close == Decimal("108.0000")

    def test_incremental_append(self, tmp_path: Path) -> None:
        lake = ParquetLake(tmp_path)
        lake.write_bars([make_bar(day=dt.date(2026, 7, 23))])
        lake.write_bars([make_bar(day=dt.date(2026, 7, 24))])
        assert len(lake.read_bars([MAOTAI])) == 2

    def test_adjust_partitions_are_isolated(self, tmp_path: Path) -> None:
        """红线 R4：不同复权口径必须物理隔离，读取时不会互相污染。"""
        lake = ParquetLake(tmp_path)
        lake.write_bars([make_bar(close="105", adjust=Adjust.NONE)])
        lake.write_bars([make_bar(close="157.5", adjust=Adjust.HFQ)])

        none_bars = lake.read_bars([MAOTAI], adjust=Adjust.NONE)
        hfq_bars = lake.read_bars([MAOTAI], adjust=Adjust.HFQ)
        assert none_bars[0].close == Decimal("105.0000")
        assert hfq_bars[0].close == Decimal("157.5000")

    def test_read_date_range_filter(self, tmp_path: Path) -> None:
        lake = ParquetLake(tmp_path)
        for day in (dt.date(2026, 7, 22), dt.date(2026, 7, 23), dt.date(2026, 7, 24)):
            lake.write_bars([make_bar(day=day)])
        loaded = lake.read_bars([MAOTAI], start=dt.date(2026, 7, 23), end=dt.date(2026, 7, 23))
        assert len(loaded) == 1
        assert loaded[0].trade_date == dt.date(2026, 7, 23)

    def test_read_missing_symbol__empty(self, tmp_path: Path) -> None:
        assert ParquetLake(tmp_path).read_bars([MAOTAI]) == []

    def test_available_symbols(self, tmp_path: Path) -> None:
        lake = ParquetLake(tmp_path)
        lake.write_bars([make_bar(MAOTAI), make_bar(CATL)])
        assert lake.available_symbols() == sorted([CATL, MAOTAI])

    def test_last_trade_date__drives_incremental_update(self, tmp_path: Path) -> None:
        """增量更新以此为起点，只拉缺失区间。"""
        lake = ParquetLake(tmp_path)
        assert lake.last_trade_date(MAOTAI) is None
        lake.write_bars([make_bar(day=dt.date(2026, 7, 22)), make_bar(day=dt.date(2026, 7, 24))])
        assert lake.last_trade_date(MAOTAI) == dt.date(2026, 7, 24)

    def test_multi_symbol_sorted_output(self, tmp_path: Path) -> None:
        lake = ParquetLake(tmp_path)
        lake.write_bars([make_bar(MAOTAI), make_bar(CATL)])
        loaded = lake.read_bars([MAOTAI, CATL])
        assert [b.symbol for b in loaded] == sorted([CATL, MAOTAI])

    def test_stats(self, tmp_path: Path) -> None:
        lake = ParquetLake(tmp_path)
        assert lake.stats()["files"] == 0
        lake.write_bars([make_bar(MAOTAI), make_bar(CATL)])
        stats = lake.stats()
        assert stats["files"] == 2
        assert stats["bytes"] > 0


class FakeSource:
    """可控的假数据源，用于测试降级链。"""

    def __init__(self, name: str, *, fails: bool = False) -> None:
        self.name = name
        self.fails = fails
        self.calls = 0

    def _guard(self) -> None:
        self.calls += 1
        if self.fails:
            msg = f"{self.name} 不可用"
            raise DataSourceError(msg)

    def fetch_daily_bars(
        self,
        symbols: Sequence[Symbol],
        *,
        start: TradeDate,
        end: TradeDate,
        adjust: Adjust,
    ) -> list[Bar]:
        self._guard()
        return [make_bar(s, source=self.name) for s in symbols]

    def fetch_instruments(self) -> list[Instrument]:
        self._guard()
        return [
            Instrument(
                symbol=MAOTAI,
                name="贵州茅台",
                asset_type=AssetType.STOCK,
                exchange=Exchange.SH,
                board=Board.MAIN,
                list_date=dt.date(2001, 8, 27),
            )
        ]

    def fetch_trading_days(self, *, start: TradeDate, end: TradeDate) -> list[TradeDate]:
        self._guard()
        return [_DAY]

    def health_check(self) -> SourceHealth:
        return SourceHealth(name=self.name, ok=not self.fails, checked_at=now())


class TestFallbackChain:
    def test_empty_chain__rejected(self) -> None:
        with pytest.raises(ValueError, match="至少需要一个数据源"):
            FallbackChain([])

    def test_primary_succeeds__secondary_not_called(self) -> None:
        primary, backup = FakeSource("tushare"), FakeSource("akshare")
        chain = FallbackChain([primary, backup])
        bars = chain.fetch_daily_bars([MAOTAI], start=_DAY, end=_DAY)
        assert bars[0].source == "tushare"
        assert backup.calls == 0

    def test_primary_fails__falls_back(self) -> None:
        primary, backup = FakeSource("tushare", fails=True), FakeSource("akshare")
        chain = FallbackChain([primary, backup])
        bars = chain.fetch_daily_bars([MAOTAI], start=_DAY, end=_DAY)
        assert bars[0].source == "akshare"

    def test_all_fail__raises_with_all_errors(self) -> None:
        """全部失败必须抛异常让上层停机——半成品数据比没有数据更危险。"""
        chain = FallbackChain(
            [FakeSource("tushare", fails=True), FakeSource("akshare", fails=True)]
        )
        with pytest.raises(DataSourceError, match="全部数据源均失败") as exc_info:
            chain.fetch_daily_bars([MAOTAI], start=_DAY, end=_DAY)
        assert set(exc_info.value.context["errors"]) == {"tushare", "akshare"}

    def test_fallback_applies_to_instruments(self) -> None:
        chain = FallbackChain([FakeSource("a", fails=True), FakeSource("b")])
        assert len(chain.fetch_instruments()) == 1

    def test_fallback_applies_to_calendar(self) -> None:
        chain = FallbackChain([FakeSource("a", fails=True), FakeSource("b")])
        assert chain.fetch_trading_days(start=_DAY, end=_DAY) == [_DAY]

    def test_health_reports_each_source(self) -> None:
        chain = FallbackChain([FakeSource("a"), FakeSource("b", fails=True)])
        health = {h.name: h.ok for h in chain.health()}
        assert health == {"a": True, "b": False}

    def test_health_counts_consecutive_failures(self) -> None:
        chain = FallbackChain([FakeSource("a", fails=True), FakeSource("b")])
        chain.fetch_instruments()
        chain.fetch_instruments()
        failures = {h.name: h.consecutive_failures for h in chain.health()}
        assert failures["a"] == 2
        assert failures["b"] == 0

    def test_source_names_preserve_priority(self) -> None:
        chain = FallbackChain([FakeSource("x"), FakeSource("y")])
        assert chain.source_names == ("x", "y")

    def test_satisfies_protocol(self) -> None:
        assert isinstance(FakeSource("a"), MarketDataSource)
