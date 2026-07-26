"""数据服务：初始化、增量更新、质量校验、数据湖状态。

CLI 与界面共用同一实现。

**与情报层相反的降级语义**：情报缺失只降级为"缺少证据"，行情缺失必须停机——
拿不准的价格算不出对的仓位。所以这里全部数据源都失败时是**抛错**而不是
返回空结果（docs/04 第七节的降级链）。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from quantstock.config.settings import Settings
from quantstock.data.lake import ParquetLake
from quantstock.data.protocols import FallbackChain, MarketDataSource
from quantstock.data.sources import AkShareSource, BaoStockSource, CsvSource
from quantstock.data.types import Bar, Instrument, SourceHealth
from quantstock.data.universe import UniverseRegistry
from quantstock.infra.clock import today
from quantstock.infra.errors import DataError
from quantstock.infra.logging import get_logger
from quantstock.infra.types import Adjust, Freq, Symbol, TradeDate

__all__ = ["DataService", "DataStatus", "UpdateReport", "build_source"]

_log = get_logger(__name__)

CORE_UNIVERSE: tuple[Symbol, ...] = (
    # 分级初始化的 core 档：主要宽基 ETF + 少量代表性个股。
    # 刻意做小——第一次用的人应该 20 分钟内看到结果，而不是等几小时全市场
    Symbol("510300.SH"),  # 沪深300
    Symbol("510500.SH"),  # 中证500
    Symbol("588000.SH"),  # 科创50
    Symbol("159915.SZ"),  # 创业板
    Symbol("511990.SH"),  # 华宝添益（现金替代）
    Symbol("600519.SH"),
    Symbol("601318.SH"),
    Symbol("000858.SZ"),
    Symbol("300750.SZ"),
    Symbol("002415.SZ"),
)


def build_source(settings: Settings, *, kind: str | None = None) -> MarketDataSource:
    """按配置构造数据源。

    Args:
        settings: 运行期配置。
        kind: 显式指定源；None 表示按 ``data.source_chain`` 构造降级链。

    Returns:
        数据源。多源时返回降级链。

    Raises:
        DataError: 指定了未知的源。
    """

    def make(name: str) -> MarketDataSource:
        if name == "csv":
            return CsvSource(settings.var_dir / "csv")
        if name == "baostock":
            return BaoStockSource()
        if name == "akshare":
            return AkShareSource()
        msg = f"未知的数据源 {name}，可选：akshare / baostock / csv"
        raise DataError(msg)

    if kind is not None:
        return make(kind)

    known = {"csv", "baostock", "akshare"}
    chain = [make(name) for name in settings.config.data.source_chain if name in known]
    if not chain:
        # 一个都没配上时退到 CSV：至少让用户能用自己的数据跑通，
        # 而不是抛一个"没有可用数据源"然后什么也做不了
        _log.warning("no_source_configured_fallback_to_csv")
        return CsvSource(settings.var_dir / "csv")
    if len(chain) == 1:
        return chain[0]
    return FallbackChain(chain)


@dataclass(frozen=True, slots=True)
class UpdateReport:
    """一次数据更新的结果。"""

    symbols: int
    bars_written: int
    start: TradeDate
    end: TradeDate
    source: str
    failures: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        """人类可读摘要。"""
        base = (
            f"{self.source}：{self.symbols} 只标的，{self.bars_written} 根 K 线"
            f"（{self.start} ~ {self.end}）"
        )
        return base + (f"，{len(self.failures)} 只失败" if self.failures else "")


@dataclass(frozen=True, slots=True)
class DataStatus:
    """数据湖状态。"""

    root: Path
    symbols: int
    files: int
    """Parquet 文件数。lake.stats() 报的是文件与字节，不是行数——
    数 K 线行数要把所有分区读一遍，对状态查询来说太贵了。"""
    bytes_on_disk: int
    latest_date: TradeDate | None
    instruments: int
    delisted: int
    health: tuple[SourceHealth, ...] = field(default_factory=tuple)

    @property
    def is_ready(self) -> bool:
        """数据是否足以出建议。"""
        return self.symbols > 0 and self.latest_date is not None

    @property
    def message(self) -> str:
        """一行摘要。"""
        if not self.is_ready:
            return "数据湖为空，请先执行 quantstock data init"
        size = self.bytes_on_disk / 1024 / 1024
        return (
            f"{self.symbols} 只标的 / {self.files} 个分区文件（{size:.1f} MB），"
            f"最新 {self.latest_date}；标的表 {self.instruments} 条（含退市 {self.delisted}）"
        )


class DataService:
    """数据服务。"""

    def __init__(self, settings: Settings, *, source: MarketDataSource | None = None) -> None:
        """初始化。

        Args:
            settings: 运行期配置。
            source: 数据源；None 表示按配置构造。
        """
        self._settings = settings
        self._lake = ParquetLake(settings.var_dir / "lake")
        self._source = source or build_source(settings)
        self._universe_path = settings.var_dir / "lake" / "universe.json"
        # UniverseRegistry 是不可变快照（as_of 查询要求如此），
        # 所以每次同步标的表后重建，而不是就地增删
        self._universe = UniverseRegistry(self.load_instruments())

    @property
    def lake(self) -> ParquetLake:
        """数据湖。"""
        return self._lake

    @property
    def source(self) -> MarketDataSource:
        """数据源。"""
        return self._source

    @property
    def universe(self) -> UniverseRegistry:
        """标的注册表。"""
        return self._universe

    def core_universe(self) -> tuple[Symbol, ...]:
        """Core 档标的池。

        Returns:
            标的元组。
        """
        return CORE_UNIVERSE

    def resolve_universe(self, tier: str = "core") -> tuple[Symbol, ...]:
        """按档位解析标的池。

        Args:
            tier: ``core`` 或 ``all``。

        Returns:
            标的元组。``all`` 档在标的表为空时退回 core。
        """
        if tier != "all":
            return CORE_UNIVERSE
        listed = self.load_instruments()
        if not listed:
            _log.warning("universe_all_requested_but_empty_fallback_core")
            return CORE_UNIVERSE
        return tuple(i.symbol for i in listed)

    def update(
        self,
        symbols: Sequence[Symbol],
        *,
        start: TradeDate | None = None,
        end: TradeDate | None = None,
        adjust: Adjust | None = None,
    ) -> UpdateReport:
        """拉取并写入行情。

        **增量优先**：不给 ``start`` 时从数据湖里已有的最后一天接着拉，
        而不是每次都从 2015 年重拉一遍。

        Args:
            symbols: 标的列表。
            start: 起始日；None 表示增量续拉。
            end: 结束日；None 表示今日。
            adjust: 复权口径；None 表示用研究口径（后复权，红线 R4）。

        Returns:
            更新报告。

        Raises:
            DataError: 数据源不可用。
        """
        convention = adjust or Adjust(self._settings.config.data.adjust_for_research)
        finish = end or today()
        begin = start or self._incremental_start(symbols, adjust=convention)

        if begin > finish:
            return UpdateReport(
                symbols=len(symbols),
                bars_written=0,
                start=begin,
                end=finish,
                source=self._source.name,
            )

        bars = self._source.fetch_daily_bars(symbols, start=begin, end=finish, adjust=convention)
        written = self._lake.write_bars(bars)

        fetched = {b.symbol for b in bars}
        failures = tuple(str(s) for s in symbols if s not in fetched)
        if failures:
            _log.warning("data_update_partial", missing=len(failures))

        _log.info(
            "data_updated",
            symbols=len(symbols),
            written=written,
            start=begin.isoformat(),
            end=finish.isoformat(),
        )
        return UpdateReport(
            symbols=len(symbols),
            bars_written=written,
            start=begin,
            end=finish,
            source=self._source.name,
            failures=failures,
        )

    def _incremental_start(self, symbols: Sequence[Symbol], *, adjust: Adjust) -> TradeDate:
        """算增量起点。

        取各标的最后一天里**最早的那个**——只要有一只落后，就从它开始补，
        否则那只会永远缺一段。多拉的部分靠幂等写入合并掉。

        Args:
            symbols: 标的列表。
            adjust: 复权口径。

        Returns:
            起始日。
        """
        configured = dt.date.fromisoformat(self._settings.config.data.start_date)
        latest: list[TradeDate] = []
        for symbol in symbols:
            last = self._lake.last_trade_date(symbol=symbol, freq=Freq.D, adjust=adjust)
            if last is None:
                return configured  # 有全新标的就得从头拉
            latest.append(last)
        if not latest:
            return configured
        return min(latest) + dt.timedelta(days=1)

    def sync_instruments(self) -> int:
        """拉取并保存标的表。

        Returns:
            标的数量。

        Raises:
            DataError: 数据源不可用。
        """
        instruments = self._source.fetch_instruments()
        self._save_instruments(instruments)
        self._universe = UniverseRegistry(instruments)
        _log.info("instruments_synced", count=len(instruments))
        return len(instruments)

    def _save_instruments(self, instruments: Sequence[Instrument]) -> Path:
        """保存标的表。

        Args:
            instruments: 标的列表。

        Returns:
            文件路径。
        """
        import json  # noqa: PLC0415 - 仅此处需要

        from quantstock.infra.serde import to_jsonable  # noqa: PLC0415

        self._universe_path.parent.mkdir(parents=True, exist_ok=True)
        self._universe_path.write_text(
            json.dumps(
                {"instruments": [to_jsonable(i) for i in instruments]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return self._universe_path

    def load_instruments(self) -> list[Instrument]:
        """读回标的表。

        Returns:
            标的列表；未同步过时为空。
        """
        if not self._universe_path.exists():
            return []
        import json  # noqa: PLC0415 - 仅此处需要

        from quantstock.infra.serde import from_jsonable  # noqa: PLC0415

        payload = json.loads(self._universe_path.read_text(encoding="utf-8"))
        return [from_jsonable(Instrument, row) for row in payload.get("instruments", [])]

    def read_bars(
        self,
        symbols: Sequence[Symbol],
        *,
        start: TradeDate,
        end: TradeDate,
        adjust: Adjust | None = None,
    ) -> dict[Symbol, list[Bar]]:
        """从数据湖读取行情。

        Args:
            symbols: 标的列表。
            start: 起始日。
            end: 结束日。
            adjust: 复权口径；None 表示研究口径。

        Returns:
            标的 → K 线列表，按日期升序。
        """
        convention = adjust or Adjust(self._settings.config.data.adjust_for_research)
        out: dict[Symbol, list[Bar]] = {}
        for symbol in symbols:
            bars = self._lake.read_bars(
                [symbol], freq=Freq.D, adjust=convention, start=start, end=end
            )
            if bars:
                out[symbol] = sorted(bars, key=lambda b: b.trade_date)
        return out

    def status(self) -> DataStatus:
        """数据湖状态。

        Returns:
            状态快照。
        """
        stats = self._lake.stats()
        instruments = self.load_instruments()
        symbols = self._lake.available_symbols(
            freq=Freq.D, adjust=Adjust(self._settings.config.data.adjust_for_research)
        )
        latest: TradeDate | None = None
        for symbol in symbols:
            last = self._lake.last_trade_date(
                symbol=symbol,
                freq=Freq.D,
                adjust=Adjust(self._settings.config.data.adjust_for_research),
            )
            if last is not None and (latest is None or last > latest):
                latest = last

        return DataStatus(
            root=self._lake.root,
            symbols=len(symbols),
            files=stats.get("files", 0),
            bytes_on_disk=stats.get("bytes", 0),
            latest_date=latest,
            instruments=len(instruments),
            delisted=sum(1 for i in instruments if i.delist_date is not None),
        )

    def health(self) -> SourceHealth:
        """数据源健康度。

        Returns:
            健康度。
        """
        return self._source.health_check()
