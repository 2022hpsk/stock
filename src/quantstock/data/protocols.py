"""数据源抽象与降级链。

规范见 docs/02-系统架构.md 第六、七节。

**原则：数据不可信时拒绝出建议，而不是用降级数据硬出建议。**
降级链逐源尝试，全部失败时抛异常让上层停机，绝不返回半成品数据。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from quantstock.data.types import Bar, Instrument, SourceHealth
from quantstock.infra.clock import now
from quantstock.infra.errors import DataSourceError
from quantstock.infra.logging import get_logger
from quantstock.infra.types import Adjust, Symbol, TradeDate

__all__ = ["FallbackChain", "MarketDataSource"]

_log = get_logger(__name__)

T = TypeVar("T")


@runtime_checkable
class MarketDataSource(Protocol):
    """行情数据源。

    适配器必须在本层把外部代码格式归一化为标准 ``Symbol``，
    禁止让原始格式流出 ``data`` 层（见 docs/01-开发规范.md 第四条）。
    """

    name: str

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
            start: 起始交易日（含）。
            end: 结束交易日（含）。
            adjust: 复权口径。

        Returns:
            K 线列表。

        Raises:
            DataSourceError: 数据源不可用或返回异常。
        """
        ...

    def fetch_instruments(self) -> list[Instrument]:
        """拉取标的列表。

        **必须包含已退市标的**，否则会造成幸存者偏差。

        Returns:
            标的列表。

        Raises:
            DataSourceError: 数据源不可用。
        """
        ...

    def fetch_trading_days(self, *, start: TradeDate, end: TradeDate) -> list[TradeDate]:
        """拉取交易日历。

        Args:
            start: 起始日期。
            end: 结束日期。

        Returns:
            交易日列表。

        Raises:
            DataSourceError: 数据源不可用。
        """
        ...

    def health_check(self) -> SourceHealth:
        """探测数据源可用性。

        Returns:
            健康状态。
        """
        ...


class FallbackChain:
    """数据源降级链。

    按配置顺序逐源尝试，任一源成功即返回；全部失败时抛异常。

    **不做"部分成功就返回"**：半成品数据比没有数据更危险——
    它看起来是正常的，但会让策略基于残缺信息做决策。
    """

    def __init__(self, sources: Sequence[MarketDataSource]) -> None:
        """初始化。

        Args:
            sources: 按优先级排序的数据源。

        Raises:
            ValueError: 数据源列表为空。
        """
        if not sources:
            msg = "降级链至少需要一个数据源"
            raise ValueError(msg)
        self._sources = list(sources)
        self._failures: dict[str, int] = {s.name: 0 for s in sources}

    @property
    def source_names(self) -> tuple[str, ...]:
        """链上各源的名称，按优先级排序。"""
        return tuple(s.name for s in self._sources)

    def fetch_daily_bars(
        self,
        symbols: Sequence[Symbol],
        *,
        start: TradeDate,
        end: TradeDate,
        adjust: Adjust = Adjust.NONE,
    ) -> list[Bar]:
        """按降级链拉取日线。

        Args:
            symbols: 标的列表。
            start: 起始交易日。
            end: 结束交易日。
            adjust: 复权口径。

        Returns:
            K 线列表。

        Raises:
            DataSourceError: 全部数据源均失败。
        """
        return self._try_each(
            lambda source: source.fetch_daily_bars(symbols, start=start, end=end, adjust=adjust),
            operation="fetch_daily_bars",
            context={"symbols": len(symbols), "start": str(start), "end": str(end)},
        )

    def fetch_instruments(self) -> list[Instrument]:
        """按降级链拉取标的列表。

        Returns:
            标的列表。

        Raises:
            DataSourceError: 全部数据源均失败。
        """
        return self._try_each(
            lambda source: source.fetch_instruments(), operation="fetch_instruments"
        )

    def fetch_trading_days(self, *, start: TradeDate, end: TradeDate) -> list[TradeDate]:
        """按降级链拉取交易日历。

        Args:
            start: 起始日期。
            end: 结束日期。

        Returns:
            交易日列表。

        Raises:
            DataSourceError: 全部数据源均失败。
        """
        return self._try_each(
            lambda source: source.fetch_trading_days(start=start, end=end),
            operation="fetch_trading_days",
        )

    def health(self) -> list[SourceHealth]:
        """逐源探测健康状态，供界面"数据源健康"面板展示。

        Returns:
            各源的健康状态。
        """
        results: list[SourceHealth] = []
        for source in self._sources:
            try:
                health = source.health_check()
            except Exception as exc:  # 健康探测不应因单源异常中断
                health = SourceHealth(
                    name=source.name, ok=False, checked_at=now(), message=str(exc)
                )
            results.append(
                SourceHealth(
                    name=health.name,
                    ok=health.ok,
                    checked_at=health.checked_at,
                    message=health.message,
                    latency_ms=health.latency_ms,
                    consecutive_failures=self._failures.get(source.name, 0),
                )
            )
        return results

    def _try_each(
        self,
        call: Callable[[MarketDataSource], T],
        *,
        operation: str,
        context: dict[str, object] | None = None,
    ) -> T:
        """逐源尝试，全失败则抛异常。

        Args:
            call: 对单个数据源执行的操作。
            operation: 操作名，用于日志与错误信息。
            context: 附加上下文。

        Returns:
            首个成功源的返回值。

        Raises:
            DataSourceError: 全部数据源均失败。
        """
        errors: dict[str, str] = {}
        for source in self._sources:
            try:
                result = call(source)
            except Exception as exc:  # 任何异常都应触发降级
                self._failures[source.name] += 1
                errors[source.name] = str(exc)
                _log.warning(
                    "source_failed_falling_back",
                    source=source.name,
                    operation=operation,
                    error=str(exc),
                )
                continue
            self._failures[source.name] = 0
            if source is not self._sources[0]:
                _log.warning("served_by_fallback_source", source=source.name, operation=operation)
            return result

        msg = "全部数据源均失败，当日拒绝出建议"
        raise DataSourceError(msg, operation=operation, errors=errors, **(context or {}))
