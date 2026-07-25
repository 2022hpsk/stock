"""数据质量校验（DQ01–DQ11）。

规范见 docs/04-数据规格.md 第六节。

分级处理：
- **致命级**（DQ01–DQ04、DQ11）：拒绝入库或拒绝回测。
- **严重级**：告警，并把有问题的标的移出当日 universe。
- **一般级**：告警，记入校验报告。

原则：**数据不可信时拒绝出建议，而不是用降级数据硬出建议。**
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from quantstock.data.types import Bar
from quantstock.infra.errors import DataQualityError
from quantstock.infra.types import Symbol, TradeDate

__all__ = [
    "QualityChecker",
    "QualityReport",
    "RuleResult",
    "Severity",
]


class Severity(StrEnum):
    """校验项严重级别。"""

    FATAL = "fatal"
    """拒绝入库。"""
    SERIOUS = "serious"
    """告警并剔除问题标的。"""
    MINOR = "minor"
    """仅记录。"""


_RULE_SEVERITY: dict[str, Severity] = {
    "DQ01": Severity.FATAL,  # 价格关系 low ≤ open,close ≤ high
    "DQ02": Severity.FATAL,  # 值域：价格为正、量额非负
    "DQ03": Severity.FATAL,  # 主键唯一
    "DQ04": Severity.FATAL,  # 交易日无整日缺失
    "DQ05": Severity.SERIOUS,  # 覆盖率
    "DQ06": Severity.SERIOUS,  # 涨跌幅超限
    "DQ07": Severity.SERIOUS,  # 复权一致性
    "DQ08": Severity.MINOR,  # 交叉源比对
    "DQ09": Severity.MINOR,  # 财务公告日合理性
    "DQ10": Severity.MINOR,  # 因子缺失率
    "DQ11": Severity.FATAL,  # 幸存者偏差
}

_RULE_DESC: dict[str, str] = {
    "DQ01": "价格关系必须满足 low ≤ min(open,close) ≤ max(open,close) ≤ high",
    "DQ02": "价格必须为正，成交量与成交额非负",
    "DQ03": "主键 (symbol, dt, freq, adjust) 必须唯一",
    "DQ04": "交易日历中的交易日不得整日缺失",
    "DQ05": "活跃标的当日数据覆盖率不得低于阈值",
    "DQ06": "日涨跌幅不得超过该标的当期涨跌幅限制",
    "DQ07": "后复权与不复权的日收益率在非除权日必须一致",
    "DQ08": "与备用数据源抽样比对的收盘价偏差不得超阈值",
    "DQ09": "财务数据公告日不得早于报告期结束日",
    "DQ10": "因子列不得全为空，单因子缺失率不得过高",
    "DQ11": "历史股票池必须包含已退市标的（防幸存者偏差）",
}


@dataclass(frozen=True, slots=True)
class RuleResult:
    """单条校验规则的结果。"""

    rule: str
    passed: bool
    severity: Severity
    message: str
    offenders: tuple[str, ...] = ()
    """违规的标的或日期，最多保留前若干条用于排查。"""

    @property
    def description(self) -> str:
        """规则说明。"""
        return _RULE_DESC.get(self.rule, "")


@dataclass(frozen=True, slots=True)
class QualityReport:
    """一批数据的完整校验报告。"""

    checked_at: dt.datetime
    trade_date: TradeDate | None
    total_bars: int
    results: tuple[RuleResult, ...] = field(default_factory=tuple)

    @property
    def failures(self) -> tuple[RuleResult, ...]:
        """未通过的规则。"""
        return tuple(r for r in self.results if not r.passed)

    @property
    def fatal_failures(self) -> tuple[RuleResult, ...]:
        """致命级失败。存在即必须拒绝入库。"""
        return tuple(r for r in self.failures if r.severity is Severity.FATAL)

    @property
    def passed(self) -> bool:
        """是否可以入库（无致命级失败）。"""
        return not self.fatal_failures

    @property
    def bad_symbols(self) -> frozenset[str]:
        """需要移出当日 universe 的标的。"""
        return frozenset(
            offender
            for r in self.failures
            if r.severity in {Severity.FATAL, Severity.SERIOUS}
            for offender in r.offenders
        )

    def raise_if_fatal(self) -> None:
        """存在致命级失败时抛异常。

        Raises:
            DataQualityError: 存在致命级失败。
        """
        fatal = self.fatal_failures
        if not fatal:
            return
        msg = "数据质量校验未通过，拒绝入库"
        raise DataQualityError(
            msg,
            rules=[r.rule for r in fatal],
            details=[r.message for r in fatal],
        )


_MAX_OFFENDERS = 20
"""报告中保留的违规样本上限——排查用，不需要全量。"""


class QualityChecker:
    """数据质量校验器。

    Attributes:
        coverage_threshold: 覆盖率下限（DQ05）。
        cross_check_max_deviation: 交叉比对偏差上限（DQ08）。
    """

    def __init__(
        self,
        *,
        coverage_threshold: float = 0.99,
        cross_check_max_deviation: float = 0.005,
        return_tolerance: Decimal = Decimal("1e-6"),
    ) -> None:
        """初始化。

        Args:
            coverage_threshold: 活跃标的覆盖率下限。
            cross_check_max_deviation: 与备用源比对的最大允许偏差。
            return_tolerance: 复权一致性校验的收益率容差。
        """
        self._coverage_threshold = Decimal(str(coverage_threshold))
        self._cross_check_max_deviation = Decimal(str(cross_check_max_deviation))
        self._return_tolerance = return_tolerance

    def check_bars(
        self,
        bars: Sequence[Bar],
        *,
        checked_at: dt.datetime,
        trade_date: TradeDate | None = None,
        expected_symbols: Iterable[Symbol] = (),
        price_limits: dict[Symbol, Decimal] | None = None,
    ) -> QualityReport:
        """校验一批 K 线。

        Args:
            bars: 待校验的 K 线。
            checked_at: 校验时刻。
            trade_date: 该批数据所属交易日。
            expected_symbols: 期望覆盖的标的，用于 DQ05 覆盖率校验。
            price_limits: 各标的当期涨跌幅限制，用于 DQ06。

        Returns:
            校验报告。
        """
        results = [
            self._check_price_relation(bars),
            self._check_value_range(bars),
            self._check_uniqueness(bars),
            self._check_coverage(bars, expected_symbols),
        ]
        if price_limits:
            results.append(self._check_price_limit(bars, price_limits))
        return QualityReport(
            checked_at=checked_at,
            trade_date=trade_date,
            total_bars=len(bars),
            results=tuple(results),
        )

    # ------------------------------------------------------------------ 各规则
    @staticmethod
    def _check_price_relation(bars: Sequence[Bar]) -> RuleResult:
        """DQ01：价格关系。"""
        offenders = [b.symbol for b in bars if "DQ01" in b.validate()]
        return _build("DQ01", offenders, f"{len(offenders)} 根 K 线的高低开收关系非法")

    @staticmethod
    def _check_value_range(bars: Sequence[Bar]) -> RuleResult:
        """DQ02：值域。"""
        offenders = [b.symbol for b in bars if "DQ02" in b.validate()]
        return _build("DQ02", offenders, f"{len(offenders)} 根 K 线的价格或量额超出合法范围")

    @staticmethod
    def _check_uniqueness(bars: Sequence[Bar]) -> RuleResult:
        """DQ03：主键唯一。

        重复行会让后续 join 静默膨胀，是最难排查的一类数据问题。
        """
        seen: dict[tuple[Symbol, dt.datetime, str, str], int] = defaultdict(int)
        for bar in bars:
            seen[(bar.symbol, bar.dt, bar.freq.value, bar.adjust.value)] += 1
        offenders = [f"{key[0]}@{key[1].date()}" for key, count in seen.items() if count > 1]
        return _build("DQ03", offenders, f"{len(offenders)} 个主键出现重复")

    def _check_coverage(
        self, bars: Sequence[Bar], expected_symbols: Iterable[Symbol]
    ) -> RuleResult:
        """DQ05：覆盖率。"""
        expected = set(expected_symbols)
        if not expected:
            return RuleResult(
                rule="DQ05",
                passed=True,
                severity=_RULE_SEVERITY["DQ05"],
                message="未提供期望标的清单，跳过覆盖率校验",
            )
        actual = {b.symbol for b in bars}
        missing = sorted(expected - actual)
        coverage = Decimal(len(actual & expected)) / Decimal(len(expected))
        passed = coverage >= self._coverage_threshold
        return RuleResult(
            rule="DQ05",
            passed=passed,
            severity=_RULE_SEVERITY["DQ05"],
            message=(
                f"覆盖率 {coverage:.2%}（阈值 {self._coverage_threshold:.2%}），"
                f"缺失 {len(missing)} 只"
            ),
            offenders=tuple(missing[:_MAX_OFFENDERS]),
        )

    @staticmethod
    def _check_price_limit(bars: Sequence[Bar], price_limits: dict[Symbol, Decimal]) -> RuleResult:
        """DQ06：涨跌幅超限。

        超出涨跌幅限制通常意味着**除权未被正确处理**，而不是行情真的异动。
        """
        offenders: list[str] = []
        for bar in bars:
            limit = price_limits.get(bar.symbol)
            if limit is None or bar.pre_close <= 0:
                continue
            # 留 0.5% 余量：涨跌停价本身要四舍五入到分，边界上会有微小偏差
            if abs(bar.change_pct) > limit * Decimal("1.005"):
                offenders.append(f"{bar.symbol}@{bar.trade_date}:{bar.change_pct:.2%}")
        return _build(
            "DQ06",
            offenders,
            f"{len(offenders)} 根 K 线涨跌幅超出板块限制（多为除权未处理）",
        )

    def check_calendar_continuity(
        self, *, actual_dates: Iterable[TradeDate], expected_dates: Iterable[TradeDate]
    ) -> RuleResult:
        """DQ04：交易日无整日缺失。

        Args:
            actual_dates: 数据中实际出现的交易日。
            expected_dates: 交易日历给出的应有交易日。

        Returns:
            校验结果。
        """
        missing = sorted(set(expected_dates) - set(actual_dates))
        return _build(
            "DQ04",
            [str(d) for d in missing],
            f"{len(missing)} 个交易日整日缺失",
        )

    def check_cross_source(
        self,
        *,
        primary: dict[Symbol, Decimal],
        secondary: dict[Symbol, Decimal],
    ) -> RuleResult:
        """DQ08：与备用数据源抽样比对收盘价。

        Args:
            primary: 主源的收盘价。
            secondary: 备用源的收盘价。

        Returns:
            校验结果。
        """
        offenders: list[str] = []
        for symbol, price in primary.items():
            other = secondary.get(symbol)
            if other is None or other <= 0:
                continue
            deviation = abs(price - other) / other
            if deviation > self._cross_check_max_deviation:
                offenders.append(f"{symbol}:{deviation:.3%}")
        return _build(
            "DQ08",
            offenders,
            f"{len(offenders)} 只标的与备用源收盘价偏差超阈值",
        )

    def check_announcement_dates(
        self, records: Iterable[tuple[Symbol, TradeDate, TradeDate]]
    ) -> RuleResult:
        """DQ09：财务数据公告日不得早于报告期结束日。

        公告日早于报告期结束，意味着 PIT 口径出错——
        会让回测提前看到当时还不存在的财务数据（红线 R2）。

        Args:
            records: ``(symbol, report_period_end, ann_date)`` 三元组。

        Returns:
            校验结果。
        """
        offenders = [
            f"{symbol}:报告期{period}/公告日{ann}"
            for symbol, period, ann in records
            if ann < period
        ]
        return _build(
            "DQ09",
            offenders,
            f"{len(offenders)} 条财务记录的公告日早于报告期结束日（PIT 口径错误）",
        )


def _build(rule: str, offenders: Sequence[str], message: str) -> RuleResult:
    """构造校验结果。

    Args:
        rule: 规则编号。
        offenders: 违规样本。
        message: 结果说明。

    Returns:
        校验结果。
    """
    return RuleResult(
        rule=rule,
        passed=not offenders,
        severity=_RULE_SEVERITY[rule],
        message=message if offenders else f"{rule} 通过",
        offenders=tuple(str(o) for o in offenders[:_MAX_OFFENDERS]),
    )
