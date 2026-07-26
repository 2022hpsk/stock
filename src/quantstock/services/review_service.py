"""复盘服务：计划-实际偏差与人工干预价值（docs/09 P12、docs/08 D3）。

**这个模块要回答的问题只有一个：你的人工干预到底是帮忙还是添乱。**

半自动系统里，人每天都在改程序的建议——跳过、改量、不执行。这些干预
凭感觉做，事后也凭感觉评价，于是永远不会变好。这里按跳过原因分组，
用**事后实际价格**来量化：

- 某类跳过长期跑赢程序 → 策略在这个维度上有系统性缺陷，该改策略；
- 长期跑输 → 该更信任程序，减少这类干预。

两个不肯妥协的地方：

1. **算不出就说算不出**。事后收益需要"跳过日价格"和"持有期后价格"两组行情，
   缺任一组就把该样本排除，而不是用 0 填充——0 会把统计悄悄拉向
   "干预没有影响"这个错误结论；
2. **样本不够就不给结论**。三次跳过里对了两次，胜率 67%，这个数字是噪声，
   但读者会当成信号。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from quantstock.advisor.store import PlanStore
from quantstock.config.settings import Settings
from quantstock.execution.report_store import ExecutionReportStore
from quantstock.infra.logging import get_logger
from quantstock.infra.money import money
from quantstock.infra.types import Side, Symbol, TradeDate
from quantstock.reporting.review import (
    DeviationReport,
    InterventionOutcome,
    InterventionValue,
    analyse_intervention_value,
    build_deviation_report,
)
from quantstock.services.data_service import DataService

# 界面与 CLI 只允许依赖 services（F20.1 分层契约），复盘契约类型在这里转出
__all__ = [
    "DeviationReport",
    "InterventionOutcome",
    "InterventionValue",
    "ReviewService",
    "ReviewSummary",
]

_log = get_logger(__name__)

MIN_SAMPLES_FOR_VERDICT = 10
"""给出"该更信任谁"结论所需的最少样本。

低于这个数只报样本量，不报结论。三次跳过里对了两次算出来的 67% 胜率，
读者会当成信号，而它其实是噪声。
"""

DEFAULT_HORIZON_DAYS = 20
"""事后评估的持有期（交易日）。约一个月，够让一次判断的对错显现，
又不至于长到被大盘走势完全淹没。"""


@dataclass(frozen=True, slots=True)
class ReviewSummary:
    """一段区间的复盘摘要。"""

    start: TradeDate
    end: TradeDate
    plans: int
    deviations: tuple[DeviationReport, ...]
    interventions: tuple[InterventionValue, ...]
    unpriced_skips: int
    """因缺少事后行情而无法评估的跳过项。**必须报出来**——
    否则用户会以为统计覆盖了全部干预。"""

    @property
    def total_planned(self) -> int:
        """计划总笔数。"""
        return sum(d.planned for d in self.deviations)

    @property
    def total_executed(self) -> int:
        """实际执行总笔数。"""
        return sum(d.executed for d in self.deviations)

    @property
    def total_skipped(self) -> int:
        """跳过总笔数。"""
        return sum(d.skipped for d in self.deviations)

    @property
    def execution_rate(self) -> float:
        """整体执行率。

        Returns:
            执行率；无计划时 0。
        """
        return self.total_executed / self.total_planned if self.total_planned else 0.0

    @property
    def sample_count(self) -> int:
        """已评估的干预样本数。"""
        return sum(i.count for i in self.interventions)

    @property
    def has_enough_samples(self) -> bool:
        """样本是否够给结论。"""
        return self.sample_count >= MIN_SAMPLES_FOR_VERDICT

    def explain(self) -> str:
        """人类可读结论。

        Returns:
            结论文本。
        """
        if not self.deviations:
            return f"{self.start} ~ {self.end} 区间内没有执行记录，无从复盘"

        base = (
            f"{self.start} ~ {self.end}：{self.plans} 份计划，"
            f"计划 {self.total_planned} 笔、执行 {self.total_executed} 笔、"
            f"跳过 {self.total_skipped} 笔（执行率 {self.execution_rate:.0%}）"
        )
        if self.unpriced_skips:
            base += f"；{self.unpriced_skips} 笔跳过因缺事后行情未纳入统计"
        if not self.has_enough_samples:
            return (
                f"{base}。已评估干预 {self.sample_count} 次，"
                f"不足 {MIN_SAMPLES_FOR_VERDICT} 次，暂不给出「该更信任谁」的结论"
                "——样本太少时算出的胜率是噪声"
            )
        return base


class ReviewService:
    """复盘编排。"""

    def __init__(self, settings: Settings, *, data: DataService | None = None) -> None:
        """初始化。

        Args:
            settings: 运行期配置。
            data: 数据服务，用于取事后价格。
        """
        self._settings = settings
        self._plans = PlanStore(settings.var_dir / "plans")
        self._reports = ExecutionReportStore(settings.var_dir / "executions")
        self._data = data or DataService(settings)

    def dates(self) -> list[TradeDate]:
        """有执行记录的日期。

        Returns:
            日期列表，升序。
        """
        return self._reports.list_dates()

    def deviation(self, trade_date: TradeDate) -> DeviationReport | None:
        """某日的计划-实际偏差。

        Args:
            trade_date: 交易日。

        Returns:
            偏差报告；该日没有执行记录时 None。
        """
        reports = self._reports.read(trade_date)
        if not reports:
            return None

        orders = [o for r in reports for o in r["orders"]]
        executed = [o for o in orders if o["status"] != "skipped"]
        skipped = [o for o in orders if o["status"] == "skipped"]

        by_reason: dict[str, int] = {}
        for order in skipped:
            reason = str(order.get("skip_reason") or "unknown")
            by_reason[reason] = by_reason.get(reason, 0) + 1

        plan = self._plans.latest(trade_date)
        planned_amount = (
            plan.total_buy_amount + plan.total_sell_amount if plan is not None else Decimal(0)
        )
        executed_amount = sum((money(o["price"]) * o["qty"] for o in executed), start=Decimal(0))

        return build_deviation_report(
            trade_date=trade_date.isoformat(),
            planned=len(orders),
            executed=len(executed),
            skipped=len(skipped),
            planned_amount=planned_amount,
            executed_amount=executed_amount,
            by_reason=by_reason,
            aborted=any(r["aborted"] for r in reports),
        )

    def summary(
        self,
        *,
        start: TradeDate,
        end: TradeDate,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
    ) -> ReviewSummary:
        """区间复盘。

        Args:
            start: 起始日。
            end: 结束日。
            horizon_days: 事后评估的持有期。

        Returns:
            复盘摘要。
        """
        dates = [d for d in self.dates() if start <= d <= end]
        reports = [r for d in dates if (r := self.deviation(d)) is not None]
        outcomes, unpriced = self._collect_outcomes(dates, horizon_days=horizon_days)

        return ReviewSummary(
            start=start,
            end=end,
            plans=len(dates),
            deviations=tuple(reports),
            interventions=tuple(analyse_intervention_value(outcomes) if outcomes else []),
            unpriced_skips=unpriced,
        )

    def intervention_value(
        self,
        *,
        start: TradeDate,
        end: TradeDate,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
    ) -> list[InterventionValue]:
        """按跳过原因分组统计人工干预的价值。

        Args:
            start: 起始日。
            end: 结束日。
            horizon_days: 事后评估的持有期（交易日）。

        Returns:
            分组统计，样本数降序。
        """
        dates = [d for d in self.dates() if start <= d <= end]
        outcomes, _ = self._collect_outcomes(dates, horizon_days=horizon_days)
        return analyse_intervention_value(outcomes) if outcomes else []

    def _collect_outcomes(
        self, dates: list[TradeDate], *, horizon_days: int
    ) -> tuple[list[InterventionOutcome], int]:
        """收集被跳过的意图及其事后结果。

        Args:
            dates: 要评估的交易日。
            horizon_days: 持有期（交易日）。

        Returns:
            ``(可评估的结果, 因缺行情被排除的条数)``。
        """
        outcomes: list[InterventionOutcome] = []
        unpriced = 0

        for date in dates:
            for report in self._reports.read(date):
                for order in report["orders"]:
                    if order["status"] != "skipped":
                        continue
                    outcome = self._to_outcome(order, date, horizon_days=horizon_days)
                    if outcome is None:
                        unpriced += 1
                    else:
                        outcomes.append(outcome)

        if unpriced:
            _log.info("intervention_samples_unpriced", excluded=unpriced, kept=len(outcomes))
        return outcomes, unpriced

    def _to_outcome(
        self, order: dict[str, Any], trade_date: TradeDate, *, horizon_days: int
    ) -> InterventionOutcome | None:
        """把一条被跳过的订单转成可评估的干预结果。

        Args:
            order: 订单字典。
            trade_date: 跳过发生的交易日。
            horizon_days: 持有期。

        Returns:
            干预结果；缺少任一端价格时 None——用 0 填充会把统计
            悄悄拉向"干预没有影响"这个错误结论。
        """
        symbol = Symbol(str(order["symbol"]))
        # 多取一段：horizon_days 是交易日，日历日要留出周末与节假日的余量
        window_end = trade_date + dt.timedelta(days=horizon_days * 2)
        bars = self._data.read_bars([symbol], start=trade_date, end=window_end).get(symbol, [])
        if len(bars) <= horizon_days:
            return None

        suggested = money(str(order["price"]))
        if suggested <= 0:
            return None

        return InterventionOutcome(
            intent_id=str(order["intent_id"]),
            symbol=symbol,
            side=Side(str(order["side"])),
            skip_reason=str(order.get("skip_reason") or "unknown"),
            suggested_price=suggested,
            later_price=bars[horizon_days].close,
            horizon_days=horizon_days,
        )
