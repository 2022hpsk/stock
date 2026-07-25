"""复盘：计划-实际偏差与人工干预价值分析（D3）。

规范见 docs/08-差距分析与设计补强.md D3。

**这是半自动系统里最有价值、也最容易被跳过的一块分析。**

用户每天都在跳过一些建议。跳过时必须选原因（``SkipReason``），
积累一段时间后就能回答一个很难自己回答的问题：

- 「不认同逻辑」的跳过**长期跑赢**程序 → 策略有系统性缺陷，该改策略；
- 「不认同逻辑」的跳过**长期跑输**程序 → 该更信任程序，减少干预。

没有这份统计，人只会记住自己躲过的那几次大跌，忘掉错过的那些上涨——
这是最典型的确认偏误，而它会一路把半自动系统拖回全手动。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from statistics import fmean

from quantstock.infra.types import Money, Side, Symbol

__all__ = [
    "DeviationReport",
    "InterventionOutcome",
    "InterventionValue",
    "analyse_intervention_value",
    "build_deviation_report",
]

MIN_SAMPLES_FOR_VERDICT = 10
"""给出结论所需的最少样本数。

低于它只报数据不下结论——用 3 次跳过的结果去论证"该不该相信程序"，
得到的是噪声而不是洞察。
"""


@dataclass(frozen=True, slots=True)
class InterventionOutcome:
    """一次人工干预的事后结果。"""

    intent_id: str
    symbol: Symbol
    side: Side
    skip_reason: str
    suggested_price: Money
    """建议时的参考价。"""
    later_price: Money
    """评估时点的价格。"""
    horizon_days: int

    @property
    def forgone_return(self) -> Decimal:
        """如果当时执行了，这笔的收益率。

        买入建议被跳过：涨了就是错过；卖出建议被跳过：跌了就是错过。
        """
        if self.suggested_price <= 0:
            return Decimal(0)
        raw = (self.later_price - self.suggested_price) / self.suggested_price
        return raw if self.side is Side.BUY else -raw

    @property
    def skip_was_right(self) -> bool:
        """这次跳过事后看是否正确。"""
        return self.forgone_return < 0


@dataclass(frozen=True, slots=True)
class InterventionValue:
    """按原因分组的人工干预价值。"""

    reason: str
    count: int
    win_rate: float
    """跳过正确的比例。"""
    mean_forgone_return: Decimal
    """平均错过的收益。**为正说明跳过让你亏了钱**（错过了本该赚的）。"""
    total_forgone: Decimal

    @property
    def has_enough_samples(self) -> bool:
        """样本是否够下结论。"""
        return self.count >= MIN_SAMPLES_FOR_VERDICT

    @property
    def verdict(self) -> str:
        """结论。样本不足时明说不下结论。"""
        if not self.has_enough_samples:
            return f"样本仅 {self.count} 次，不足以下结论（需 ≥ {MIN_SAMPLES_FOR_VERDICT}）"
        if self.mean_forgone_return > 0:
            return "这类干预长期跑输程序，建议减少此类跳过"
        if self.mean_forgone_return < 0:
            return "这类干预长期跑赢程序，值得检查策略是否有系统性缺陷"
        return "这类干预与程序表现相当"

    def explain(self) -> str:
        """人类可读说明。

        Returns:
            说明文本。
        """
        return (
            f"[{self.reason}] {self.count} 次，跳对 {self.win_rate:.0%}，"
            f"平均错过收益 {self.mean_forgone_return:+.2%}。{self.verdict}"
        )


def analyse_intervention_value(
    outcomes: Sequence[InterventionOutcome],
) -> list[InterventionValue]:
    """按跳过原因分组统计人工干预的价值。

    Args:
        outcomes: 干预的事后结果。

    Returns:
        按样本数降序的分组统计。
    """
    buckets: dict[str, list[InterventionOutcome]] = {}
    for outcome in outcomes:
        buckets.setdefault(outcome.skip_reason, []).append(outcome)

    results = [
        InterventionValue(
            reason=reason,
            count=len(group),
            win_rate=sum(1 for o in group if o.skip_was_right) / len(group),
            mean_forgone_return=Decimal(str(fmean(float(o.forgone_return) for o in group))),
            total_forgone=sum((o.forgone_return for o in group), start=Decimal(0)),
        )
        for reason, group in buckets.items()
    ]
    return sorted(results, key=lambda r: (-r.count, r.reason))


@dataclass(frozen=True, slots=True)
class DeviationReport:
    """计划-实际偏差报告。"""

    trade_date: str
    planned: int
    executed: int
    skipped: int
    aborted: bool
    planned_amount: Money
    executed_amount: Money
    by_reason: dict[str, int]

    @property
    def execution_rate(self) -> float:
        """执行率。"""
        return self.executed / self.planned if self.planned else 0.0

    @property
    def amount_drift(self) -> Decimal:
        """实际与计划的金额偏差比例。"""
        if self.planned_amount <= 0:
            return Decimal(0)
        return (self.executed_amount - self.planned_amount) / self.planned_amount

    @property
    def needs_attention(self) -> bool:
        """是否需要关注。

        执行率长期偏低说明建议与用户判断系统性不一致——要么策略该改，
        要么用户该调整对系统的预期。哪一种都需要先被看见。
        """
        return self.aborted or (self.planned > 0 and self.execution_rate < 0.5)  # noqa: PLR2004

    def explain(self) -> str:
        """人类可读说明。

        Returns:
            说明文本。
        """
        if self.aborted:
            return f"{self.trade_date}：计划被中止，未提交任何订单"
        base = (
            f"{self.trade_date}：计划 {self.planned} 笔，执行 {self.executed} 笔"
            f"（{self.execution_rate:.0%}），跳过 {self.skipped} 笔"
        )
        if self.by_reason:
            detail = "、".join(f"{k}×{v}" for k, v in self.by_reason.items())
            return f"{base}；跳过原因：{detail}"
        return base


def build_deviation_report(
    *,
    trade_date: str,
    planned: int,
    executed: int,
    skipped: int,
    planned_amount: Money,
    executed_amount: Money,
    by_reason: dict[str, int] | None = None,
    aborted: bool = False,
) -> DeviationReport:
    """构造计划-实际偏差报告。

    Args:
        trade_date: 交易日。
        planned: 计划笔数。
        executed: 实际执行笔数。
        skipped: 跳过笔数。
        planned_amount: 计划金额。
        executed_amount: 实际金额。
        by_reason: 按原因的跳过次数。
        aborted: 计划是否被中止。

    Returns:
        偏差报告。
    """
    return DeviationReport(
        trade_date=trade_date,
        planned=planned,
        executed=executed,
        skipped=skipped,
        aborted=aborted,
        planned_amount=planned_amount,
        executed_amount=executed_amount,
        by_reason=dict(by_reason or {}),
    )
