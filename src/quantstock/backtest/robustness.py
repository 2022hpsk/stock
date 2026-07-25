"""稳健性检验与容量分析（A6、A7）。

规范见 docs/08-差距分析与设计补强.md A6、A7。

三块内容：

- **参数敏感性**：最优参数邻域的表现分布。尖峰与高原的差别在这里量化。
- **成本敏感性**：把成本乘以 1.5×、2× 后策略还剩多少。一个只在零成本假设下
  成立的策略是不存在的策略。
- **策略容量**（A7）：在给定冲击成本容忍度下能承载的最大资金量。小资金无感，
  但资金增长到千万级时，小盘股策略会因冲击成本失效——而那时候才发现就晚了。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from statistics import fmean, stdev

from quantstock.infra.money import money
from quantstock.infra.types import Money

__all__ = [
    "CapacityEstimate",
    "CostSensitivity",
    "SensitivityReport",
    "cost_sensitivity",
    "estimate_capacity",
    "parameter_sensitivity",
]

DEFAULT_IMPACT_TOLERANCE = 0.002
"""默认可容忍的单边冲击成本：20 bp。超过它策略的边际收益基本被吃掉。"""

DEFAULT_PARTICIPATION = 0.10
"""默认参与率上限：单日成交不超过该标的日均成交额的 10%。

再高就不只是"付出冲击成本"，而是自己在推动价格——回测里的成交价假设
在那种量级下完全不成立。
"""


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    """参数敏感性报告。"""

    best_value: float
    neighbourhood_mean: float
    neighbourhood_std: float
    worst_in_neighbourhood: float
    is_plateau: bool
    """邻域均值为正且最差值不至于太糟，才算高原。"""

    @property
    def degradation(self) -> float:
        """从最优点到邻域均值的衰减比例。"""
        if self.best_value == 0:
            return 0.0
        return (self.best_value - self.neighbourhood_mean) / abs(self.best_value)

    def explain(self) -> str:
        """人类可读结论。

        Returns:
            结论文本。
        """
        shape = "参数高原" if self.is_plateau else "参数尖峰（危险）"
        return (
            f"{shape}：最优 {self.best_value:.3f}，邻域均值 {self.neighbourhood_mean:.3f}"
            f"（衰减 {self.degradation:.1%}），邻域最差 {self.worst_in_neighbourhood:.3f}"
        )


def parameter_sensitivity(
    best_value: float,
    neighbourhood: Sequence[float],
    *,
    max_degradation: float = 0.5,
) -> SensitivityReport:
    """参数敏感性分析。

    ``max_degradation`` 默认 0.5：邻域均值跌掉一半以上就判为尖峰。
    这个阈值偏宽松是刻意的——真实策略的参数曲面本来就不平坦，
    卡太严会把所有策略都判死，那样这个检验就没人看了。

    Args:
        best_value: 最优参数点的表现。
        neighbourhood: 邻域内（默认 ±20%）各点的表现。
        max_degradation: 可容忍的衰减比例。

    Returns:
        敏感性报告。

    Raises:
        ValueError: 邻域为空。
    """
    values = list(neighbourhood)
    if not values:
        msg = "邻域不能为空——没做检验不等于通过检验"
        raise ValueError(msg)

    mean = fmean(values)
    spread = stdev(values) if len(values) > 1 else 0.0
    worst = min(values)

    degradation = (best_value - mean) / abs(best_value) if best_value else 0.0
    is_plateau = mean > 0 and degradation <= max_degradation

    return SensitivityReport(
        best_value=best_value,
        neighbourhood_mean=mean,
        neighbourhood_std=spread,
        worst_in_neighbourhood=worst,
        is_plateau=is_plateau,
    )


@dataclass(frozen=True, slots=True)
class CostSensitivity:
    """成本敏感性结果。"""

    multipliers: tuple[float, ...]
    returns: tuple[float, ...]
    breakeven_multiplier: float | None
    """成本放大到多少倍时收益归零。None 表示扫描范围内始终为正。"""

    @property
    def survives_double_cost(self) -> bool:
        """成本翻倍后是否仍为正。

        这是一条经验性的及格线：真实成本比回测假设高一倍是常见的，
        撑不过 2× 的策略上线后大概率是负收益。
        """
        for multiplier, value in zip(self.multipliers, self.returns, strict=True):
            if multiplier >= 2.0 and value <= 0:  # noqa: PLR2004 - 2× 就是这条经验线
                return False
        return True

    def explain(self) -> str:
        """人类可读结论。

        Returns:
            结论文本。
        """
        if self.breakeven_multiplier is None:
            return f"成本放大 {max(self.multipliers):.1f}× 后收益仍为正"
        return f"成本放大到 {self.breakeven_multiplier:.2f}× 时收益归零"


def cost_sensitivity(
    gross_return: float,
    cost_per_turn: float,
    turnover: float,
    *,
    multipliers: Sequence[float] = (1.0, 1.5, 2.0, 3.0),
) -> CostSensitivity:
    """成本敏感性扫描。

    Args:
        gross_return: 未扣成本的收益。
        cost_per_turn: 单次换手的单边成本率。
        turnover: 期内换手次数（双边计）。
        multipliers: 成本放大倍数。

    Returns:
        敏感性结果。
    """
    base_cost = cost_per_turn * turnover
    returns = tuple(gross_return - base_cost * m for m in multipliers)

    breakeven: float | None = None
    if base_cost > 0:
        candidate = gross_return / base_cost
        if candidate <= max(multipliers):
            breakeven = candidate

    return CostSensitivity(
        multipliers=tuple(multipliers), returns=returns, breakeven_multiplier=breakeven
    )


@dataclass(frozen=True, slots=True)
class CapacityEstimate:
    """策略容量估计（A7）。"""

    max_capital: Money
    binding_symbol: str
    """最先触及约束的标的——容量由最难交易的那只决定，不是平均值。"""
    current_capital: Money
    impact_tolerance: float
    participation_limit: float

    @property
    def utilisation(self) -> float:
        """当前资金占容量上限的比例。日报里要提示这个数。"""
        if self.max_capital <= 0:
            return 1.0
        return float(self.current_capital / self.max_capital)

    @property
    def is_constrained(self) -> bool:
        """是否已经接近容量上限。"""
        return self.utilisation >= 0.8  # noqa: PLR2004 - 80% 是开始要注意的位置

    def explain(self) -> str:
        """人类可读结论。

        Returns:
            结论文本。
        """
        warn = "，[已接近上限]" if self.is_constrained else ""
        return (
            f"策略容量约 {self.max_capital:,.0f} 元，当前资金占 {self.utilisation:.1%}"
            f"（瓶颈标的 {self.binding_symbol}）{warn}"
        )


def estimate_capacity(
    *,
    weights: dict[str, float],
    adv: dict[str, Money],
    current_capital: Money,
    impact_tolerance: float = DEFAULT_IMPACT_TOLERANCE,
    participation_limit: float = DEFAULT_PARTICIPATION,
    impact_coefficient: float = 0.1,
) -> CapacityEstimate:
    """估算策略容量。

    用平方根冲击模型 ``impact = k · √(order / ADV)``：这是市场微结构文献里
    最稳健的经验形式，比线性模型更贴近实际——冲击成本随规模是次线性增长的。

    两个约束取更严的那个：

    1. **冲击成本约束**：单笔冲击不超过 ``impact_tolerance``；
    2. **参与率约束**：单笔不超过 ADV 的 ``participation_limit``。

    容量由**最紧的那只标的**决定而不是平均值——组合里有一只买不进去，
    整个组合就建不起来。

    Args:
        weights: 标的 → 目标权重。
        adv: 标的 → 日均成交额（元）。
        current_capital: 当前资金。
        impact_tolerance: 可容忍的单边冲击成本率。
        participation_limit: 参与率上限。
        impact_coefficient: 冲击模型系数 k。

    Returns:
        容量估计。

    Raises:
        ValueError: 权重为空或含非正权重。
    """
    active = {s: w for s, w in weights.items() if w > 0}
    if not active:
        msg = "权重为空，无法估算容量"
        raise ValueError(msg)

    best_capital: Decimal | None = None
    binding = ""

    for symbol, weight in active.items():
        daily_volume = adv.get(symbol)
        if daily_volume is None or daily_volume <= 0:
            # 没有成交额数据的标的按容量为 0 处理：宁可低估也不要高估，
            # 高估容量的代价是真金白银的冲击成本
            best_capital = money("0")
            binding = symbol
            break

        # 冲击约束：k·√(order/ADV) ≤ tol → order ≤ ADV·(tol/k)²
        impact_order = float(daily_volume) * (impact_tolerance / impact_coefficient) ** 2
        # 参与率约束
        participation_order = float(daily_volume) * participation_limit
        allowed_order = min(impact_order, participation_order)

        capital_here = money(str(allowed_order / weight))
        if best_capital is None or capital_here < best_capital:
            best_capital = capital_here
            binding = symbol

    assert best_capital is not None  # noqa: S101 - active 非空保证至少走一轮
    return CapacityEstimate(
        max_capital=best_capital,
        binding_symbol=binding,
        current_capital=current_capital,
        impact_tolerance=impact_tolerance,
        participation_limit=participation_limit,
    )


def annualised_sharpe(returns: Sequence[float], *, periods_per_year: int = 252) -> float:
    """年化 Sharpe。

    Args:
        returns: 逐期收益。
        periods_per_year: 每年期数。

    Returns:
        年化 Sharpe；波动为 0 或样本不足时为 0。
    """
    if len(returns) < 2:  # noqa: PLR2004 - 标准差至少需要两个观测
        return 0.0
    spread = stdev(returns)
    if spread <= 0:
        return 0.0
    return fmean(returns) / spread * math.sqrt(periods_per_year)
