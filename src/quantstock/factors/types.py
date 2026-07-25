"""因子层的数据契约。

规范见 docs/03-功能规格.md F2。

关键：**因子必须先通过有效性检验才允许进策略**。没有 IC 检验就上策略，
等于把噪声当信号（见 F2.4）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from quantstock.infra.types import Symbol, TradeDate

__all__ = [
    "FactorCategory",
    "FactorMeta",
    "FactorPanel",
    "FactorValue",
    "ICStats",
    "LayerStats",
]


_SIGNIFICANT_T_STAT = 2.0
"""统计显著性的经验判据：|t| > 2。"""


class FactorCategory(StrEnum):
    """因子类别。"""

    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MONEYFLOW = "moneyflow"
    SENTIMENT = "sentiment"
    INTEL = "intel"
    """情报软因子。影响被限制在 ±20% 以内，且须先通过 IC 检验（见 F12.10）。"""


@dataclass(frozen=True, slots=True)
class FactorMeta:
    """因子元信息。

    每个因子都必须注册元信息——``direction`` 尤其关键：
    忘了标注方向会让"越小越好"的因子被当成"越大越好"，符号一反，
    策略从赚钱变成亏钱，而回测曲线看上去仍然"有信号"。
    """

    name: str
    category: FactorCategory
    direction: int
    """+1 表示越大越好，-1 表示越小越好。"""
    lookback: int
    """计算所需的历史交易日数。"""
    description: str
    version: str = "v1"
    freq: str = "D"

    def __post_init__(self) -> None:
        """校验元信息合法。

        Raises:
            ValueError: 方向非 ±1 或回看期为负。
        """
        if self.direction not in (1, -1):
            msg = f"因子方向必须是 +1 或 -1，收到 {self.direction}"
            raise ValueError(msg)
        if self.lookback < 0:
            msg = f"回看期不能为负，收到 {self.lookback}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class FactorValue:
    """单个标的在某日的因子值。"""

    symbol: Symbol
    trade_date: TradeDate
    name: str
    raw: float
    """原始值。"""
    standardized: float = 0.0
    """横截面标准化后的值。"""
    rank_pct: float = 0.0
    """横截面分位（0~1），用于建议解释中的"处于全市场 87% 分位"。"""


@dataclass(frozen=True, slots=True)
class FactorPanel:
    """某一交易日的横截面因子面板。"""

    trade_date: TradeDate
    name: str
    values: dict[Symbol, float]

    def __len__(self) -> int:
        """标的数量。"""
        return len(self.values)

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        """面板覆盖的标的，升序。"""
        return tuple(sorted(self.values))

    def coverage(self, universe: Sequence[Symbol]) -> float:
        """相对给定股票池的覆盖率。

        Args:
            universe: 期望覆盖的标的。

        Returns:
            覆盖率 0~1；股票池为空时返回 0。
        """
        if not universe:
            return 0.0
        return len(set(universe) & set(self.values)) / len(universe)


@dataclass(frozen=True, slots=True)
class ICStats:
    """因子 IC 检验结果。

    ``ir`` （信息比率 = IC均值 / IC标准差）比 IC 均值本身更能说明问题：
    均值高但波动更高的因子并不可用。
    """

    name: str
    periods: int
    ic_mean: float
    ic_std: float
    ir: float
    positive_rate: float
    """IC 为正的期数占比。稳定在 0.5 附近说明因子无效。"""
    t_stat: float

    @property
    def is_significant(self) -> bool:
        """是否具备统计显著性（|t| > 2 的经验判据）。"""
        return abs(self.t_stat) > _SIGNIFICANT_T_STAT


@dataclass(frozen=True, slots=True)
class LayerStats:
    """分层回测结果。

    检验因子是否**单调**：从第 1 组到第 N 组收益应当递增（或递减）。
    只有头尾两组有差异、中间乱序的因子通常是噪声。
    """

    name: str
    layers: int
    mean_returns: tuple[float, ...]
    """各组的平均收益，按因子值从低到高排列。"""
    long_short_return: float
    """多空组合收益（最高组 - 最低组）。"""

    @property
    def is_monotonic(self) -> bool:
        """各层收益是否单调。"""
        increasing = all(
            self.mean_returns[i] <= self.mean_returns[i + 1]
            for i in range(len(self.mean_returns) - 1)
        )
        decreasing = all(
            self.mean_returns[i] >= self.mean_returns[i + 1]
            for i in range(len(self.mean_returns) - 1)
        )
        return increasing or decreasing


FactorFunc = Callable[..., float]
"""因子计算函数签名。"""
