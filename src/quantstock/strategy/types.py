"""策略层的数据契约。

规范见 docs/03-功能规格.md F3、docs/02-系统架构.md 第五节。

多周期融合::

    最终目标权重 = 总仓位中枢(LONG) × 个股相对权重(MEDIUM) × 择时系数(SHORT)

三层解耦，可分别回测、分别归因，任一层可单独关闭。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, runtime_checkable

from quantstock.backtest.engine import MarketView
from quantstock.infra.types import Direction, Horizon, Symbol, TradeDate

__all__ = [
    "Evidence",
    "Signal",
    "Strategy",
    "StrategyContext",
]


@dataclass(frozen=True, slots=True)
class Evidence:
    """一条量化依据。

    进入建议解释支柱①。必须带上具体数值与分位——
    "动量不错"不是依据，"momentum_60d = 0.18，处于全市场 87% 分位"才是。
    """

    factor: str
    value: float
    rank_pct: float
    """横截面分位 0~1。"""
    contribution: float
    """对综合得分的贡献。"""
    statement: str
    """人类可读的一句话。"""


@dataclass(frozen=True, slots=True)
class Signal:
    """策略输出的信号。

    信号只表达"看多/看空的程度"，**不含数量与价格**——
    仓位由 ``portfolio`` 层根据实际资金与约束决定（见模块职责边界）。
    """

    symbol: Symbol
    trade_date: TradeDate
    direction: Direction
    score: float
    """横截面可比的打分，越高越看好。"""
    confidence: float
    """置信度 0~1。"""
    horizon: Horizon
    strategy_id: str
    strategy_version: str
    evidence: tuple[Evidence, ...] = ()
    counter_evidence: tuple[Evidence, ...] = ()
    """反面证据。缺它的建议不完整（见 F7.3 支柱④）。"""
    rationale: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """校验取值范围。

        Raises:
            ValueError: 置信度越界。
        """
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"置信度必须在 [0, 1] 之间，收到 {self.confidence}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class StrategyContext:
    """策略运行上下文。

    ``market`` 是 PIT 安全的——只暴露 ``as_of`` 及之前的数据。
    策略拿不到未来 bar，这是引擎层面的保证（红线 R2）。
    """

    as_of: TradeDate
    market: MarketView
    universe: tuple[Symbol, ...]
    industries: dict[Symbol, str] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class Strategy(Protocol):
    """策略接口。

    每个策略必须在 ``docs/strategies/<id>.md`` 说明其经济学逻辑与失效条件——
    **没有经济学逻辑解释的策略不允许进入实盘候选池**（见开发规范第十二条）。
    """

    id: str
    version: str
    horizon: Horizon

    def required_lookback(self) -> int:
        """所需的历史交易日数。

        Returns:
            回看窗口长度。
        """
        ...

    def generate(self, ctx: StrategyContext) -> list[Signal]:
        """生成信号。

        Args:
            ctx: 运行上下文。

        Returns:
            信号列表。数据不足的标的应当被跳过而非给出默认值。
        """
        ...


@dataclass(frozen=True, slots=True)
class ExposureSignal:
    """LONG 层输出：总仓位中枢。

    决定"该拿几成仓"，与选哪些股完全解耦。
    """

    trade_date: TradeDate
    target_exposure: Decimal
    """目标权益仓位 0~1。"""
    rationale: tuple[str, ...] = ()
    strategy_id: str = ""

    def __post_init__(self) -> None:
        """校验仓位范围。

        Raises:
            ValueError: 仓位越界。
        """
        if not Decimal(0) <= self.target_exposure <= Decimal(1):
            msg = f"目标仓位必须在 [0, 1] 之间，收到 {self.target_exposure}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TimingSignal:
    """SHORT 层输出：择时系数。

    **只能降低权重不能提高**（保守优先，见 F3.3）——
    短周期择时的胜率通常不足以支撑加仓，但用来避险是划算的。
    """

    symbol: Symbol
    trade_date: TradeDate
    coefficient: Decimal
    """择时系数，取值 [0, 1]。"""
    rationale: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """校验系数范围。

        Raises:
            ValueError: 系数越界。
        """
        if not Decimal(0) <= self.coefficient <= Decimal(1):
            msg = f"择时系数必须在 [0, 1] 之间（只降不升），收到 {self.coefficient}"
            raise ValueError(msg)
