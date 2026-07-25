"""内置策略。

规范见 docs/03-功能规格.md F3.2。

**这些是基线策略，不是已验证有效的策略。**能不能用要看回测——
DSR < 0 或 PBO > 0.5 禁止进实盘候选池。每个策略的经济学逻辑写在类的 docstring 里，
没有经济学逻辑解释的策略不允许进入实盘候选池。
"""

from __future__ import annotations

from decimal import Decimal

from quantstock.factors.pipeline import rank_pct
from quantstock.factors.technical import (
    drawdown_from_peak,
    momentum,
    moving_average,
    realized_volatility,
)
from quantstock.infra.types import Direction, Horizon, Symbol
from quantstock.strategy.types import (
    Evidence,
    ExposureSignal,
    Signal,
    StrategyContext,
    TimingSignal,
)

__all__ = [
    "EtfRotationStrategy",
    "MacroExposureStrategy",
    "MomentumTrendStrategy",
    "TimingOverlayStrategy",
    "blend_scores",
]

_LONG_SCORE_THRESHOLD = 0.5
"""打分高于该分位才给出 LONG 方向；低于则视为不参与。"""


class MomentumTrendStrategy:
    """中期动量 + 趋势确认（MEDIUM 层）。

    **经济学逻辑**：中期动量效应源于投资者对信息的反应不足与随后的羊群效应。
    单纯的动量因子在 A 股容易在震荡市反复挨打，因此叠加趋势确认——
    只有价格站上长期均线、且短期均线在长期均线之上时才认可动量信号。

    **失效条件**：风格急速切换的市场（如 2021 年初的抱团瓦解）；
    政策驱动的普涨普跌行情中趋势确认会显著滞后。
    """

    id = "momentum_trend"
    version = "v1"
    horizon = Horizon.MEDIUM

    def __init__(
        self,
        *,
        momentum_window: int = 60,
        skip_recent: int = 5,
        fast_ma: int = 20,
        slow_ma: int = 60,
    ) -> None:
        """初始化。

        Args:
            momentum_window: 动量回看窗口。
            skip_recent: 跳过最近若干日，避开短期反转效应。
            fast_ma: 短期均线窗口。
            slow_ma: 长期均线窗口。

        Raises:
            ValueError: 短期均线不短于长期均线。
        """
        if fast_ma >= slow_ma:
            msg = f"短期均线必须短于长期均线：{fast_ma} >= {slow_ma}"
            raise ValueError(msg)
        self._momentum_window = momentum_window
        self._skip_recent = skip_recent
        self._fast_ma = fast_ma
        self._slow_ma = slow_ma

    def required_lookback(self) -> int:
        """所需历史交易日数。"""
        return max(self._momentum_window + self._skip_recent + 1, self._slow_ma) + 1

    def generate(self, ctx: StrategyContext) -> list[Signal]:
        """生成动量信号。

        Args:
            ctx: 运行上下文。

        Returns:
            信号列表；数据不足的标的被跳过。
        """
        raw: dict[Symbol, float] = {}
        details: dict[Symbol, tuple[float, float, float]] = {}

        for symbol in ctx.universe:
            closes = ctx.market.closes(symbol)
            if len(closes) < self.required_lookback():
                continue
            try:
                mom = momentum(closes, self._momentum_window, skip_recent=self._skip_recent)
                fast = moving_average(closes, self._fast_ma)
                slow = moving_average(closes, self._slow_ma)
            except ValueError:
                continue
            raw[symbol] = mom
            details[symbol] = (mom, fast, slow)

        if not raw:
            return []

        ranks = rank_pct(raw)
        signals: list[Signal] = []
        for symbol, (mom, fast, slow) in details.items():
            trend_ok = fast > slow and ctx.market.closes(symbol)[-1] > slow
            rank = ranks[symbol]
            # 趋势不确认时打分打对折——不是直接归零，因为动量本身仍有信息
            score = rank if trend_ok else rank * 0.5

            evidence = [
                Evidence(
                    factor=f"momentum_{self._momentum_window}d",
                    value=mom,
                    rank_pct=rank,
                    contribution=score,
                    statement=(
                        f"{self._momentum_window} 日动量 {mom:+.2%}，处于当前股票池 {rank:.0%} 分位"
                    ),
                )
            ]
            counter: list[Evidence] = []
            if not trend_ok:
                counter.append(
                    Evidence(
                        factor="trend_confirm",
                        value=fast - slow,
                        rank_pct=0.0,
                        contribution=-rank * 0.5,
                        statement=(
                            f"趋势未确认：MA{self._fast_ma}={fast:.2f} 未站上 "
                            f"MA{self._slow_ma}={slow:.2f}，动量信号打对折"
                        ),
                    )
                )

            signals.append(
                Signal(
                    symbol=symbol,
                    trade_date=ctx.as_of,
                    direction=Direction.LONG if score > _LONG_SCORE_THRESHOLD else Direction.FLAT,
                    score=score,
                    confidence=0.6 if trend_ok else 0.35,
                    horizon=self.horizon,
                    strategy_id=self.id,
                    strategy_version=self.version,
                    evidence=tuple(evidence),
                    counter_evidence=tuple(counter),
                    rationale=(f"动量分位 {rank:.0%}，趋势{'已' if trend_ok else '未'}确认",),
                )
            )
        return signals


class EtfRotationStrategy:
    """宽基/行业 ETF 动量轮动（MEDIUM 层）。

    **经济学逻辑**：行业景气度具有持续性，资金流入领先行业会自我强化一段时间。
    ETF 无个股黑天鹅风险、流动性好、费用低（无印花税无过户费），
    是用动量思路最干净的载体。

    **失效条件**：行业快速轮动的震荡市会持续高买低卖；
    单边下跌市中"最强 ETF"仍然是亏钱的。因此必须与 LONG 层仓位控制配合使用。
    """

    id = "etf_rotation"
    version = "v1"
    horizon = Horizon.MEDIUM

    def __init__(self, *, lookback: int = 20, top_n: int = 3) -> None:
        """初始化。

        Args:
            lookback: 动量回看窗口。
            top_n: 持有排名前几的 ETF。

        Raises:
            ValueError: 参数非正。
        """
        if lookback <= 0 or top_n <= 0:
            msg = "lookback 与 top_n 必须为正"
            raise ValueError(msg)
        self._lookback = lookback
        self._top_n = top_n

    def required_lookback(self) -> int:
        """所需历史交易日数。"""
        return self._lookback + 1

    def generate(self, ctx: StrategyContext) -> list[Signal]:
        """生成轮动信号。

        Args:
            ctx: 运行上下文。

        Returns:
            信号列表；只有排名前 top_n 且动量为正的给出 LONG。
        """
        scores: dict[Symbol, float] = {}
        for symbol in ctx.universe:
            closes = ctx.market.closes(symbol)
            if len(closes) < self.required_lookback():
                continue
            try:
                scores[symbol] = momentum(closes, self._lookback)
            except ValueError:
                continue

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = {sym for sym, mom in ranked[: self._top_n] if mom > 0}
        ranks = rank_pct(scores)

        return [
            Signal(
                symbol=symbol,
                trade_date=ctx.as_of,
                direction=Direction.LONG if symbol in selected else Direction.FLAT,
                score=ranks[symbol],
                confidence=0.55,
                horizon=self.horizon,
                strategy_id=self.id,
                strategy_version=self.version,
                evidence=(
                    Evidence(
                        factor=f"etf_momentum_{self._lookback}d",
                        value=mom,
                        rank_pct=ranks[symbol],
                        contribution=ranks[symbol],
                        statement=f"{self._lookback} 日动量 {mom:+.2%}，池内排名 {rank + 1}",
                    ),
                ),
                counter_evidence=()
                if mom > 0
                else (
                    Evidence(
                        factor="momentum_sign",
                        value=mom,
                        rank_pct=ranks[symbol],
                        contribution=0.0,
                        statement="动量为负，即便排名靠前也不建仓",
                    ),
                ),
                rationale=(f"ETF 动量轮动，取前 {self._top_n} 强",),
            )
            for rank, (symbol, mom) in enumerate(ranked)
        ]


class MacroExposureStrategy:
    """总仓位中枢（LONG 层）。

    **经济学逻辑**：个人投资者的最大回撤主要来自"熊市里满仓"。
    与其在个股上博弈，不如先把总仓位控制住——用大盘长期均线状态与
    市场宽度（多少比例的标的处于上升趋势）判断系统性风险，据此决定权益暴露。

    **失效条件**：均线择时在震荡市会反复假信号，导致来回摩擦成本。
    因此仓位是**连续调整**而非 0/1 开关，且设有下限避免完全空仓错过反弹。
    """

    id = "macro_exposure"
    version = "v1"
    horizon = Horizon.LONG

    def __init__(
        self,
        *,
        trend_window: int = 120,
        min_exposure: Decimal = Decimal("0.2"),
        max_exposure: Decimal = Decimal("0.95"),
    ) -> None:
        """初始化。

        Args:
            trend_window: 判断大盘趋势的均线窗口。
            min_exposure: 仓位下限。设下限是为了避免完全空仓而错过反弹。
            max_exposure: 仓位上限。

        Raises:
            ValueError: 上下限倒置。
        """
        if min_exposure > max_exposure:
            msg = f"仓位下限不能高于上限：{min_exposure} > {max_exposure}"
            raise ValueError(msg)
        self._trend_window = trend_window
        self._min = min_exposure
        self._max = max_exposure

    def required_lookback(self) -> int:
        """所需历史交易日数。"""
        return self._trend_window + 1

    def generate_exposure(self, ctx: StrategyContext) -> ExposureSignal:
        """计算目标权益仓位。

        Args:
            ctx: 运行上下文。

        Returns:
            仓位信号。数据不足时返回下限仓位——**不确定时取保守一侧**。
        """
        above_ma = 0
        counted = 0
        for symbol in ctx.universe:
            closes = ctx.market.closes(symbol)
            if len(closes) < self.required_lookback():
                continue
            counted += 1
            if closes[-1] > moving_average(closes, self._trend_window):
                above_ma += 1

        if counted == 0:
            return ExposureSignal(
                trade_date=ctx.as_of,
                target_exposure=self._min,
                rationale=("数据不足以判断市场状态，取仓位下限（不确定时保守）",),
                strategy_id=self.id,
            )

        breadth = above_ma / counted
        span = self._max - self._min
        exposure = self._min + span * Decimal(str(breadth))
        return ExposureSignal(
            trade_date=ctx.as_of,
            target_exposure=exposure.quantize(Decimal("0.01")),
            rationale=(
                f"市场宽度 {breadth:.0%}（{above_ma}/{counted} 只站上 "
                f"MA{self._trend_window}），目标权益仓位 {exposure:.0%}",
            ),
            strategy_id=self.id,
        )


class TimingOverlayStrategy:
    """择时叠加（SHORT 层）。

    **经济学逻辑**：趋势破位后的短期下跌具有延续性，及时降权可以显著改善回撤。
    但短周期择时的胜率不足以支撑加仓，因此**只降不升**——
    这是不对称的：做错时只是少赚，做对时能少亏。

    **失效条件**：快速 V 型反转中会在底部降权、错过反弹。
    """

    id = "timing_overlay"
    version = "v1"
    horizon = Horizon.SHORT

    def __init__(
        self,
        *,
        ma_window: int = 20,
        drawdown_threshold: float = -0.15,
        vol_window: int = 20,
        high_vol_threshold: float = 0.6,
    ) -> None:
        """初始化。

        Args:
            ma_window: 破位判断的均线窗口。
            drawdown_threshold: 回撤阈值，超过则降权。
            vol_window: 波动率窗口。
            high_vol_threshold: 年化波动率高于此值时降权。
        """
        self._ma_window = ma_window
        self._drawdown_threshold = drawdown_threshold
        self._vol_window = vol_window
        self._high_vol_threshold = high_vol_threshold

    def required_lookback(self) -> int:
        """所需历史交易日数。"""
        return max(self._ma_window, self._vol_window) + 1

    def generate_timing(self, ctx: StrategyContext) -> dict[Symbol, TimingSignal]:
        """计算各标的的择时系数。

        Args:
            ctx: 运行上下文。

        Returns:
            标的到择时系数的映射。数据不足的标的系数为 1（不干预）。
        """
        result: dict[Symbol, TimingSignal] = {}
        for symbol in ctx.universe:
            closes = ctx.market.closes(symbol)
            if len(closes) < self.required_lookback():
                continue

            coefficient = Decimal("1.0")
            reasons: list[str] = []

            ma = moving_average(closes, self._ma_window)
            if closes[-1] < ma:
                coefficient *= Decimal("0.7")
                reasons.append(f"跌破 MA{self._ma_window}（{ma:.2f}），降权至 70%")

            dd = drawdown_from_peak(closes, self._vol_window)
            if dd < self._drawdown_threshold:
                coefficient *= Decimal("0.7")
                reasons.append(f"近期回撤 {dd:.1%} 超阈值，再降权至 70%")

            try:
                vol = realized_volatility(closes, self._vol_window)
            except ValueError:
                vol = 0.0
            if vol > self._high_vol_threshold:
                coefficient *= Decimal("0.8")
                reasons.append(f"年化波动率 {vol:.0%} 偏高，降权至 80%")

            result[symbol] = TimingSignal(
                symbol=symbol,
                trade_date=ctx.as_of,
                coefficient=coefficient.quantize(Decimal("0.01")),
                rationale=tuple(reasons) if reasons else ("技术面正常，不做择时干预",),
            )
        return result


def blend_scores(
    signals_by_strategy: dict[str, list[Signal]], weights: dict[str, float]
) -> dict[Symbol, float]:
    """多策略打分融合（F3.3）。

    按配置权重加权。缺失该标的信号的策略不参与该标的的加权——
    用 0 填充会把"没覆盖"误判成"看空"。

    Args:
        signals_by_strategy: 策略 ID 到信号列表的映射。
        weights: 策略 ID 到权重的映射。

    Returns:
        标的到融合得分的映射。
    """
    accumulated: dict[Symbol, list[tuple[float, float]]] = {}
    for strategy_id, signals in signals_by_strategy.items():
        weight = weights.get(strategy_id, 0.0)
        if weight <= 0:
            continue
        for signal in signals:
            accumulated.setdefault(signal.symbol, []).append((signal.score, weight))

    blended: dict[Symbol, float] = {}
    for symbol, pairs in accumulated.items():
        total_weight = sum(w for _, w in pairs)
        if total_weight <= 0:
            continue
        blended[symbol] = sum(s * w for s, w in pairs) / total_weight
    return blended
