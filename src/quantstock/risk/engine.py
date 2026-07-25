"""风控规则引擎与熔断状态机。

规范见 docs/05-风控规范.md。

三类规则：
- **A 类**（市场规则）：T+1、涨跌停、整手、停牌、急停。**不可关闭、不可调整**，
  因此它们不出现在配置里，界面上也不提供关闭入口。
- **B 类**（组合约束）：阈值可配，规则本身不可关闭。
- **C 类**（标的筛选）：可启停。

风控拒绝**不是异常流程**——``pre_trade_check`` 返回 ``RiskDecision``，
只有调用方显式要求时才抛异常（见开发规范第六条）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from quantstock.account.types import Position
from quantstock.config.models import CircuitBreakerConfig
from quantstock.data.types import Bar
from quantstock.infra.errors import RiskRejectedError
from quantstock.infra.money import ZERO, safe_div
from quantstock.infra.types import Money, Side, Symbol, TradeDate
from quantstock.portfolio.builder import PortfolioConstraints, RebalanceOrder

__all__ = [
    "CircuitState",
    "RiskDecision",
    "RiskEngine",
    "RuleOutcome",
    "Severity",
]


_MAX_DAILY_TURNOVER = Decimal("0.20")
"""单日换手上限（B09）。超出只告警不阻断——由 advisor 按打分优先级截断。"""


class Severity(StrEnum):
    """规则违反的处理方式。"""

    BLOCK = "block"
    """拒绝该笔。"""
    ADJUST = "adjust"
    """修正数量后放行——资金不足时缩量比直接丢弃更有用。"""
    WARN = "warn"
    """仅提示，不阻断。"""


class CircuitState(StrEnum):
    """组合级熔断状态。"""

    NORMAL = "normal"
    WATCH = "watch"
    """禁止新开仓，允许加仓已有持仓（上限收紧）。"""
    HALTED = "halted"
    """只出卖出/减仓建议，禁止任何买入。需人工执行 resume 才能恢复。"""


@dataclass(frozen=True, slots=True)
class RuleOutcome:
    """单条规则的判定结果。

    ``message`` **必须包含具体数值**——"仓位超限"没法让用户判断该怎么办，
    "单票占比将达 17.2% > 上限 15.0%"才可以。
    """

    rule_id: str
    rule_name: str
    passed: bool
    severity: Severity
    message: str
    adjusted_qty: int | None = None


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """风控整体判定。"""

    passed: bool
    circuit_state: CircuitState
    approved: tuple[RebalanceOrder, ...]
    rejected: tuple[tuple[RebalanceOrder, str], ...]
    per_order: dict[str, tuple[RuleOutcome, ...]] = field(default_factory=dict)
    portfolio_level: tuple[RuleOutcome, ...] = ()

    def raise_if_rejected(self) -> None:
        """存在被拒订单时抛异常。

        Raises:
            RiskRejectedError: 存在被拒订单。
        """
        if not self.rejected:
            return
        msg = "风控拒绝了部分交易意图"
        raise RiskRejectedError(
            msg,
            rejected=[f"{o.symbol}: {reason}" for o, reason in self.rejected],
            circuit_state=self.circuit_state.value,
        )


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """风控所需的市场快照。"""

    bars: Mapping[Symbol, Bar]
    industries: Mapping[Symbol, str] = field(default_factory=dict)
    avg_amount_20d: Mapping[Symbol, Money] = field(default_factory=dict)
    blacklist: frozenset[Symbol] = frozenset()
    """情报风险否决产生的黑名单（通路 1，见 docs/07 第六节）。"""


class RiskEngine:
    """风控规则引擎。"""

    def __init__(
        self,
        *,
        constraints: PortfolioConstraints | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
        min_avg_amount_20d: Money = Decimal("30000000"),
    ) -> None:
        """初始化。

        Args:
            constraints: 组合约束（B 类规则阈值）。
            circuit_config: 熔断阈值。
            min_avg_amount_20d: 流动性下限（B08）。
        """
        self._constraints = constraints or PortfolioConstraints()
        self._circuit = circuit_config or CircuitBreakerConfig()
        self._min_liquidity = min_avg_amount_20d

    # ------------------------------------------------------------------ 熔断
    def evaluate_circuit(
        self,
        *,
        daily_return: Decimal,
        drawdown_20d: Decimal,
        current: CircuitState = CircuitState.NORMAL,
    ) -> CircuitState:
        """评估熔断状态。

        **HALTED 不会自动恢复**——必须人工执行 ``risk resume``。
        自动恢复会让系统在剧烈波动中反复进出，而那正是最该停手的时候。

        Args:
            daily_return: 当日收益率（负值为亏损）。
            drawdown_20d: 20 日回撤（负值）。
            current: 当前状态。

        Returns:
            新状态。
        """
        cfg = self._circuit
        loss = -daily_return
        dd = -drawdown_20d

        if loss >= Decimal(str(cfg.halted_daily_loss)) or dd >= Decimal(
            str(cfg.halted_drawdown_20d)
        ):
            return CircuitState.HALTED
        if current is CircuitState.HALTED:
            return CircuitState.HALTED
        if loss >= Decimal(str(cfg.watch_daily_loss)) or dd >= Decimal(str(cfg.watch_drawdown_20d)):
            return CircuitState.WATCH
        if current is CircuitState.WATCH and dd <= Decimal(str(cfg.recover_drawdown)):
            return CircuitState.NORMAL
        return current

    # ------------------------------------------------------------------ 事前检查
    def pre_trade_check(
        self,
        *,
        orders: Sequence[RebalanceOrder],
        positions: Mapping[Symbol, Position],
        cash: Money,
        total_value: Money,
        market: MarketSnapshot,
        trade_date: TradeDate,
        circuit_state: CircuitState = CircuitState.NORMAL,
    ) -> RiskDecision:
        """事前风控检查。

        Args:
            orders: 待检查的调仓指令。
            positions: 当前持仓。
            cash: 可用资金。
            total_value: 账户总资产。
            market: 市场快照。
            trade_date: 交易日。
            circuit_state: 当前熔断状态。

        Returns:
            风控判定。被拒的每笔都带具体原因与数值。
        """
        approved: list[RebalanceOrder] = []
        rejected: list[tuple[RebalanceOrder, str]] = []
        per_order: dict[str, tuple[RuleOutcome, ...]] = {}

        remaining_cash = cash
        industry_exposure = self._current_industry_exposure(positions, market, total_value)

        for order in orders:
            outcomes = self._check_order(
                order=order,
                positions=positions,
                cash=remaining_cash,
                total_value=total_value,
                market=market,
                trade_date=trade_date,
                circuit_state=circuit_state,
                industry_exposure=industry_exposure,
            )
            per_order[str(order.symbol)] = tuple(outcomes)

            blocking = [o for o in outcomes if not o.passed and o.severity is Severity.BLOCK]
            if blocking:
                rejected.append((order, "；".join(o.message for o in blocking)))
                continue

            adjusted = next((o.adjusted_qty for o in outcomes if o.adjusted_qty is not None), None)
            final = order if adjusted is None else _with_qty(order, adjusted)
            if final.qty <= 0:
                rejected.append((order, "修正后数量为 0"))
                continue

            approved.append(final)
            if final.side is Side.BUY:
                remaining_cash -= final.amount
                industry = market.industries.get(final.symbol, "")
                if industry:
                    industry_exposure[industry] = industry_exposure.get(industry, ZERO) + safe_div(
                        final.amount, total_value
                    )

        return RiskDecision(
            passed=not rejected,
            circuit_state=circuit_state,
            approved=tuple(approved),
            rejected=tuple(rejected),
            per_order=per_order,
            portfolio_level=self._portfolio_checks(approved, total_value),
        )

    def _check_order(  # noqa: C901, PLR0912 - 规则是一串平铺的判定，拆开会让"检查了哪些"变得不明显
        self,
        *,
        order: RebalanceOrder,
        positions: Mapping[Symbol, Position],
        cash: Money,
        total_value: Money,
        market: MarketSnapshot,
        trade_date: TradeDate,
        circuit_state: CircuitState,
        industry_exposure: Mapping[str, Decimal],
    ) -> list[RuleOutcome]:
        """对单笔指令逐条应用规则。

        Args:
            order: 调仓指令。
            positions: 当前持仓。
            cash: 剩余可用资金。
            total_value: 账户总资产。
            market: 市场快照。
            trade_date: 交易日。
            circuit_state: 熔断状态。
            industry_exposure: 各行业当前暴露。

        Returns:
            规则判定列表。
        """
        outcomes: list[RuleOutcome] = []
        bar = market.bars.get(order.symbol)
        is_buy = order.side is Side.BUY

        # ---- A 类：市场规则，不可关闭 ----
        if bar is None:
            outcomes.append(
                _fail("A04", "行情可得性", Severity.BLOCK, f"{order.symbol} 无当日行情，无法交易")
            )
            return outcomes

        if bar.is_suspended:
            outcomes.append(_fail("A04", "停牌", Severity.BLOCK, f"{order.symbol} 停牌"))
        if is_buy and bar.is_limit_up:
            outcomes.append(
                _fail("A02", "涨跌停", Severity.BLOCK, f"{order.symbol} 涨停，无法买入")
            )
        if not is_buy and bar.is_limit_down:
            outcomes.append(
                _fail("A02", "涨跌停", Severity.BLOCK, f"{order.symbol} 跌停，无法卖出")
            )

        if is_buy and order.qty % self._constraints.lot_size != 0:
            outcomes.append(
                _fail(
                    "A03",
                    "整手买入",
                    Severity.BLOCK,
                    f"买入数量 {order.qty} 不是 {self._constraints.lot_size} 的整数倍",
                )
            )

        if not is_buy:
            position = positions.get(order.symbol)
            available = position.available_qty if position else 0
            if order.qty > available:
                outcomes.append(
                    RuleOutcome(
                        rule_id="A01",
                        rule_name="T+1 可卖量",
                        passed=False,
                        severity=Severity.ADJUST,
                        message=f"可卖量 {available} 少于委托 {order.qty}，缩量至可卖量",
                        adjusted_qty=available,
                    )
                )

        if is_buy and order.amount > cash:
            outcomes.append(
                RuleOutcome(
                    rule_id="A07",
                    rule_name="资金充足性",
                    passed=False,
                    severity=Severity.ADJUST,
                    message=f"所需 {order.amount} 超过可用资金 {cash}，按可用资金缩量",
                    adjusted_qty=_affordable_qty(cash, order, self._constraints.lot_size),
                )
            )

        # ---- 熔断状态：HALTED 下禁止一切买入 ----
        if is_buy and circuit_state is CircuitState.HALTED:
            outcomes.append(
                _fail(
                    "CB",
                    "熔断",
                    Severity.BLOCK,
                    "系统处于 HALTED 状态，禁止任何买入，需人工复核后 resume",
                )
            )
        if is_buy and circuit_state is CircuitState.WATCH and order.current_qty == 0:
            outcomes.append(_fail("CB", "熔断", Severity.BLOCK, "WATCH 状态下禁止新开仓"))

        # ---- 情报黑名单（通路 1，单向否决）----
        if is_buy and order.symbol in market.blacklist:
            outcomes.append(
                _fail(
                    "I-R2",
                    "情报风险否决",
                    Severity.BLOCK,
                    f"{order.symbol} 处于情报黑名单，禁止买入",
                )
            )

        # ---- B 类：组合约束 ----
        if is_buy:
            position = positions.get(order.symbol)
            current_value = bar.close * (position.qty if position else 0)
            new_weight = safe_div(current_value + order.amount, total_value)
            if new_weight > self._constraints.max_single_position:
                outcomes.append(
                    _fail(
                        "B01",
                        "单票仓位上限",
                        Severity.BLOCK,
                        f"{order.symbol} 买入后占比将达 {new_weight:.1%} > "
                        f"上限 {self._constraints.max_single_position:.1%}",
                    )
                )

            industry = market.industries.get(order.symbol, "")
            if industry:
                after = industry_exposure.get(industry, ZERO) + safe_div(order.amount, total_value)
                if after > self._constraints.max_industry_exposure:
                    outcomes.append(
                        _fail(
                            "B02",
                            "行业集中度上限",
                            Severity.BLOCK,
                            f"行业「{industry}」买入后占比将达 {after:.1%} > "
                            f"上限 {self._constraints.max_industry_exposure:.1%}",
                        )
                    )

            liquidity = market.avg_amount_20d.get(order.symbol)
            if liquidity is not None and liquidity < self._min_liquidity:
                outcomes.append(
                    _fail(
                        "B08",
                        "流动性下限",
                        Severity.BLOCK,
                        f"{order.symbol} 近 20 日均额 {liquidity} 低于下限 "
                        f"{self._min_liquidity}，买入后可能卖不掉",
                    )
                )

            if order.amount < self._constraints.min_position_value:
                outcomes.append(
                    _fail(
                        "B06",
                        "单笔最小金额",
                        Severity.BLOCK,
                        f"金额 {order.amount} 低于下限 "
                        f"{self._constraints.min_position_value}，手续费占比过高",
                    )
                )

        _ = trade_date
        return outcomes

    def _portfolio_checks(
        self, approved: Sequence[RebalanceOrder], total_value: Money
    ) -> tuple[RuleOutcome, ...]:
        """组合层面的检查。

        Args:
            approved: 已通过的指令。
            total_value: 账户总资产。

        Returns:
            组合级判定。
        """
        turnover = safe_div(sum((o.amount for o in approved), start=ZERO), total_value)
        return (
            RuleOutcome(
                rule_id="B09",
                rule_name="单日换手上限",
                passed=turnover <= _MAX_DAILY_TURNOVER,
                severity=Severity.WARN,
                message=f"本次换手 {turnover:.1%}（上限 {_MAX_DAILY_TURNOVER:.0%}）",
            ),
        )

    @staticmethod
    def _current_industry_exposure(
        positions: Mapping[Symbol, Position],
        market: MarketSnapshot,
        total_value: Money,
    ) -> dict[str, Decimal]:
        """计算各行业当前暴露。

        Args:
            positions: 当前持仓。
            market: 市场快照。
            total_value: 账户总资产。

        Returns:
            行业到暴露占比的映射。
        """
        exposure: dict[str, Decimal] = {}
        for symbol, position in positions.items():
            industry = market.industries.get(symbol, "")
            bar = market.bars.get(symbol)
            if not industry or bar is None:
                continue
            weight = safe_div(bar.close * position.qty, total_value)
            exposure[industry] = exposure.get(industry, ZERO) + weight
        return exposure


def _fail(rule_id: str, name: str, severity: Severity, message: str) -> RuleOutcome:
    """构造一条未通过的判定。

    Args:
        rule_id: 规则编号。
        name: 规则名。
        severity: 严重度。
        message: 含具体数值的说明。

    Returns:
        判定结果。
    """
    return RuleOutcome(
        rule_id=rule_id, rule_name=name, passed=False, severity=severity, message=message
    )


def _with_qty(order: RebalanceOrder, qty: int) -> RebalanceOrder:
    """返回修正数量后的新指令。

    Args:
        order: 原指令。
        qty: 新数量。

    Returns:
        新指令。
    """
    return RebalanceOrder(
        symbol=order.symbol,
        side=order.side,
        qty=qty,
        reference_price=order.reference_price,
        current_qty=order.current_qty,
        target_qty=order.target_qty,
        reason=f"{order.reason}（风控缩量至 {qty}）",
    )


def _affordable_qty(cash: Money, order: RebalanceOrder, lot_size: int) -> int:
    """按可用资金算出能买多少（整手，未含费用）。

    这里刻意不含费用——精确的可负担数量由 ``CostModel.max_affordable_qty`` 计算，
    风控层只做粗粒度缩量，最终由执行层再校验一次。

    Args:
        cash: 可用资金。
        order: 原指令。
        lot_size: 每手股数。

    Returns:
        可买数量。
    """
    if order.reference_price <= 0:
        return 0
    lots = int(cash / (order.reference_price * lot_size))
    return max(lots, 0) * lot_size
