"""目标组合构建与调仓差分。

规范见 docs/03-功能规格.md F4.3、F4.4。

三步：**打分 → 目标权重 → 与当前持仓差分**。

两条容易被忽略但很关键的设计：
- **缓冲带**：目标与当前偏离小于阈值时不调仓，否则每天都在为几百块钱的偏离付手续费。
- **整手向下取整**：宁可少买一手，也不能因为漏算费用导致资金不足被废单。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from quantstock.account.types import Position
from quantstock.infra.money import ZERO, align_lot, quantize_cny, safe_div
from quantstock.infra.types import Money, Side, Symbol

__all__ = [
    "PortfolioConstraints",
    "RebalanceOrder",
    "TargetPosition",
    "build_targets",
    "diff_to_orders",
]


@dataclass(frozen=True, slots=True)
class PortfolioConstraints:
    """组合约束。

    这些是 B 类规则（阈值可配、规则不可关闭），见 docs/05-风控规范.md 第二节。
    """

    max_single_position: Decimal = Decimal("0.15")
    max_industry_exposure: Decimal = Decimal("0.30")
    max_equity_exposure: Decimal = Decimal("0.90")
    max_holdings: int = 12
    min_position_value: Money = Decimal("5000")
    """单笔最小金额。低于此值时手续费占比过高（最低佣金 5 元），不如不做。"""
    rebalance_band: Decimal = Decimal("0.02")
    lot_size: int = 100


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """目标持仓。"""

    symbol: Symbol
    target_weight: Decimal
    target_qty: int
    reference_price: Money
    reason: str = ""

    @property
    def target_value(self) -> Money:
        """目标市值。"""
        return self.reference_price * self.target_qty


@dataclass(frozen=True, slots=True)
class RebalanceOrder:
    """调仓指令。"""

    symbol: Symbol
    side: Side
    qty: int
    reference_price: Money
    current_qty: int
    target_qty: int
    reason: str = ""

    @property
    def amount(self) -> Money:
        """预计成交金额。"""
        return quantize_cny(self.reference_price * self.qty)


@dataclass(frozen=True, slots=True)
class SkippedOrder:
    """被跳过的调仓，附原因。

    日报中必须显示这些——用户看到"为什么没动"和"为什么动了"同样重要。
    """

    symbol: Symbol
    reason: str


def build_targets(  # noqa: C901 - 权重约束是一串顺序施加的过滤，拆开会打乱施加次序
    *,
    scores: Mapping[Symbol, float],
    prices: Mapping[Symbol, Money],
    total_value: Money,
    exposure: Decimal,
    constraints: PortfolioConstraints,
    industries: Mapping[Symbol, str] | None = None,
    timing: Mapping[Symbol, Decimal] | None = None,
) -> list[TargetPosition]:
    """由打分构建目标持仓。

    权重分配用**波动率无关的打分加权**并逐一施加约束。
    个人账户持仓通常 ≤12 只，均值方差优化的收益远小于其估计误差风险
    （见 docs/08-差距分析与设计补强.md A4）。

    Args:
        scores: 标的打分，越高越看好。
        prices: 参考价（不复权真实价）。
        total_value: 账户总资产。
        exposure: LONG 层给出的权益仓位中枢 0~1。
        constraints: 组合约束。
        industries: 标的到行业的映射，用于行业集中度约束。
        timing: SHORT 层择时系数，**只能降低权重**。

    Returns:
        目标持仓列表，按权重降序。

    Raises:
        ValueError: 总资产非正或仓位越界。
    """
    if total_value <= 0:
        msg = f"总资产必须为正，收到 {total_value}"
        raise ValueError(msg)
    if not Decimal(0) <= exposure <= Decimal(1):
        msg = f"权益仓位必须在 [0, 1] 之间，收到 {exposure}"
        raise ValueError(msg)

    # 只保留有正打分且有价格的标的
    candidates = {
        sym: score for sym, score in scores.items() if score > 0 and prices.get(sym, ZERO) > 0
    }
    if not candidates:
        return []

    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    selected = ranked[: constraints.max_holdings]

    total_score = sum(score for _, score in selected)
    if total_score <= 0:
        return []

    effective_exposure = min(exposure, constraints.max_equity_exposure)

    industry_used: dict[str, Decimal] = {}
    targets: list[TargetPosition] = []

    for symbol, score in selected:
        weight = Decimal(str(score / total_score)) * effective_exposure
        weight = min(weight, constraints.max_single_position)

        # 择时系数在**单票上限之后**应用。顺序反过来的话，只要上限生效，
        # 择时就完全失效了——而上限恰恰在高打分标的上最常生效，
        # 那正是最需要择时保护的地方。系数只降不升（SHORT 层的不对称设计）。
        if timing is not None and (coef := timing.get(symbol)) is not None:
            weight *= min(coef, Decimal(1))

        # 行业集中度：超出部分直接截断而非按比例缩放，
        # 因为按比例缩放会让所有行业都贴着上限走，失去分散的意义
        industry = (industries or {}).get(symbol, "")
        if industry:
            used = industry_used.get(industry, ZERO)
            remaining = constraints.max_industry_exposure - used
            if remaining <= 0:
                continue
            weight = min(weight, remaining)
            industry_used[industry] = used + weight

        value = total_value * weight
        if value < constraints.min_position_value:
            continue

        price = prices[symbol]
        qty = align_lot(int(value / price), lot_size=constraints.lot_size)
        if qty <= 0:
            continue

        targets.append(
            TargetPosition(
                symbol=symbol,
                target_weight=weight.quantize(Decimal("0.0001")),
                target_qty=qty,
                reference_price=price,
                reason=f"打分 {score:.3f}，目标权重 {weight:.2%}",
            )
        )

    targets.sort(key=lambda t: t.target_weight, reverse=True)
    return targets


def diff_to_orders(  # noqa: C901, PLR0912 - 差分分支多但都是平铺的，每支对应一条可展示的跳过原因
    *,
    targets: list[TargetPosition],
    positions: Mapping[Symbol, Position],
    prices: Mapping[Symbol, Money],
    total_value: Money,
    constraints: PortfolioConstraints,
) -> tuple[list[RebalanceOrder], list[SkippedOrder]]:
    """目标持仓与当前持仓差分，得到调仓指令。

    Args:
        targets: 目标持仓。
        positions: 当前持仓。
        prices: 参考价。
        total_value: 账户总资产。
        constraints: 组合约束。

    Returns:
        ``(调仓指令, 被跳过的项)``。被跳过项必须在日报中展示。

    Raises:
        ValueError: 总资产非正。
    """
    if total_value <= 0:
        msg = f"总资产必须为正，收到 {total_value}"
        raise ValueError(msg)

    target_map = {t.symbol: t for t in targets}
    orders: list[RebalanceOrder] = []
    skipped: list[SkippedOrder] = []

    # 不在目标里的持仓全部清仓
    for symbol, position in positions.items():
        if symbol in target_map or position.qty <= 0:
            continue
        price = prices.get(symbol)
        if price is None or price <= 0:
            skipped.append(SkippedOrder(symbol, "无参考价，无法生成清仓指令"))
            continue
        sellable = position.available_qty
        if sellable <= 0:
            skipped.append(SkippedOrder(symbol, "T+1 未到可卖日，明日再清"))
            continue
        orders.append(
            RebalanceOrder(
                symbol=symbol,
                side=Side.SELL,
                qty=sellable,
                reference_price=price,
                current_qty=position.qty,
                target_qty=0,
                reason="已不在目标组合中，清仓",
            )
        )

    for target in targets:
        held = positions.get(target.symbol)
        current_qty = held.qty if held else 0
        delta = target.target_qty - current_qty
        if delta == 0:
            continue

        price = target.reference_price
        current_weight = safe_div(price * current_qty, total_value)
        drift = abs(target.target_weight - current_weight)

        # 缓冲带：偏离太小就不动，否则每天都在为几百块的偏离付手续费
        if drift < constraints.rebalance_band:
            skipped.append(
                SkippedOrder(
                    target.symbol,
                    f"权重偏离 {drift:.2%} 小于缓冲带 {constraints.rebalance_band:.2%}，不调仓",
                )
            )
            continue

        amount = quantize_cny(price * abs(delta))
        if amount < constraints.min_position_value:
            skipped.append(
                SkippedOrder(
                    target.symbol,
                    f"调仓金额 {amount} 低于最小值 {constraints.min_position_value}，"
                    "手续费占比过高",
                )
            )
            continue

        if delta > 0:
            qty = align_lot(delta, lot_size=constraints.lot_size)
            if qty <= 0:
                skipped.append(SkippedOrder(target.symbol, "增仓不足一手"))
                continue
            side = Side.BUY
        else:
            sellable = held.available_qty if held else 0
            qty = min(-delta, sellable)
            if qty <= 0:
                skipped.append(SkippedOrder(target.symbol, "T+1 未到可卖日，明日再减"))
                continue
            side = Side.SELL

        orders.append(
            RebalanceOrder(
                symbol=target.symbol,
                side=side,
                qty=qty,
                reference_price=price,
                current_qty=current_qty,
                target_qty=target.target_qty,
                reason=(f"当前权重 {current_weight:.2%} → 目标 {target.target_weight:.2%}"),
            )
        )

    # 卖出排在买入前面：先释放资金再用，避免"钱还没到账就想买"
    orders.sort(key=lambda o: (o.side is not Side.SELL, o.symbol))
    return orders, skipped
