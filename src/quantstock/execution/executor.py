"""执行编排。

规范见 docs/03-功能规格.md F8.1。

提交前的检查顺序是**刻意安排**的，从最不可绕过的开始：

1. **急停标志**（``var/HALT``）——放在第一行，任何代码路径都绕不过；
2. **计划有效期**——隔夜的计划不能照单执行；
3. **价格漂移复核**——T 日收盘生成、T+1 开盘执行，中间隔了一夜；
4. **绝对金额硬闸**——独立于比例风控的双保险，超限中止整个计划；
5. **人工逐单确认**——真实通道下不可关闭（红线 R5）；
6. **幂等校验**——同一 intent 不得重复提交。
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal

from quantstock.advisor.types import TradeIntent, TradePlan
from quantstock.execution.brokers import Broker, ManualBroker
from quantstock.execution.types import (
    BrokerOrder,
    DriftCheck,
    ExecutionReport,
    OrderBook,
    OrderStatus,
    PriceType,
    SkipReason,
)
from quantstock.infra.clock import now, today
from quantstock.infra.errors import ExecutionError
from quantstock.infra.logging import get_logger
from quantstock.infra.money import quantize_order_price, safe_div
from quantstock.infra.types import Money, Side, Symbol
from quantstock.risk.halt import HaltSwitch, HardLimitGuard

__all__ = ["ConfirmationDecision", "ExecutionRequest", "Executor"]

_log = get_logger(__name__)

DEFAULT_DRIFT_THRESHOLD = Decimal("0.03")


@dataclass(frozen=True, slots=True)
class ConfirmationDecision:
    """人工对单笔意图的决定。

    跳过时**必须给出原因**——复盘要按原因分组统计人工干预的价值
    （见 docs/08-差距分析与设计补强.md D3）。
    """

    intent_id: str
    accepted: bool
    adjusted_qty: int | None = None
    skip_reason: SkipReason | None = None
    skip_note: str = ""

    def __post_init__(self) -> None:
        """校验跳过必须带原因。

        Raises:
            ValueError: 跳过但未给原因。
        """
        if not self.accepted and self.skip_reason is None:
            msg = "跳过某条建议时必须选择原因，复盘要按原因统计人工干预的价值"
            raise ValueError(msg)


ConfirmFn = Callable[[TradeIntent, DriftCheck | None], ConfirmationDecision]
"""逐单确认回调：给定意图与漂移复核结果，返回人工决定。"""


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """一次执行请求。"""

    plan: TradePlan
    current_prices: dict[Symbol, Money]
    confirmed_by: str
    only_symbols: frozenset[Symbol] | None = None
    """只执行指定标的，对应 ``--only``。"""


class Executor:
    """执行编排器。"""

    def __init__(
        self,
        *,
        broker: Broker,
        halt_switch: HaltSwitch,
        hard_limits: HardLimitGuard,
        order_book: OrderBook | None = None,
        drift_threshold: Decimal = DEFAULT_DRIFT_THRESHOLD,
        plan_valid_days: int = 1,
    ) -> None:
        """初始化。

        Args:
            broker: 交易通道。
            halt_switch: 急停开关。
            hard_limits: 绝对金额硬闸。
            order_book: 订单簿，负责幂等。
            drift_threshold: 价格漂移阈值。
            plan_valid_days: 计划有效期（交易日）。
        """
        self._broker = broker
        self._halt = halt_switch
        self._hard_limits = hard_limits
        self._book = order_book or OrderBook()
        self._drift_threshold = drift_threshold
        self._plan_valid_days = plan_valid_days

    def execute(
        self, request: ExecutionRequest, *, confirm: ConfirmFn, live: bool = False
    ) -> ExecutionReport:
        """执行交易计划。

        Args:
            request: 执行请求。
            confirm: 逐单确认回调。
            live: 是否使用真实资金通道。真实通道必须显式开启（红线 R5）。

        Returns:
            执行报告。任一前置检查失败时 ``aborted=True`` 且不提交任何订单。

        Raises:
            ExecutionError: 真实通道未显式开启，或确认人为空。
        """
        # ① 急停：第一行，任何路径都绕不过（风控 A12）
        self._halt.ensure_not_halted()

        if self._broker.requires_live_flag and not live:
            msg = "该通道涉及真实资金，必须显式传入 live=True（红线 R5）"
            raise ExecutionError(msg, broker=self._broker.name)
        if not request.confirmed_by.strip():
            msg = "必须记录确认人（红线 R5）"
            raise ExecutionError(msg, plan_id=request.plan.plan_id)

        plan = request.plan

        # ② 计划有效期：隔夜的计划不能照单执行
        if (expired := self._check_expiry(plan)) is not None:
            return self._abort(plan, expired)

        intents = self._filter_intents(plan, request.only_symbols)

        # ③④⑤ 逐单：漂移复核 → 人工确认
        confirmed: list[BrokerOrder] = []
        skipped: list[BrokerOrder] = []
        for intent in intents:
            drift = self.check_drift(intent, request.current_prices)
            decision = confirm(intent, drift)
            order = self._to_order(intent, plan, decision)

            if not decision.accepted:
                skipped.append(
                    order.with_status(
                        OrderStatus.SKIPPED,
                        skip_reason=decision.skip_reason,
                        skip_note=decision.skip_note,
                    )
                )
                continue
            if self._book.is_duplicate(intent.intent_id):
                skipped.append(
                    order.with_status(
                        OrderStatus.SKIPPED,
                        skip_reason=SkipReason.OTHER,
                        skip_note="该意图已提交过，拒绝重复下单",
                    )
                )
                _log.warning("duplicate_intent_blocked", intent_id=str(intent.intent_id))
                continue
            confirmed.append(order.with_status(OrderStatus.CONFIRMED))

        # ⑥ 绝对金额硬闸：超限中止**整个**计划而非跳过单笔
        limit_result = self._hard_limits.check_orders(
            order_amounts=[o.amount for o in confirmed],
            order_quantities=[o.qty for o in confirmed],
        )
        if not limit_result.passed:
            reason = "；".join(f.message for f in limit_result.failures)
            return self._abort(plan, f"触发绝对金额硬闸，已中止整个计划：{reason}")

        if not confirmed:
            return ExecutionReport(
                plan_id=plan.plan_id,
                trade_date=plan.trade_date,
                executed_at=now(),
                broker=self._broker.name,
                orders=tuple(skipped),
                confirmed_by=request.confirmed_by,
            )

        submitted = self._broker.submit(confirmed)
        for order in submitted:
            self._book.mark_submitted(order.intent_id)

        fills = self._broker.fetch_fills([o.order_id for o in submitted])
        checklist = (
            ManualBroker.build_checklist(submitted)
            if isinstance(self._broker, ManualBroker)
            else ()
        )

        _log.info(
            "plan_executed",
            plan_id=plan.plan_id,
            broker=self._broker.name,
            submitted=len(submitted),
            skipped=len(skipped),
            confirmed_by=request.confirmed_by,
        )
        return ExecutionReport(
            plan_id=plan.plan_id,
            trade_date=plan.trade_date,
            executed_at=now(),
            broker=self._broker.name,
            orders=(*submitted, *skipped),
            fills=tuple(fills),
            confirmed_by=request.confirmed_by,
            manual_checklist=tuple(checklist),
        )

    def cancel_all(self) -> int:
        """撤销所有未成交委托。

        急停时应立即调用。

        Returns:
            撤单数量。
        """
        count = self._broker.cancel_all()
        _log.warning("cancel_all_issued", broker=self._broker.name, count=count)
        return count

    # ------------------------------------------------------------------ 内部
    def _check_expiry(self, plan: TradePlan) -> str | None:
        """校验计划是否过期。

        Args:
            plan: 交易计划。

        Returns:
            过期说明；未过期时 None。
        """
        elapsed = (today() - plan.trade_date).days
        if elapsed > self._plan_valid_days:
            return (
                f"计划生成于 {plan.trade_date}，已过期 {elapsed} 天"
                f"（有效期 {self._plan_valid_days} 天），请重新生成"
            )
        return None

    @staticmethod
    def _filter_intents(plan: TradePlan, only: frozenset[Symbol] | None) -> Sequence[TradeIntent]:
        """按 ``--only`` 过滤意图。

        Args:
            plan: 交易计划。
            only: 只执行的标的。

        Returns:
            过滤后的意图。
        """
        if only is None:
            return plan.intents
        return tuple(i for i in plan.intents if i.symbol in only)

    def check_drift(self, intent: TradeIntent, prices: dict[Symbol, Money]) -> DriftCheck | None:
        """价格漂移复核。

        Args:
            intent: 交易意图。
            prices: 当前实时价。

        Returns:
            复核结果；无实时价时返回 None（由确认回调决定是否继续）。
        """
        current = prices.get(intent.symbol)
        if current is None or current <= 0:
            return None
        reference = (intent.price_low + intent.price_high) / 2
        drift = safe_div(current - reference, reference)
        return DriftCheck(
            symbol=intent.symbol,
            reference_price=reference,
            current_price=current,
            drift=drift,
            is_stale=abs(drift) > self._drift_threshold,
            threshold=self._drift_threshold,
        )

    @staticmethod
    def _to_order(
        intent: TradeIntent, plan: TradePlan, decision: ConfirmationDecision
    ) -> BrokerOrder:
        """把意图转成券商订单。

        限价取区间的不利一侧：买入用上沿、卖出用下沿——
        提高成交概率，代价是略差的价格。宁可成交在略差的价位，
        也不要因为挂在有利一侧而整天不成交。

        Args:
            intent: 交易意图。
            plan: 所属计划。
            decision: 人工决定。

        Returns:
            券商订单（DRAFT 状态）。
        """
        is_buy = intent.side is Side.BUY
        # 再对齐到 0.01 报价单位：1577.478 这样的价格在券商 App 里输不进去，
        # 而手工通道就是照着这个价去下单的
        price = quantize_order_price(
            intent.price_high if is_buy else intent.price_low, aggressive=is_buy
        )
        qty = decision.adjusted_qty if decision.adjusted_qty is not None else intent.qty
        return BrokerOrder(
            order_id=uuid.uuid4().hex[:12],
            intent_id=intent.intent_id,
            plan_id=plan.plan_id,
            symbol=intent.symbol,
            side=intent.side,
            qty=max(qty, 0),
            price=price,
            price_type=PriceType.LIMIT,
            status=OrderStatus.DRAFT,
        )

    def _abort(self, plan: TradePlan, reason: str) -> ExecutionReport:
        """中止执行，不提交任何订单。

        Args:
            plan: 交易计划。
            reason: 中止原因。

        Returns:
            标记为中止的报告。
        """
        _log.critical("execution_aborted", plan_id=plan.plan_id, reason=reason)
        return ExecutionReport(
            plan_id=plan.plan_id,
            trade_date=plan.trade_date,
            executed_at=now(),
            broker=self._broker.name,
            orders=(),
            aborted=True,
            abort_reason=reason,
        )
