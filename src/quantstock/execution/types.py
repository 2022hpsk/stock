"""执行层的数据契约。

规范见 docs/03-功能规格.md F8.3。

订单状态机::

    DRAFT → CONFIRMED → SUBMITTED → PARTIAL → FILLED
                                  ↘ CANCELLED / REJECTED

**幂等是硬要求**：同一 ``intent_id`` 重复提交必须被拒绝。
重复下单在真实资金上的代价是不可逆的。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import StrEnum

from quantstock.infra.money import ZERO
from quantstock.infra.types import IntentId, Money, PlanId, Side, Symbol, TradeDate

__all__ = [
    "BrokerOrder",
    "DriftCheck",
    "ExecutionReport",
    "OrderBook",
    "OrderStatus",
    "PriceType",
    "SkipReason",
    "TradeFill",
    "can_transition",
]


class OrderStatus(StrEnum):
    """订单状态。"""

    DRAFT = "draft"
    """由计划生成，尚未确认。"""
    CONFIRMED = "confirmed"
    """已人工确认，等待提交（红线 R5）。"""
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    """人工跳过，附原因。"""

    @property
    def is_terminal(self) -> bool:
        """是否为终态。终态订单不再变化。"""
        return self in _TERMINAL_STATUSES

    @property
    def is_live(self) -> bool:
        """是否仍在券商处挂单。"""
        return self in {OrderStatus.SUBMITTED, OrderStatus.PARTIAL}


_TERMINAL_STATUSES = frozenset(
    {
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.REJECTED,
        OrderStatus.SKIPPED,
    }
)

_ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.DRAFT: frozenset({OrderStatus.CONFIRMED, OrderStatus.SKIPPED}),
    OrderStatus.CONFIRMED: frozenset({OrderStatus.SUBMITTED, OrderStatus.REJECTED}),
    OrderStatus.SUBMITTED: frozenset(
        {
            OrderStatus.PARTIAL,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.PARTIAL: frozenset({OrderStatus.FILLED, OrderStatus.CANCELLED}),
}


def can_transition(source: OrderStatus, target: OrderStatus) -> bool:
    """判断状态迁移是否合法。

    非法迁移通常意味着回报乱序或逻辑出错——例如已成交的订单又收到"已撤单"。
    显式校验能把这类问题在写入账本之前拦下。

    Args:
        source: 当前状态。
        target: 目标状态。

    Returns:
        合法则 True。
    """
    return target in _ALLOWED_TRANSITIONS.get(source, frozenset())


class PriceType(StrEnum):
    """委托价格类型。"""

    LIMIT = "limit"
    MARKET = "market"
    """A 股用"最优五档即时成交剩余撤销"实现。"""


class SkipReason(StrEnum):
    """人工跳过的原因。

    **必须让用户从枚举里选**而非自由输入——复盘时要按原因分组统计
    人工干预的胜率：若"不认同逻辑"的跳过长期跑赢程序，说明策略有系统性缺陷；
    长期跑输则说明该更信任程序（见 docs/08-差距分析与设计补强.md D3）。
    """

    DISAGREE_LOGIC = "disagree_logic"
    """不认同策略逻辑。"""
    CASH_RESERVED = "cash_reserved"
    """资金另有安排。"""
    BAD_TIMING = "bad_timing"
    """认为时机不对。"""
    OTHER_INFO = "other_info"
    """已有其他渠道信息。"""
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class TradeFill:
    """一笔成交回报。"""

    fill_id: str
    order_id: str
    symbol: Symbol
    side: Side
    qty: int
    price: Money
    filled_at: dt.datetime
    fee: Money = ZERO

    @property
    def amount(self) -> Money:
        """成交金额。"""
        return self.price * self.qty


@dataclass(frozen=True, slots=True)
class BrokerOrder:
    """提交给券商的订单。

    ``intent_id`` 贯穿建议 → 计划 → 订单 → 成交，是幂等与追溯的锚点（红线 R6）。
    """

    order_id: str
    intent_id: IntentId
    plan_id: PlanId
    symbol: Symbol
    side: Side
    qty: int
    price: Money
    price_type: PriceType = PriceType.LIMIT
    status: OrderStatus = OrderStatus.DRAFT
    filled_qty: int = 0
    avg_fill_price: Money = ZERO
    submitted_at: dt.datetime | None = None
    broker_order_id: str = ""
    message: str = ""
    skip_reason: SkipReason | None = None
    skip_note: str = ""

    @property
    def remaining_qty(self) -> int:
        """未成交数量。"""
        return max(self.qty - self.filled_qty, 0)

    @property
    def amount(self) -> Money:
        """委托金额。"""
        return self.price * self.qty

    def with_status(self, status: OrderStatus, **changes: object) -> BrokerOrder:
        """返回状态迁移后的新订单。

        Args:
            status: 目标状态。
            **changes: 同时变更的其它字段。

        Returns:
            新订单实例。

        Raises:
            ValueError: 状态迁移非法。
        """
        if status is not self.status and not can_transition(self.status, status):
            msg = f"非法的订单状态迁移：{self.status} → {status}"
            raise ValueError(msg)
        return replace(self, status=status, **changes)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """一次执行的完整报告。"""

    plan_id: PlanId
    trade_date: TradeDate
    executed_at: dt.datetime
    broker: str
    orders: tuple[BrokerOrder, ...]
    fills: tuple[TradeFill, ...] = ()
    aborted: bool = False
    abort_reason: str = ""
    confirmed_by: str = ""
    manual_checklist: tuple[str, ...] = ()
    """手工执行清单，ManualBroker 输出。"""

    @property
    def submitted(self) -> tuple[BrokerOrder, ...]:
        """已提交的订单。"""
        return tuple(o for o in self.orders if o.status is not OrderStatus.SKIPPED)

    @property
    def skipped(self) -> tuple[BrokerOrder, ...]:
        """被跳过的订单。"""
        return tuple(o for o in self.orders if o.status is OrderStatus.SKIPPED)

    @property
    def total_amount(self) -> Money:
        """已提交订单的金额合计。"""
        return sum((o.amount for o in self.submitted), start=Decimal(0))

    def skip_reasons(self) -> dict[str, int]:
        """按原因统计跳过次数，供复盘的人工干预价值分析。

        Returns:
            原因到次数的映射。
        """
        counts: dict[str, int] = {}
        for order in self.skipped:
            key = order.skip_reason.value if order.skip_reason else "unspecified"
            counts[key] = counts.get(key, 0) + 1
        return counts


@dataclass(frozen=True, slots=True)
class DriftCheck:
    """价格漂移复核结果。

    T 日收盘后生成的计划在 T+1 开盘执行，中间隔了一夜。
    开盘价偏离建议区间过多时必须停下来让人重新判断，而不是照单执行。
    """

    symbol: Symbol
    reference_price: Money
    current_price: Money
    drift: Decimal
    is_stale: bool
    threshold: Decimal

    @property
    def message(self) -> str:
        """人类可读的说明。"""
        verdict = "超出阈值，需二次确认" if self.is_stale else "在可接受范围内"
        return (
            f"{self.symbol} 现价 {self.current_price} 相对建议参考价 "
            f"{self.reference_price} 漂移 {self.drift:+.2%}（阈值 "
            f"{self.threshold:.1%}），{verdict}"
        )


@dataclass(frozen=True, slots=True)
class OrderBook:
    """订单簿，负责幂等。

    同一 ``intent_id`` 只允许提交一次——重复下单在真实资金上的代价不可逆。
    """

    submitted_intents: set[str] = field(default_factory=set)

    def is_duplicate(self, intent_id: IntentId) -> bool:
        """该意图是否已提交过。

        Args:
            intent_id: 意图标识。

        Returns:
            已提交过则 True。
        """
        return str(intent_id) in self.submitted_intents

    def mark_submitted(self, intent_id: IntentId) -> None:
        """标记该意图已提交。

        Args:
            intent_id: 意图标识。
        """
        self.submitted_intents.add(str(intent_id))
