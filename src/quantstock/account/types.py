"""持仓账本的数据契约。

规范见 docs/11-持仓账本规格.md。

核心设计：**事件溯源**。交易流水不可变、只追加（红线 R8），
持仓与批次全部由流水重放得出，可随时完全重建。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import StrEnum

from quantstock.infra.money import ZERO
from quantstock.infra.types import AccountId, IntentId, Money, PlanId, Symbol, TradeDate

__all__ = [
    "CashFlow",
    "Lot",
    "LotConsumption",
    "Position",
    "Transaction",
    "TxnSource",
    "TxnType",
]


class TxnType(StrEnum):
    """流水类型。"""

    BUY = "buy"
    SELL = "sell"
    DIVIDEND_CASH = "dividend"
    """现金分红入账（税前）。"""
    DIVIDEND_TAX = "dividend_tax"
    """卖出时按持股期限补扣的红利税。"""
    SHARE_BONUS = "share_bonus"
    """送股/转增。按比例增加股数并等比下调成本，**建仓日不变**。"""
    RIGHTS_ISSUE = "rights_issue"
    """配股。产生新批次，建仓日为缴款日。"""
    SPLIT = "split"
    DEPOSIT = "deposit"
    WITHDRAW = "withdraw"
    IPO_ALLOT = "ipo_allot"
    """打新中签。"""
    DELIST = "delist"
    """退市清算。"""
    ADJUST = "adjust"
    """人工校正。流水不可修改，写错只能用反向 ADJUST 冲正，且必须填理由。"""


class TxnSource(StrEnum):
    """流水来源，用于追溯（红线 R6）。"""

    PLAN = "plan"
    """由交易计划执行产生，带 plan_id / intent_id。"""
    MANUAL = "manual"
    """人工在券商 App 下单后回填。"""
    BROKER_SYNC = "broker_sync"
    """从券商同步。"""
    CORPORATE_ACTION = "corporate_action"
    """公司行为自动生成。"""
    ADJUST = "adjust"
    """人工校正。"""


@dataclass(frozen=True, slots=True)
class Transaction:
    """交易流水。**不可变、只追加**——账本的唯一真相来源。

    ``qty`` 有符号：买入为正、卖出为负。``net_cash`` 为对现金的净影响，
    同样有符号（买入为负、卖出为正）。
    """

    txn_id: str
    account_id: AccountId
    txn_type: TxnType
    trade_date: TradeDate
    occurred_at: dt.datetime
    symbol: Symbol | None = None
    """资金类流水（入金/出金）为 None。"""
    qty: int = 0
    price: Money = ZERO
    amount: Money = ZERO
    commission: Money = ZERO
    stamp_tax: Money = ZERO
    transfer_fee: Money = ZERO
    exchange_fee: Money = ZERO
    regulatory_fee: Money = ZERO
    dividend_tax: Money = ZERO
    net_cash: Money = ZERO
    source: TxnSource = TxnSource.MANUAL
    plan_id: PlanId | None = None
    intent_id: IntentId | None = None
    order_id: str | None = None
    note: str = ""
    ratio: Decimal = ZERO
    """送股/拆股比例。SHARE_BONUS 时表示每股送转的股数（如 10 送 3 则为 0.3）。"""

    @property
    def total_fee(self) -> Money:
        """费用合计（含红利税）。"""
        return (
            self.commission
            + self.stamp_tax
            + self.transfer_fee
            + self.exchange_fee
            + self.regulatory_fee
            + self.dividend_tax
        )

    def requires_symbol(self) -> bool:
        """该类型流水是否必须带标的。"""
        return self.txn_type not in _CASH_ONLY_TYPES


_CASH_ONLY_TYPES = frozenset({TxnType.DEPOSIT, TxnType.WITHDRAW})


@dataclass(frozen=True, slots=True)
class Lot:
    """持仓批次。

    **批次级追踪是必须的**：红利税按持股期限分三档、持有期收益、
    "再持有 N 天可免税"提示，全都需要知道每一份股票是哪天买的。
    只存一个平均成本做不到这些（见 docs/11-持仓账本规格.md 第三节）。
    """

    lot_id: str
    symbol: Symbol
    open_date: TradeDate
    """建仓日。红利税与持有期的计算基准；送股不重置该日期。"""
    open_txn_id: str
    original_qty: int
    remaining_qty: int
    cost_price: Money
    """单位成本，含买入费用摊销。"""
    accrued_dividend: Money = ZERO
    """该批次已收到的现金分红（税前），卖出时作为红利税税基。"""

    @property
    def cost_total(self) -> Money:
        """剩余部分的总成本。"""
        return self.cost_price * self.remaining_qty

    @property
    def is_closed(self) -> bool:
        """该批次是否已全部卖出。"""
        return self.remaining_qty <= 0

    def holding_days(self, as_of: TradeDate) -> int:
        """持有自然日数，用于红利税分档。

        Args:
            as_of: 计算基准日。

        Returns:
            自建仓日起的自然日数。
        """
        return (as_of - self.open_date).days

    def with_qty(self, remaining_qty: int) -> Lot:
        """返回调整剩余数量后的新批次。

        Args:
            remaining_qty: 新的剩余数量。

        Returns:
            新的 Lot 实例（原实例不可变）。
        """
        return replace(self, remaining_qty=remaining_qty)


@dataclass(frozen=True, slots=True)
class LotConsumption:
    """一笔卖出消耗掉的批次明细。

    用于精确计算该笔交易的真实盈亏与红利税——不同批次的成本和持股期限不同。
    """

    lot_id: str
    qty: int
    open_date: TradeDate
    cost_price: Money
    holding_days: int
    dividend_base: Money
    """该部分对应的已收分红，作为红利税税基。"""
    dividend_tax: Money
    realized_pnl: Money
    """该部分的已实现盈亏（已扣除卖出费用分摊与红利税）。"""


@dataclass(frozen=True, slots=True)
class Position:
    """聚合持仓。由流水重放得出，不直接修改。"""

    account_id: AccountId
    symbol: Symbol
    qty: int
    available_qty: int
    """T+1 可卖量：昨仓减去已挂卖单冻结。"""
    frozen_qty: int
    cost_basis_avg: Money
    """加权平均成本（含费用）。用于止损计算。"""
    cost_basis_tax: Money
    """券商/税务口径成本。用于对账与红利税。"""
    first_open_date: TradeDate
    last_trade_date: TradeDate
    lots: tuple[Lot, ...] = ()
    realized_pnl: Money = ZERO
    total_dividend: Money = ZERO
    total_dividend_tax: Money = ZERO
    total_fee: Money = ZERO
    total_bought_qty: int = 0
    total_sold_qty: int = 0

    @property
    def is_empty(self) -> bool:
        """是否已清仓。"""
        return self.qty <= 0

    def holding_days(self, as_of: TradeDate) -> int:
        """自首次建仓的持有自然日数。

        Args:
            as_of: 计算基准日。

        Returns:
            自然日数。
        """
        return (as_of - self.first_open_date).days

    def market_value(self, price: Money) -> Money:
        """按给定价格计算市值。

        Args:
            price: 现价（不复权真实价）。

        Returns:
            持仓市值。
        """
        return price * self.qty

    def unrealized_pnl(self, price: Money) -> Money:
        """浮动盈亏。

        Args:
            price: 现价。

        Returns:
            浮动盈亏金额。
        """
        return (price - self.cost_basis_avg) * self.qty


@dataclass(frozen=True, slots=True)
class CashFlow:
    """资金流水条目。

    净值计算必须区分"策略赚的"与"入金带来的"——否则入金会被误算为收益
    （见 docs/08-差距分析与设计补强.md D2）。
    """

    date: TradeDate
    amount: Money
    """有符号：流入为正、流出为负。"""
    kind: TxnType
    note: str = ""


@dataclass(frozen=True, slots=True)
class LedgerState:
    """账本在某一时刻的完整状态。"""

    account_id: AccountId
    as_of: TradeDate
    cash: Money
    positions: dict[Symbol, Position] = field(default_factory=dict)
    realized_pnl: Money = ZERO
    total_fee: Money = ZERO
    total_dividend: Money = ZERO
    total_dividend_tax: Money = ZERO
    total_deposit: Money = ZERO
    total_withdraw: Money = ZERO

    def holdings_value(self, prices: dict[Symbol, Money]) -> Money:
        """持仓市值合计。

        Args:
            prices: 各标的现价。缺失价格的标的按 0 计并不会静默——
                调用方应先确保价格齐全。

        Returns:
            市值合计。
        """
        return sum(
            (pos.market_value(prices.get(sym, ZERO)) for sym, pos in self.positions.items()),
            start=ZERO,
        )

    def total_value(self, prices: dict[Symbol, Money]) -> Money:
        """总资产 = 现金 + 持仓市值。

        Args:
            prices: 各标的现价。

        Returns:
            总资产。
        """
        return self.cash + self.holdings_value(prices)
