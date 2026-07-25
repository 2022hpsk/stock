"""账本重放引擎。

规范见 docs/11-持仓账本规格.md。

**所有持仓状态都由不可变流水推导得出**（红线 R8）。这样做的价值：

- 任意历史时点的持仓可精确还原，支撑回测、审计与对账；
- 成本口径或税规则变更时，只需改重放逻辑并重算，原始记录不受影响；
- 与券商对账时能精确定位到是哪一笔流水对不上。
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from decimal import Decimal

from quantstock.account.types import (
    LedgerState,
    Lot,
    LotConsumption,
    Position,
    Transaction,
    TxnType,
)
from quantstock.costs import dividend_tax_rate
from quantstock.infra.errors import LedgerError
from quantstock.infra.money import ZERO, quantize_cny, quantize_price, safe_div
from quantstock.infra.types import AccountId, Money, Symbol, TradeDate

__all__ = ["Ledger", "replay"]


class _SymbolBook:
    """单个标的的批次账，内部可变；对外只暴露不可变快照。"""

    def __init__(self, symbol: Symbol) -> None:
        self.symbol = symbol
        self.lots: list[Lot] = []
        self.realized_pnl: Money = ZERO
        self.total_dividend: Money = ZERO
        self.total_dividend_tax: Money = ZERO
        self.total_fee: Money = ZERO
        self.total_bought_qty = 0
        self.total_sold_qty = 0
        self.first_open_date: TradeDate | None = None
        self.last_trade_date: TradeDate | None = None
        self.cost_basis_tax: Money = ZERO
        """券商口径成本：累计买入净支出 / 累计持股数。"""
        self.tax_cost_total: Money = ZERO

    @property
    def qty(self) -> int:
        """当前持仓总数。"""
        return sum(lot.remaining_qty for lot in self.lots)

    def cost_total(self) -> Money:
        """当前持仓总成本（加权平均口径）。"""
        return sum((lot.cost_total for lot in self.lots), start=ZERO)


class Ledger:
    """账本。

    通过重放不可变流水得到持仓状态。构造后调用 :meth:`apply` 逐笔重放，
    或直接用模块级函数 :func:`replay` 一次性重放整个序列。
    """

    def __init__(self, account_id: AccountId) -> None:
        """初始化空账本。

        Args:
            account_id: 账户标识。
        """
        self.account_id = account_id
        self._books: dict[Symbol, _SymbolBook] = {}
        self._cash: Money = ZERO
        self._total_deposit: Money = ZERO
        self._total_withdraw: Money = ZERO
        self._seen_txn_ids: set[str] = set()
        self._last_date: TradeDate | None = None
        self._lot_counter = itertools.count(1)
        self.consumptions: list[LotConsumption] = []
        """所有卖出消耗的批次明细，供绩效归因与税务核对。"""

    # ------------------------------------------------------------------ 重放
    def apply(self, txn: Transaction) -> None:
        """应用一笔流水。

        Args:
            txn: 待应用的流水。

        Raises:
            LedgerError: 流水非法——账户不匹配、重复 txn_id、时间倒流、
                缺少必需字段、卖出超过持仓等。
        """
        self._validate(txn)
        self._seen_txn_ids.add(txn.txn_id)
        self._last_date = txn.trade_date

        match txn.txn_type:
            case TxnType.BUY | TxnType.IPO_ALLOT | TxnType.RIGHTS_ISSUE:
                self._apply_buy(txn)
            case TxnType.SELL:
                self._apply_sell(txn)
            case TxnType.DIVIDEND_CASH:
                self._apply_dividend(txn)
            case TxnType.SHARE_BONUS | TxnType.SPLIT:
                self._apply_share_bonus(txn)
            case TxnType.DELIST:
                self._apply_delist(txn)
            case TxnType.ADJUST:
                self._apply_adjust(txn)
            case TxnType.DEPOSIT | TxnType.WITHDRAW:
                self._apply_cash_only(txn)
            case TxnType.DIVIDEND_TAX:
                pass  # 只影响现金，由下方统一处理

        self._cash += txn.net_cash

    def _validate(self, txn: Transaction) -> None:
        """校验流水的基本合法性。

        Args:
            txn: 待校验流水。

        Raises:
            LedgerError: 校验不通过。
        """
        if txn.account_id != self.account_id:
            msg = "流水账户与账本不匹配"
            raise LedgerError(
                msg, txn_id=txn.txn_id, expected=self.account_id, actual=txn.account_id
            )
        if txn.txn_id in self._seen_txn_ids:
            # 幂等：重复导入同一笔成交必须被拒绝，防止重复记账
            msg = "流水重复"
            raise LedgerError(msg, txn_id=txn.txn_id)
        if self._last_date is not None and txn.trade_date < self._last_date:
            msg = "流水必须按时间升序重放"
            raise LedgerError(
                msg, txn_id=txn.txn_id, trade_date=txn.trade_date, last_date=self._last_date
            )
        if txn.requires_symbol() and txn.symbol is None:
            msg = f"{txn.txn_type} 类型的流水必须带 symbol"
            raise LedgerError(msg, txn_id=txn.txn_id)
        if txn.txn_type is TxnType.ADJUST and not txn.note:
            # 人工校正必须留下理由，否则账本无法审计
            msg = "ADJUST 类型的流水必须填写 note 说明校正原因"
            raise LedgerError(msg, txn_id=txn.txn_id)

    def _book(self, symbol: Symbol) -> _SymbolBook:
        """取或创建某标的的批次账。

        Args:
            symbol: 标的。

        Returns:
            该标的的批次账。
        """
        book = self._books.get(symbol)
        if book is None:
            book = _SymbolBook(symbol)
            self._books[symbol] = book
        return book

    # ------------------------------------------------------------------ 各类型处理
    def _apply_buy(self, txn: Transaction) -> None:
        """买入：新建一个批次。成本含买入费用摊销。"""
        if txn.qty <= 0:
            msg = "买入数量必须为正"
            raise LedgerError(msg, txn_id=txn.txn_id, qty=txn.qty)
        assert txn.symbol is not None  # noqa: S101 - 已由 _validate 保证

        book = self._book(txn.symbol)
        buy_fees = (
            txn.commission
            + txn.stamp_tax
            + txn.transfer_fee
            + txn.exchange_fee
            + txn.regulatory_fee
        )
        cost_total = txn.amount + buy_fees
        cost_price = quantize_price(cost_total / txn.qty)

        book.lots.append(
            Lot(
                lot_id=f"{txn.symbol}-{next(self._lot_counter):06d}",
                symbol=txn.symbol,
                open_date=txn.trade_date,
                open_txn_id=txn.txn_id,
                original_qty=txn.qty,
                remaining_qty=txn.qty,
                cost_price=cost_price,
            )
        )
        book.total_fee += buy_fees
        book.total_bought_qty += txn.qty
        book.tax_cost_total += cost_total
        if book.first_open_date is None:
            book.first_open_date = txn.trade_date
        book.last_trade_date = txn.trade_date
        self._refresh_tax_basis(book)

    def _apply_sell(self, txn: Transaction) -> None:
        """卖出：按 FIFO 消耗批次，逐批次计算红利税与已实现盈亏。"""
        sell_qty = -txn.qty if txn.qty < 0 else txn.qty
        if sell_qty <= 0:
            msg = "卖出数量必须非零"
            raise LedgerError(msg, txn_id=txn.txn_id, qty=txn.qty)
        assert txn.symbol is not None  # noqa: S101 - 已由 _validate 保证

        book = self._book(txn.symbol)
        if book.qty < sell_qty:
            msg = "卖出数量超过持仓"
            raise LedgerError(
                msg, txn_id=txn.txn_id, symbol=txn.symbol, held=book.qty, selling=sell_qty
            )

        sell_fees = (
            txn.commission
            + txn.stamp_tax
            + txn.transfer_fee
            + txn.exchange_fee
            + txn.regulatory_fee
        )
        unit_price = safe_div(txn.amount, Decimal(sell_qty))

        remaining = sell_qty
        total_tax: Money = ZERO
        total_pnl: Money = ZERO

        # FIFO：先进先出，与券商红利税计算口径一致
        for idx, lot in enumerate(book.lots):
            if remaining <= 0:
                break
            if lot.remaining_qty <= 0:
                continue

            take = min(lot.remaining_qty, remaining)
            holding_days = lot.holding_days(txn.trade_date)

            # 红利税：按该批次持股期限分档，税基为该部分对应的已收分红
            dividend_base = quantize_cny(
                safe_div(lot.accrued_dividend * take, Decimal(lot.remaining_qty))
            )
            tax = quantize_cny(dividend_base * dividend_tax_rate(holding_days))

            gross = unit_price * take
            cost = lot.cost_price * take
            fee_share = quantize_cny(safe_div(sell_fees * take, Decimal(sell_qty)))
            pnl = quantize_cny(gross - cost - fee_share - tax)

            self.consumptions.append(
                LotConsumption(
                    lot_id=lot.lot_id,
                    qty=take,
                    open_date=lot.open_date,
                    cost_price=lot.cost_price,
                    holding_days=holding_days,
                    dividend_base=dividend_base,
                    dividend_tax=tax,
                    realized_pnl=pnl,
                )
            )

            book.lots[idx] = Lot(
                lot_id=lot.lot_id,
                symbol=lot.symbol,
                open_date=lot.open_date,
                open_txn_id=lot.open_txn_id,
                original_qty=lot.original_qty,
                remaining_qty=lot.remaining_qty - take,
                cost_price=lot.cost_price,
                accrued_dividend=quantize_cny(lot.accrued_dividend - dividend_base),
            )
            book.tax_cost_total -= cost
            total_tax += tax
            total_pnl += pnl
            remaining -= take

        book.lots = [lot for lot in book.lots if lot.remaining_qty > 0]
        book.realized_pnl += total_pnl
        book.total_dividend_tax += total_tax
        book.total_fee += sell_fees + total_tax
        book.total_sold_qty += sell_qty
        book.last_trade_date = txn.trade_date
        self._refresh_tax_basis(book)

    def _apply_dividend(self, txn: Transaction) -> None:
        """现金分红入账（税前），按持股比例分摊到各批次。

        红利税不在此时扣除——由券商在**卖出时**按持股期限补扣。
        """
        assert txn.symbol is not None  # noqa: S101 - 已由 _validate 保证
        book = self._book(txn.symbol)
        held = book.qty
        if held <= 0:
            msg = "分红时无持仓"
            raise LedgerError(msg, txn_id=txn.txn_id, symbol=txn.symbol)

        per_share = safe_div(txn.amount, Decimal(held))
        book.lots = [
            Lot(
                lot_id=lot.lot_id,
                symbol=lot.symbol,
                open_date=lot.open_date,
                open_txn_id=lot.open_txn_id,
                original_qty=lot.original_qty,
                remaining_qty=lot.remaining_qty,
                cost_price=lot.cost_price,
                accrued_dividend=quantize_cny(lot.accrued_dividend + per_share * lot.remaining_qty),
            )
            for lot in book.lots
        ]
        book.total_dividend += txn.amount

    def _apply_share_bonus(self, txn: Transaction) -> None:
        """送股/转增：按比例增加各批次股数并等比下调成本。

        **建仓日保持不变**——送股不重置持股期限，红利税档位不变。
        """
        assert txn.symbol is not None  # noqa: S101 - 已由 _validate 保证
        if txn.ratio <= 0:
            msg = "送股比例必须为正"
            raise LedgerError(msg, txn_id=txn.txn_id, ratio=txn.ratio)

        book = self._book(txn.symbol)
        new_lots: list[Lot] = []
        for lot in book.lots:
            bonus_qty = int(lot.remaining_qty * txn.ratio)
            new_qty = lot.remaining_qty + bonus_qty
            if new_qty <= 0:
                continue
            new_lots.append(
                Lot(
                    lot_id=lot.lot_id,
                    symbol=lot.symbol,
                    open_date=lot.open_date,  # 关键：不重置
                    open_txn_id=lot.open_txn_id,
                    original_qty=lot.original_qty,
                    remaining_qty=new_qty,
                    cost_price=quantize_price(lot.cost_price * lot.remaining_qty / new_qty),
                    accrued_dividend=lot.accrued_dividend,
                )
            )
        book.lots = new_lots
        book.last_trade_date = txn.trade_date
        self._refresh_tax_basis(book)

    def _apply_cash_only(self, txn: Transaction) -> None:
        """入金/出金：只影响现金，不涉及持仓。"""
        if txn.txn_type is TxnType.DEPOSIT:
            self._total_deposit += txn.net_cash
        else:
            self._total_withdraw += -txn.net_cash

    def _apply_adjust(self, txn: Transaction) -> None:
        """人工校正：只调整现金与（可选的）持仓数量，必须带理由。"""
        if txn.symbol is not None and txn.qty != 0:
            if txn.qty > 0:
                self._apply_buy(txn)
            else:
                self._apply_sell(txn)

    def _apply_delist(self, txn: Transaction) -> None:
        """退市清算：按给定金额清空持仓。

        无法平仓时 ``amount`` 传 0，损失全额计入已实现盈亏——
        这正是幸存者偏差被消除后回测应当承受的代价。
        """
        assert txn.symbol is not None  # noqa: S101 - 已由 _validate 保证
        book = self._book(txn.symbol)
        cost = book.cost_total()
        book.realized_pnl += quantize_cny(txn.amount - cost)
        book.total_sold_qty += book.qty
        book.lots = []
        book.tax_cost_total = ZERO
        book.last_trade_date = txn.trade_date
        self._refresh_tax_basis(book)

    @staticmethod
    def _refresh_tax_basis(book: _SymbolBook) -> None:
        """重算券商口径成本。

        Args:
            book: 待刷新的批次账。
        """
        qty = book.qty
        book.cost_basis_tax = (
            quantize_price(safe_div(book.tax_cost_total, Decimal(qty))) if qty > 0 else ZERO
        )

    # ------------------------------------------------------------------ 查询
    @property
    def cash(self) -> Money:
        """当前现金。"""
        return quantize_cny(self._cash)

    def position(self, symbol: Symbol, *, as_of: TradeDate | None = None) -> Position | None:
        """取某标的的持仓。

        Args:
            symbol: 标的。
            as_of: 用于计算 T+1 可卖量的基准日；None 表示不区分可卖量。

        Returns:
            持仓；无持仓记录时返回 None。
        """
        book = self._books.get(symbol)
        if book is None or book.first_open_date is None:
            return None

        qty = book.qty
        cost_total = book.cost_total()
        # T+1：当日买入的批次不可卖（风控规则 A01）
        available = (
            sum(lot.remaining_qty for lot in book.lots if lot.open_date < as_of)
            if as_of is not None
            else qty
        )

        return Position(
            account_id=self.account_id,
            symbol=symbol,
            qty=qty,
            available_qty=available,
            frozen_qty=0,
            cost_basis_avg=quantize_price(safe_div(cost_total, Decimal(qty))) if qty > 0 else ZERO,
            cost_basis_tax=book.cost_basis_tax,
            first_open_date=book.first_open_date,
            last_trade_date=book.last_trade_date or book.first_open_date,
            lots=tuple(book.lots),
            realized_pnl=quantize_cny(book.realized_pnl),
            total_dividend=quantize_cny(book.total_dividend),
            total_dividend_tax=quantize_cny(book.total_dividend_tax),
            total_fee=quantize_cny(book.total_fee),
            total_bought_qty=book.total_bought_qty,
            total_sold_qty=book.total_sold_qty,
        )

    def positions(self, *, as_of: TradeDate | None = None) -> dict[Symbol, Position]:
        """取全部**非空**持仓。

        Args:
            as_of: 用于计算可卖量的基准日。

        Returns:
            标的到持仓的映射，已清仓的标的不包含在内。
        """
        result: dict[Symbol, Position] = {}
        for symbol in self._books:
            pos = self.position(symbol, as_of=as_of)
            if pos is not None and not pos.is_empty:
                result[symbol] = pos
        return result

    def state(self, *, as_of: TradeDate | None = None) -> LedgerState:
        """导出账本完整状态快照。

        Args:
            as_of: 快照基准日；None 时取最后一笔流水的日期。

        Returns:
            账本状态。

        Raises:
            LedgerError: 账本为空且未指定 as_of。
        """
        effective = as_of or self._last_date
        if effective is None:
            msg = "账本为空，无法导出状态；请指定 as_of 或先应用流水"
            raise LedgerError(msg, account_id=self.account_id)

        return LedgerState(
            account_id=self.account_id,
            as_of=effective,
            cash=self.cash,
            positions=self.positions(as_of=as_of),
            realized_pnl=quantize_cny(
                sum((b.realized_pnl for b in self._books.values()), start=ZERO)
            ),
            total_fee=quantize_cny(sum((b.total_fee for b in self._books.values()), start=ZERO)),
            total_dividend=quantize_cny(
                sum((b.total_dividend for b in self._books.values()), start=ZERO)
            ),
            total_dividend_tax=quantize_cny(
                sum((b.total_dividend_tax for b in self._books.values()), start=ZERO)
            ),
            total_deposit=quantize_cny(self._total_deposit),
            total_withdraw=quantize_cny(self._total_withdraw),
        )

    def days_to_tax_free(self, symbol: Symbol, *, as_of: TradeDate) -> int | None:
        """距离该标的最早批次满 1 年免红利税还有几天。

        卖出建议中应据此提示"再持有 N 天可免征红利税"——
        对高股息标的可能价值数千元（见 docs/11-持仓账本规格.md 第五节）。

        Args:
            symbol: 标的。
            as_of: 基准日。

        Returns:
            剩余天数；已满 1 年或无持仓时返回 None。
        """
        book = self._books.get(symbol)
        if book is None or not book.lots:
            return None
        # 最早的批次最先被 FIFO 卖出，因此它决定了下一笔卖出的税档
        earliest = min(book.lots, key=lambda lot: lot.open_date)
        held = earliest.holding_days(as_of)
        remaining = 366 - held
        return remaining if remaining > 0 else None


def replay(
    account_id: AccountId,
    transactions: Iterable[Transaction],
    *,
    as_of: TradeDate | None = None,
) -> Ledger:
    """重放流水序列，得到账本。

    这是账本正确性的最终保障——派生数据（持仓、批次、快照）随时可从
    ``transactions.jsonl`` 完全重建。

    Args:
        account_id: 账户标识。
        transactions: 流水序列，会按 ``(trade_date, occurred_at)`` 排序后重放。
        as_of: 只重放到该日期（含）为止，用于还原历史时点的持仓。

    Returns:
        重放后的账本。

    Raises:
        LedgerError: 任一流水非法。
    """
    ledger = Ledger(account_id)
    ordered: Sequence[Transaction] = sorted(
        transactions, key=lambda t: (t.trade_date, t.occurred_at, t.txn_id)
    )
    for txn in ordered:
        if as_of is not None and txn.trade_date > as_of:
            break
        ledger.apply(txn)
    return ledger
