"""持仓账本测试。

覆盖 docs/11-持仓账本规格.md 第十一节要求的全部不变量：
重放一致性、FIFO 正确性、红利税边界、送股不重置期限、T+1 可卖量、恒等式、幂等。
"""

from __future__ import annotations

import datetime as dt
import itertools
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from quantstock.account.ledger import Ledger, replay
from quantstock.account.types import Transaction, TxnType
from quantstock.infra.clock import CST
from quantstock.infra.errors import LedgerError
from quantstock.infra.types import AccountId, Symbol

ACC = AccountId("main")
MAOTAI = Symbol("600519.SH")
CATL = Symbol("300750.SZ")

_counter = itertools.count(1)


def txn(
    txn_type: TxnType,
    *,
    day: dt.date,
    symbol: Symbol | None = MAOTAI,
    qty: int = 0,
    price: str = "0",
    amount: str = "0",
    commission: str = "0",
    stamp_tax: str = "0",
    net_cash: str = "0",
    ratio: str = "0",
    note: str = "",
) -> Transaction:
    """构造一笔流水，字段默认值让测试只需写出关心的部分。"""
    return Transaction(
        txn_id=f"T{next(_counter):05d}",
        account_id=ACC,
        txn_type=txn_type,
        trade_date=day,
        occurred_at=dt.datetime.combine(day, dt.time(14, 0), tzinfo=CST),
        symbol=symbol,
        qty=qty,
        price=Decimal(price),
        amount=Decimal(amount),
        commission=Decimal(commission),
        stamp_tax=Decimal(stamp_tax),
        net_cash=Decimal(net_cash),
        ratio=Decimal(ratio),
        note=note,
    )


def buy(day: dt.date, qty: int, price: str, *, symbol: Symbol = MAOTAI) -> Transaction:
    """买入流水，金额与现金影响自动算出（费用取 5 元最低佣金）。"""
    amount = Decimal(price) * qty
    return txn(
        TxnType.BUY,
        day=day,
        symbol=symbol,
        qty=qty,
        price=price,
        amount=str(amount),
        commission="5",
        net_cash=str(-(amount + 5)),
    )


def sell(day: dt.date, qty: int, price: str, *, symbol: Symbol = MAOTAI) -> Transaction:
    """卖出流水。"""
    amount = Decimal(price) * qty
    return txn(
        TxnType.SELL,
        day=day,
        symbol=symbol,
        qty=-qty,
        price=price,
        amount=str(amount),
        commission="5",
        net_cash=str(amount - 5),
    )


D1 = dt.date(2026, 1, 5)
D2 = dt.date(2026, 1, 6)
D3 = dt.date(2026, 3, 10)


class TestBasicFlow:
    def test_buy__creates_lot_with_fee_in_cost(self) -> None:
        ledger = replay(ACC, [buy(D1, 100, "100")])
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.qty == 100
        assert len(pos.lots) == 1
        # 成本含费用：(100×100 + 5) / 100 = 100.05
        assert pos.cost_basis_avg == Decimal("100.0500")

    def test_sell_all__position_empty(self) -> None:
        ledger = replay(ACC, [buy(D1, 100, "100"), sell(D2, 100, "110")])
        assert ledger.positions() == {}

    def test_cash_tracks_net_flows(self) -> None:
        ledger = replay(
            ACC,
            [
                txn(TxnType.DEPOSIT, day=D1, symbol=None, net_cash="100000"),
                buy(D1, 100, "100"),
            ],
        )
        assert ledger.cash == Decimal("89995.00")  # 100000 - 10000 - 5

    def test_sell_more_than_held__rejected(self) -> None:
        with pytest.raises(LedgerError, match="卖出数量超过持仓"):
            replay(ACC, [buy(D1, 100, "100"), sell(D2, 200, "110")])


class TestValidation:
    def test_duplicate_txn_id__rejected(self) -> None:
        """幂等：重复导入同一笔成交不得重复记账。"""
        first = buy(D1, 100, "100")
        ledger = Ledger(ACC)
        ledger.apply(first)
        with pytest.raises(LedgerError, match="流水重复"):
            ledger.apply(first)

    def test_wrong_account__rejected(self) -> None:
        ledger = Ledger(AccountId("other"))
        with pytest.raises(LedgerError, match="账户与账本不匹配"):
            ledger.apply(buy(D1, 100, "100"))

    def test_out_of_order__rejected(self) -> None:
        ledger = Ledger(ACC)
        ledger.apply(buy(D2, 100, "100"))
        with pytest.raises(LedgerError, match="按时间升序"):
            ledger.apply(buy(D1, 100, "100"))

    def test_adjust_without_note__rejected(self) -> None:
        """人工校正必须留下理由，否则账本无法审计。"""
        with pytest.raises(LedgerError, match="必须填写 note"):
            replay(ACC, [txn(TxnType.ADJUST, day=D1, qty=100, note="")])

    def test_missing_symbol__rejected(self) -> None:
        with pytest.raises(LedgerError, match="必须带 symbol"):
            replay(ACC, [txn(TxnType.BUY, day=D1, symbol=None, qty=100)])


class TestFifo:
    def test_consumes_oldest_lot_first(self) -> None:
        ledger = replay(
            ACC,
            [buy(D1, 100, "100"), buy(D2, 100, "200"), sell(D3, 100, "300")],
        )
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.qty == 100
        # 剩下的应是第二批（成本 200 档），第一批被消耗
        assert len(pos.lots) == 1
        assert pos.lots[0].open_date == D2
        assert ledger.consumptions[0].open_date == D1

    def test_partial_consumption_keeps_remainder(self) -> None:
        ledger = replay(ACC, [buy(D1, 300, "100"), sell(D2, 100, "110")])
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.qty == 200
        assert pos.lots[0].remaining_qty == 200
        assert pos.lots[0].open_date == D1  # 建仓日不变

    def test_spans_multiple_lots(self) -> None:
        ledger = replay(
            ACC,
            [buy(D1, 100, "100"), buy(D2, 100, "100"), sell(D3, 150, "120")],
        )
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.qty == 50
        assert len(ledger.consumptions) == 2
        assert [c.qty for c in ledger.consumptions] == [100, 50]


class TestT1Settlement:
    def test_same_day_purchase_not_sellable(self) -> None:
        """风控规则 A01：当日买入当日不可卖。"""
        ledger = replay(ACC, [buy(D1, 100, "100")])
        pos = ledger.position(MAOTAI, as_of=D1)
        assert pos is not None
        assert pos.qty == 100
        assert pos.available_qty == 0

    def test_next_day__sellable(self) -> None:
        ledger = replay(ACC, [buy(D1, 100, "100")])
        pos = ledger.position(MAOTAI, as_of=D2)
        assert pos is not None
        assert pos.available_qty == 100

    def test_mixed_lots__only_yesterday_sellable(self) -> None:
        ledger = replay(ACC, [buy(D1, 100, "100"), buy(D2, 200, "100")])
        pos = ledger.position(MAOTAI, as_of=D2)
        assert pos is not None
        assert pos.qty == 300
        assert pos.available_qty == 100


class TestDividendAndTax:
    """红利税按持股期限分三档，且在卖出时补扣而非分红当日扣。"""

    def _with_dividend(self, sell_day: dt.date) -> Ledger:
        return replay(
            ACC,
            [
                buy(D1, 1000, "100"),
                txn(TxnType.DIVIDEND_CASH, day=D2, amount="1000", net_cash="1000"),
                sell(sell_day, 1000, "110"),
            ],
        )

    def test_dividend_credited_to_cash(self) -> None:
        ledger = replay(
            ACC,
            [
                buy(D1, 1000, "100"),
                txn(TxnType.DIVIDEND_CASH, day=D2, amount="1000", net_cash="1000"),
            ],
        )
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.total_dividend == Decimal("1000.00")

    def test_sell_within_one_month__20_percent(self) -> None:
        ledger = self._with_dividend(D1 + dt.timedelta(days=20))
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.total_dividend_tax == Decimal("200.00")

    def test_sell_between_one_month_and_one_year__10_percent(self) -> None:
        ledger = self._with_dividend(D1 + dt.timedelta(days=100))
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.total_dividend_tax == Decimal("100.00")

    def test_sell_at_day_365__still_taxed(self) -> None:
        """恰好满 1 年仍需缴税，第 366 天才免征——这一天之差值 100 元。"""
        ledger = self._with_dividend(D1 + dt.timedelta(days=365))
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.total_dividend_tax == Decimal("100.00")

    def test_sell_at_day_366__exempt(self) -> None:
        ledger = self._with_dividend(D1 + dt.timedelta(days=366))
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.total_dividend_tax == Decimal("0.00")

    def test_dividend_not_taxed_on_receipt(self) -> None:
        """分红当日只入账不扣税。"""
        ledger = replay(
            ACC,
            [
                buy(D1, 1000, "100"),
                txn(TxnType.DIVIDEND_CASH, day=D2, amount="1000", net_cash="1000"),
            ],
        )
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.total_dividend_tax == Decimal("0.00")


class TestDaysToTaxFree:
    def test_reports_remaining_days(self) -> None:
        ledger = replay(ACC, [buy(D1, 100, "100")])
        as_of = D1 + dt.timedelta(days=350)
        assert ledger.days_to_tax_free(MAOTAI, as_of=as_of) == 16

    def test_already_exempt__returns_none(self) -> None:
        ledger = replay(ACC, [buy(D1, 100, "100")])
        assert ledger.days_to_tax_free(MAOTAI, as_of=D1 + dt.timedelta(days=400)) is None

    def test_no_position__returns_none(self) -> None:
        assert Ledger(ACC).days_to_tax_free(MAOTAI, as_of=D1) is None

    def test_uses_earliest_lot(self) -> None:
        """FIFO 下最早的批次最先卖出，因此它决定下一笔卖出的税档。"""
        ledger = replay(ACC, [buy(D1, 100, "100"), buy(D3, 100, "100")])
        as_of = D1 + dt.timedelta(days=300)
        assert ledger.days_to_tax_free(MAOTAI, as_of=as_of) == 66


class TestShareBonus:
    def test_increases_qty_and_lowers_cost(self) -> None:
        """10 送 3：股数 ×1.3，成本 ÷1.3。"""
        ledger = replay(
            ACC,
            [buy(D1, 1000, "130"), txn(TxnType.SHARE_BONUS, day=D2, ratio="0.3")],
        )
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.qty == 1300
        assert pos.cost_basis_avg == pytest.approx(Decimal("100.0038"), abs=Decimal("0.01"))

    def test_does_not_reset_holding_period(self) -> None:
        """关键：送股不重置建仓日，红利税档位不受影响。"""
        ledger = replay(
            ACC,
            [buy(D1, 1000, "130"), txn(TxnType.SHARE_BONUS, day=D3, ratio="0.3")],
        )
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.lots[0].open_date == D1


class TestDelisting:
    def test_worthless_delisting__full_loss_realized(self) -> None:
        """退市清零的损失必须真实入账——这正是消除幸存者偏差后应承受的代价。"""
        ledger = replay(
            ACC,
            [buy(D1, 1000, "100"), txn(TxnType.DELIST, day=D3, amount="0", net_cash="0")],
        )
        assert ledger.positions() == {}
        state = ledger.state()
        assert state.realized_pnl == Decimal("-100005.00")


class TestReplayConsistency:
    def test_replaying_twice__identical(self) -> None:
        """重放一致性：同一流水序列重放两次结果必须完全相同。"""
        txns = [
            buy(D1, 100, "100"),
            buy(D2, 200, "110"),
            txn(TxnType.DIVIDEND_CASH, day=D2, amount="300", net_cash="300"),
            sell(D3, 150, "120"),
        ]
        first = replay(ACC, txns).state()
        second = replay(ACC, txns).state()
        assert first == second

    def test_order_independence__sorted_by_date(self) -> None:
        """输入顺序不影响结果——replay 会按时间排序。"""
        txns = [buy(D1, 100, "100"), buy(D2, 100, "110")]
        assert replay(ACC, txns).state() == replay(ACC, list(reversed(txns))).state()

    def test_as_of__reconstructs_historical_state(self) -> None:
        """任意历史时点的持仓可精确还原。"""
        txns = [buy(D1, 100, "100"), buy(D2, 200, "110"), sell(D3, 300, "120")]
        historical = replay(ACC, txns, as_of=D2)
        pos = historical.position(MAOTAI)
        assert pos is not None
        assert pos.qty == 300

    def test_empty_ledger_state__raises(self) -> None:
        with pytest.raises(LedgerError, match="账本为空"):
            Ledger(ACC).state()


class TestMultiSymbol:
    def test_positions_are_independent(self) -> None:
        ledger = replay(
            ACC,
            [buy(D1, 100, "100", symbol=MAOTAI), buy(D1, 200, "50", symbol=CATL)],
        )
        positions = ledger.positions()
        assert set(positions) == {MAOTAI, CATL}
        assert positions[MAOTAI].qty == 100
        assert positions[CATL].qty == 200

    def test_closed_position_excluded(self) -> None:
        ledger = replay(
            ACC,
            [
                buy(D1, 100, "100", symbol=MAOTAI),
                buy(D1, 200, "50", symbol=CATL),
                sell(D2, 100, "110", symbol=MAOTAI),
            ],
        )
        assert set(ledger.positions()) == {CATL}


class TestInvariants:
    @settings(max_examples=50, deadline=None)
    @given(
        trades=st.lists(
            st.tuples(
                st.booleans(),  # True=买
                st.integers(min_value=1, max_value=20),  # 手数
                st.integers(min_value=1, max_value=500),  # 价格
            ),
            min_size=1,
            max_size=12,
        )
    )
    def test_cash_equals_sum_of_net_flows(self, trades: list[tuple[bool, int, int]]) -> None:
        """恒等式：现金 == 全部流水 net_cash 之和。"""
        ledger = Ledger(ACC)
        expected = Decimal("0")
        held = 0
        day = D1
        for is_buy, lots, price in trades:
            qty = lots * 100
            if not is_buy and held < qty:
                continue
            record = buy(day, qty, str(price)) if is_buy else sell(day, qty, str(price))
            ledger.apply(record)
            expected += record.net_cash
            held += qty if is_buy else -qty
            day += dt.timedelta(days=1)
        assert ledger.cash == expected

    @settings(max_examples=50, deadline=None)
    @given(
        buys=st.lists(
            st.tuples(st.integers(min_value=1, max_value=10), st.integers(1, 300)),
            min_size=1,
            max_size=8,
        )
    )
    def test_qty_equals_sum_of_lot_remainders(self, buys: list[tuple[int, int]]) -> None:
        """恒等式：持仓数量 == 各批次剩余量之和。"""
        ledger = Ledger(ACC)
        day = D1
        total = 0
        for lots, price in buys:
            qty = lots * 100
            ledger.apply(buy(day, qty, str(price)))
            total += qty
            day += dt.timedelta(days=1)
        pos = ledger.position(MAOTAI)
        assert pos is not None
        assert pos.qty == total
        assert sum(lot.remaining_qty for lot in pos.lots) == total
