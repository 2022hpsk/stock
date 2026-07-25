"""交易成本与市场规则参数测试。

重点：费率与涨跌幅限制必须按**生效日期**取值，回测用历史口径。
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantstock.costs import (
    ST_LIMIT_ALIGNED_DATE,
    STAMP_TAX_HALVED_DATE,
    CostModel,
    dividend_tax_rate,
    get_price_limit_pct,
)
from quantstock.infra.types import AssetType, Board, Side

BEFORE_HALVING = STAMP_TAX_HALVED_DATE - dt.timedelta(days=1)
TODAY = dt.date(2026, 7, 24)


class TestStampTax:
    def test_before_2023_08_28__full_rate(self) -> None:
        fees = CostModel().compute(
            amount=Decimal("100000"), side=Side.SELL, trade_date=BEFORE_HALVING
        )
        assert fees.stamp_tax == Decimal("100.00")  # 0.1%

    def test_on_and_after_2023_08_28__halved(self) -> None:
        fees = CostModel().compute(
            amount=Decimal("100000"), side=Side.SELL, trade_date=STAMP_TAX_HALVED_DATE
        )
        assert fees.stamp_tax == Decimal("50.00")  # 0.05%

    def test_buy_side__no_stamp_tax(self) -> None:
        """印花税单边征收，只在卖出时收。"""
        fees = CostModel().compute(amount=Decimal("100000"), side=Side.BUY, trade_date=TODAY)
        assert fees.stamp_tax == Decimal("0")

    @pytest.mark.parametrize("asset_type", [AssetType.ETF, AssetType.LOF])
    def test_fund__exempt(self, asset_type: AssetType) -> None:
        fees = CostModel().compute(
            amount=Decimal("100000"),
            side=Side.SELL,
            trade_date=TODAY,
            asset_type=asset_type,
        )
        assert fees.stamp_tax == Decimal("0")
        assert fees.transfer_fee == Decimal("0")


class TestCommission:
    def test_min_commission_applies_to_small_trades(self) -> None:
        """小额交易的成本主要来自最低佣金——这是 B06 最小成交金额约束的由来。"""
        fees = CostModel().compute(amount=Decimal("1000"), side=Side.BUY, trade_date=TODAY)
        assert fees.commission == Decimal("5")  # 1000 × 0.025% = 0.25 < 5

    def test_rate_applies_to_large_trades(self) -> None:
        fees = CostModel().compute(amount=Decimal("100000"), side=Side.BUY, trade_date=TODAY)
        assert fees.commission == Decimal("25.00")

    def test_net_commission_broker__regulatory_fees_charged_separately(self) -> None:
        model = CostModel(include_regulatory_in_commission=False)
        fees = model.compute(amount=Decimal("100000"), side=Side.BUY, trade_date=TODAY)
        assert fees.exchange_fee > 0
        assert fees.regulatory_fee > 0


class TestCostProperties:
    def test_zero_amount__rejected(self) -> None:
        with pytest.raises(ValueError, match="成交金额必须为正"):
            CostModel().compute(amount=Decimal("0"), side=Side.BUY, trade_date=TODAY)

    @given(amount=st.integers(min_value=1, max_value=50_000_000))
    def test_fees_never_negative(self, amount: int) -> None:
        for side in (Side.BUY, Side.SELL):
            fees = CostModel().compute(amount=Decimal(amount), side=side, trade_date=TODAY)
            assert fees.total >= 0

    @given(amount=st.integers(min_value=100_000, max_value=50_000_000))
    def test_sell_costs_more_than_buy(self, amount: int) -> None:
        """同金额下卖出费用必然更高——多了一道印花税。"""
        model = CostModel()
        buy = model.buy_cost(amount=Decimal(amount), trade_date=TODAY)
        sell = model.sell_cost(amount=Decimal(amount), trade_date=TODAY)
        assert sell > buy


class TestMaxAffordableQty:
    def test_accounts_for_fees(self) -> None:
        """必须保守：漏算费用会导致资金不足被废单。"""
        model = CostModel()
        qty = model.max_affordable_qty(cash=Decimal("10000"), price=Decimal("10"), trade_date=TODAY)
        amount = Decimal(qty) * Decimal("10")
        assert amount + model.buy_cost(amount=amount, trade_date=TODAY) <= Decimal("10000")
        assert qty % 100 == 0

    def test_insufficient_for_one_lot__returns_zero(self) -> None:
        qty = CostModel().max_affordable_qty(
            cash=Decimal("100"), price=Decimal("1580"), trade_date=TODAY
        )
        assert qty == 0

    def test_invalid_price__raises(self) -> None:
        with pytest.raises(ValueError, match="价格必须为正"):
            CostModel().max_affordable_qty(
                cash=Decimal("10000"), price=Decimal("0"), trade_date=TODAY
            )

    @given(
        cash=st.integers(min_value=0, max_value=10_000_000),
        price_cents=st.integers(min_value=1, max_value=500_000),
    )
    def test_never_exceeds_cash(self, cash: int, price_cents: int) -> None:
        """不变量：买入总支出（含费用）永不超过可用资金。"""
        model = CostModel()
        price = Decimal(price_cents) / 100
        qty = model.max_affordable_qty(cash=Decimal(cash), price=price, trade_date=TODAY)
        if qty > 0:
            amount = price * qty
            assert amount + model.buy_cost(amount=amount, trade_date=TODAY) <= Decimal(cash)


class TestDividendTax:
    """差别化个人所得税：持股越久税率越低，直接影响卖出时机决策。"""

    @pytest.mark.parametrize(
        ("days", "expected"),
        [
            (0, "0.20"),
            (1, "0.20"),
            (30, "0.20"),  # 恰好 1 个月，仍是最高档
            (31, "0.10"),  # 跨过 1 个月
            (200, "0.10"),
            (365, "0.10"),  # 恰好 1 年，仍需缴税
            (366, "0"),  # 超过 1 年，免征
            (1000, "0"),
        ],
    )
    def test_tier_boundaries(self, days: int, expected: str) -> None:
        assert dividend_tax_rate(days) == Decimal(expected)

    def test_negative_days__raises(self) -> None:
        with pytest.raises(ValueError, match="持股天数不能为负"):
            dividend_tax_rate(-1)


class TestPriceLimits:
    def test_main_board_normal(self) -> None:
        assert get_price_limit_pct(Board.MAIN, as_of=TODAY) == Decimal("0.10")

    def test_st_before_2026_07_06__five_percent(self) -> None:
        """历史口径：ST 股曾为 ±5%。回测该区间必须用当时的值。"""
        before = ST_LIMIT_ALIGNED_DATE - dt.timedelta(days=1)
        assert get_price_limit_pct(Board.MAIN, as_of=before, is_st=True) == Decimal("0.05")

    def test_st_on_and_after_2026_07_06__ten_percent(self) -> None:
        assert get_price_limit_pct(Board.MAIN, as_of=ST_LIMIT_ALIGNED_DATE, is_st=True) == Decimal(
            "0.10"
        )

    @pytest.mark.parametrize(
        ("board", "expected"),
        [
            (Board.GEM, "0.20"),
            (Board.STAR, "0.20"),
            (Board.BSE, "0.30"),
            (Board.ETF, "0.10"),
        ],
    )
    def test_other_boards(self, board: Board, expected: str) -> None:
        assert get_price_limit_pct(board, as_of=TODAY) == Decimal(expected)

    @pytest.mark.parametrize("board", [Board.GEM, Board.STAR, Board.BSE])
    def test_new_listing__no_limit_for_first_five_days(self, board: Board) -> None:
        assert get_price_limit_pct(board, as_of=TODAY, trading_days_since_listing=0) is None
        assert get_price_limit_pct(board, as_of=TODAY, trading_days_since_listing=4) is None
        assert get_price_limit_pct(board, as_of=TODAY, trading_days_since_listing=5) is not None

    def test_main_board_new_listing__still_limited(self) -> None:
        """主板新股不适用"前5日不设限"规则。"""
        assert get_price_limit_pct(
            Board.MAIN, as_of=TODAY, trading_days_since_listing=0
        ) == Decimal("0.10")
