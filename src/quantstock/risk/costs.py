"""交易成本与市场规则参数。

规范见 docs/05-风控规范.md 第五节。

**核心设计：所有费率与限制都是「按生效日期的区间表」，不是硬编码常量。**
回测必须使用当时口径，实盘使用当前口径——否则 2023 年的回测会用今天的印花税率，
结果直接失真。

参数核对日期：2026-07-25。实盘前请以自己券商的实际费率与交易所最新公告二次核对。
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from quantstock.infra.money import ZERO, quantize_cny
from quantstock.infra.types import AssetType, Board, Money, Side

__all__ = [
    "STAMP_TAX_HALVED_DATE",
    "ST_LIMIT_ALIGNED_DATE",
    "CostModel",
    "DividendTaxTier",
    "FeeBreakdown",
    "dividend_tax_rate",
    "get_price_limit_pct",
]

STAMP_TAX_HALVED_DATE: Final[dt.date] = dt.date(2023, 8, 28)
"""印花税由 0.1% 减半至 0.05% 的生效日。"""

TRANSFER_FEE_UNIFIED_DATE: Final[dt.date] = dt.date(2022, 4, 29)
"""沪深过户费统一为成交额 0.001%（双向）的生效日。"""

ST_LIMIT_ALIGNED_DATE: Final[dt.date] = dt.date(2026, 7, 6)
"""沪深主板风险警示股票涨跌幅由 ±5% 调整为 ±10% 的生效日。

回测历史区间必须使用当时口径，不得用当前值回溯——否则 2025 年的 ST 股回测
会允许 10% 的日内波动，与事实不符。
"""


@dataclass(frozen=True, slots=True)
class _RatePeriod:
    """一段时期内生效的费率。"""

    effective_from: dt.date
    rate: Decimal


def _rate_at(periods: Sequence[_RatePeriod], on: dt.date) -> Decimal:
    """取指定日期生效的费率。

    Args:
        periods: 按 ``effective_from`` 升序排列的费率区间。
        on: 目标日期。

    Returns:
        该日期适用的费率；早于所有区间时返回最早一段的费率。
    """
    idx = bisect_right([p.effective_from for p in periods], on) - 1
    return periods[max(idx, 0)].rate


_STAMP_TAX: Final = (
    _RatePeriod(dt.date(1990, 1, 1), Decimal("0.001")),
    _RatePeriod(STAMP_TAX_HALVED_DATE, Decimal("0.0005")),
)

_TRANSFER_FEE: Final = (
    _RatePeriod(dt.date(1990, 1, 1), Decimal("0.00002")),
    _RatePeriod(TRANSFER_FEE_UNIFIED_DATE, Decimal("0.00001")),
)


# --------------------------------------------------------------------- 涨跌幅限制
@dataclass(frozen=True, slots=True)
class _LimitPeriod:
    """一段时期内某板块的涨跌幅限制。"""

    effective_from: dt.date
    pct: Decimal


_PRICE_LIMITS: Final[dict[tuple[Board, bool], tuple[_LimitPeriod, ...]]] = {
    (Board.MAIN, False): (_LimitPeriod(dt.date(1996, 12, 16), Decimal("0.10")),),
    # ST 股：2026-07-06 起由 5% 调整为 10%，与主板其它股票一致
    (Board.MAIN, True): (
        _LimitPeriod(dt.date(1996, 12, 16), Decimal("0.05")),
        _LimitPeriod(ST_LIMIT_ALIGNED_DATE, Decimal("0.10")),
    ),
    (Board.GEM, False): (_LimitPeriod(dt.date(2020, 8, 24), Decimal("0.20")),),
    (Board.GEM, True): (_LimitPeriod(dt.date(2020, 8, 24), Decimal("0.20")),),
    (Board.STAR, False): (_LimitPeriod(dt.date(2019, 7, 22), Decimal("0.20")),),
    (Board.STAR, True): (_LimitPeriod(dt.date(2019, 7, 22), Decimal("0.20")),),
    (Board.BSE, False): (_LimitPeriod(dt.date(2021, 11, 15), Decimal("0.30")),),
    (Board.BSE, True): (_LimitPeriod(dt.date(2021, 11, 15), Decimal("0.30")),),
    (Board.ETF, False): (_LimitPeriod(dt.date(1990, 1, 1), Decimal("0.10")),),
    (Board.ETF, True): (_LimitPeriod(dt.date(1990, 1, 1), Decimal("0.10")),),
}

_NEW_LISTING_NO_LIMIT_DAYS: Final[dict[Board, int]] = {
    Board.GEM: 5,
    Board.STAR: 5,
    Board.BSE: 5,
}


def get_price_limit_pct(
    board: Board,
    *,
    as_of: dt.date,
    is_st: bool = False,
    trading_days_since_listing: int | None = None,
) -> Decimal | None:
    """取指定日期、指定板块的涨跌幅限制比例。

    Args:
        board: 板块。
        as_of: 目标日期。回测必须传当时的日期而非今天。
        is_st: 是否为风险警示股票。
        trading_days_since_listing: 上市后已交易天数。创业板/科创板/北交所
            新股上市前 5 个交易日不设涨跌幅限制，传入该值可正确处理。

    Returns:
        涨跌幅比例（如 ``Decimal("0.10")`` 表示 ±10%）；不设限制时返回 None。
    """
    if (
        trading_days_since_listing is not None
        and (no_limit_days := _NEW_LISTING_NO_LIMIT_DAYS.get(board)) is not None
        and trading_days_since_listing < no_limit_days
    ):
        return None

    periods = _PRICE_LIMITS.get((board, is_st))
    if periods is None:
        return None
    idx = bisect_right([p.effective_from for p in periods], as_of) - 1
    return periods[max(idx, 0)].pct


# --------------------------------------------------------------------- 红利税
class DividendTaxTier:
    """红利税分档阈值（自然日）。

    差别化个人所得税（2015-09-08 起）：持股期限越长税率越低，
    目的是鼓励长期持有。这直接影响卖出决策——临近满 1 年时多持有几天可能省下不少钱。
    """

    ONE_MONTH_DAYS: Final = 30
    """≤ 1 个月：全额计入，实际税率 20%。"""
    ONE_YEAR_DAYS: Final = 365
    """1 个月 ~ 1 年：减半计入，实际税率 10%。> 1 年免征。"""


_TAX_RATE_SHORT: Final = Decimal("0.20")
_TAX_RATE_MEDIUM: Final = Decimal("0.10")
_TAX_RATE_LONG: Final = ZERO


def dividend_tax_rate(holding_days: int) -> Decimal:
    """按持股期限取红利税实际税率。

    红利税在**卖出时**由券商按持股期限补扣，不是分红当日扣除。
    持股期限按先进先出匹配到具体批次。

    Args:
        holding_days: 该批次从建仓日到卖出日的自然日数。

    Returns:
        实际税率：``0.20``（≤1月）/ ``0.10``（1月~1年）/ ``0``（>1年）。

    Raises:
        ValueError: 持股天数为负。
    """
    if holding_days < 0:
        msg = f"持股天数不能为负，收到 {holding_days}"
        raise ValueError(msg)
    if holding_days <= DividendTaxTier.ONE_MONTH_DAYS:
        return _TAX_RATE_SHORT
    if holding_days <= DividendTaxTier.ONE_YEAR_DAYS:
        return _TAX_RATE_MEDIUM
    return _TAX_RATE_LONG


# --------------------------------------------------------------------- 费用
@dataclass(frozen=True, slots=True)
class FeeBreakdown:
    """费用明细。

    逐项记录而非只存总额——对账与成本分析都需要明细
    （见 docs/11-持仓账本规格.md 第二节）。
    """

    commission: Money = ZERO
    stamp_tax: Money = ZERO
    transfer_fee: Money = ZERO
    exchange_fee: Money = ZERO
    regulatory_fee: Money = ZERO

    @property
    def total(self) -> Money:
        """费用合计。"""
        return quantize_cny(
            self.commission
            + self.stamp_tax
            + self.transfer_fee
            + self.exchange_fee
            + self.regulatory_fee
        )


@dataclass(frozen=True, slots=True)
class CostModel:
    """交易成本模型。

    回测与实盘建议使用**同一个实现**——两套实现迟早会算出不同结果，
    让回测失去意义。

    Attributes:
        commission_rate: 券商佣金费率，默认万 2.5。按自己券商实际调整。
        min_commission: 最低佣金（元/笔）。小额交易的成本占比主要来自这里。
        include_regulatory_in_commission: 券商是否"全佣"（规费已含在佣金内）。
            标注"净佣"的券商需设为 False 并单独计规费。
    """

    commission_rate: Decimal = Decimal("0.00025")
    min_commission: Money = Decimal("5")
    exchange_fee_rate: Decimal = Decimal("0.0000341")
    regulatory_fee_rate: Decimal = Decimal("0.00002")
    include_regulatory_in_commission: bool = True

    def compute(
        self,
        *,
        amount: Money,
        side: Side,
        trade_date: dt.date,
        asset_type: AssetType = AssetType.STOCK,
    ) -> FeeBreakdown:
        """计算一笔交易的费用明细。

        Args:
            amount: 成交金额（元），必须为正。
            side: 买入或卖出。
            trade_date: 成交日期，用于取当时生效的费率。
            asset_type: 资产类型。ETF/LOF 无印花税、无过户费。

        Returns:
            费用明细。

        Raises:
            ValueError: 成交金额非正。
        """
        if amount <= 0:
            msg = f"成交金额必须为正，收到 {amount}"
            raise ValueError(msg)

        commission = max(quantize_cny(amount * self.commission_rate), self.min_commission)

        is_fund = asset_type in {AssetType.ETF, AssetType.LOF}

        # 印花税：仅卖出单边征收；ETF/LOF 免征
        if side is Side.SELL and not is_fund:
            stamp_tax = quantize_cny(amount * _rate_at(_STAMP_TAX, trade_date))
        else:
            stamp_tax = ZERO

        # 过户费：双向；ETF/LOF 免征
        transfer_fee = (
            ZERO if is_fund else quantize_cny(amount * _rate_at(_TRANSFER_FEE, trade_date))
        )

        # 规费：多数券商已含在佣金内（"全佣"）
        if self.include_regulatory_in_commission:
            exchange_fee = ZERO
            regulatory_fee = ZERO
        else:
            exchange_fee = quantize_cny(amount * self.exchange_fee_rate)
            regulatory_fee = quantize_cny(amount * self.regulatory_fee_rate)

        return FeeBreakdown(
            commission=commission,
            stamp_tax=stamp_tax,
            transfer_fee=transfer_fee,
            exchange_fee=exchange_fee,
            regulatory_fee=regulatory_fee,
        )

    def buy_cost(
        self, *, amount: Money, trade_date: dt.date, asset_type: AssetType = AssetType.STOCK
    ) -> Money:
        """买入的总费用。

        Args:
            amount: 成交金额。
            trade_date: 成交日期。
            asset_type: 资产类型。

        Returns:
            费用合计。
        """
        return self.compute(
            amount=amount, side=Side.BUY, trade_date=trade_date, asset_type=asset_type
        ).total

    def sell_cost(
        self, *, amount: Money, trade_date: dt.date, asset_type: AssetType = AssetType.STOCK
    ) -> Money:
        """卖出的总费用（不含红利税，红利税按批次单独计算）。

        Args:
            amount: 成交金额。
            trade_date: 成交日期。
            asset_type: 资产类型。

        Returns:
            费用合计。
        """
        return self.compute(
            amount=amount, side=Side.SELL, trade_date=trade_date, asset_type=asset_type
        ).total

    def max_affordable_qty(
        self,
        *,
        cash: Money,
        price: Money,
        trade_date: dt.date,
        lot_size: int = 100,
        asset_type: AssetType = AssetType.STOCK,
    ) -> int:
        """给定可用资金，最多能买多少股（含费用，已对齐整手）。

        必须保守：宁可少买一手，也不能因为漏算费用导致资金不足被废单。

        Args:
            cash: 可用资金。
            price: 委托价格。
            trade_date: 成交日期。
            lot_size: 每手股数。
            asset_type: 资产类型。

        Returns:
            可买股数，已对齐整手；买不起一手时返回 0。

        Raises:
            ValueError: 价格非正。
        """
        if price <= 0:
            msg = f"价格必须为正，收到 {price}"
            raise ValueError(msg)

        # 先按不含费用估算，再逐手回退直到资金足够。
        # 手数通常个位数到几十，循环代价可忽略，而闭式解在最低佣金的分段处容易出错。
        lots = int(cash / (price * lot_size))
        while lots > 0:
            qty = lots * lot_size
            amount = price * qty
            if (
                amount + self.buy_cost(amount=amount, trade_date=trade_date, asset_type=asset_type)
                <= cash
            ):
                return qty
            lots -= 1
        return 0
