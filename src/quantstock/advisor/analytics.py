"""持仓与技术分析（建议解释支柱②）。

规范见 docs/11-持仓账本规格.md 第五节。

已持仓标的必须使用**真实成本与真实持仓历史**（来自账本），不得用市价近似——
"浮亏 1.69%"这种数字只有基于真实成本才有意义。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from quantstock.account.ledger import Ledger
from quantstock.advisor.types import PositionAnalytics
from quantstock.factors.technical import (
    atr,
    moving_average,
    position_in_range,
    volume_ratio,
)
from quantstock.infra.money import quantize_price, safe_div
from quantstock.infra.types import Money, Symbol, TradeDate
from quantstock.risk.costs import dividend_tax_rate

__all__ = ["ATR_STOP_MULTIPLE", "build_analytics"]


@dataclass(frozen=True, slots=True)
class _HoldingFields:
    """从账本提取的持仓维度。

    单独建类而非返回 ``dict[str, object]``——后者展开到 dataclass 时
    类型检查完全失效，正是最容易埋错的地方。
    """

    holding_days: int = 0
    cost_basis: Money | None = None
    unrealized_pnl_pct: Decimal | None = None
    weight_in_portfolio: Decimal | None = None
    holding_excess_vs_benchmark: Decimal | None = None
    days_to_tax_free: int | None = None
    tax_saving_if_wait: Money | None = None


ATR_STOP_MULTIPLE = Decimal("2.5")
"""ATR 止损倍数。用 ATR 而非固定百分比，让止损宽度自适应个股波动。"""

_MA_SHORT, _MA_MID, _MA_LONG = 5, 20, 60
_YEAR_WINDOW = 252


def build_analytics(
    *,
    symbol: Symbol,
    as_of: TradeDate,
    closes: Sequence[float],
    highs: Sequence[float] | None = None,
    lows: Sequence[float] | None = None,
    volumes: Sequence[float] | None = None,
    ledger: Ledger | None = None,
    total_value: Money | None = None,
    benchmark_return: Decimal | None = None,
) -> PositionAnalytics:
    """构建某标的的持仓与技术分析。

    数据不足的维度**留空而非填占位值**——``statements()`` 会自动省略，
    输出"MA60=0.00"这种假数据比不输出更糟。

    Args:
        symbol: 标的。
        as_of: 基准日。
        closes: 后复权收盘价序列（截至 as_of）。
        highs: 最高价序列，计算 ATR 用。
        lows: 最低价序列。
        volumes: 成交量序列。
        ledger: 账本。提供时会填充持仓维度。
        total_value: 账户总资产，用于算持仓占比。
        benchmark_return: 持有期内基准的收益率，用于算超额。

    Returns:
        分析结果。

    Raises:
        ValueError: 价格序列为空。
    """
    if not closes:
        msg = f"{symbol} 的价格序列为空，无法生成技术分析"
        raise ValueError(msg)

    price = quantize_price(Decimal(str(closes[-1])))
    ma5 = _safe_ma(closes, _MA_SHORT)
    ma20 = _safe_ma(closes, _MA_MID)
    ma60 = _safe_ma(closes, _MA_LONG)

    atr20: float | None = None
    if highs is not None and lows is not None and len(closes) > _MA_MID:
        try:
            atr20 = atr(list(highs), list(lows), list(closes), _MA_MID)
        except ValueError:
            atr20 = None

    stop_price: Money | None = None
    distance: Decimal | None = None
    if atr20 is not None and atr20 > 0:
        stop_price = quantize_price(price - ATR_STOP_MULTIPLE * Decimal(str(atr20)))
        distance = safe_div(stop_price - price, price)

    pct_range: float | None = None
    if len(closes) >= _MA_MID:
        pct_range = position_in_range(list(closes), min(len(closes), _YEAR_WINDOW))

    vol_ratio: float | None = None
    if volumes is not None and len(volumes) > _MA_MID:
        try:
            vol_ratio = volume_ratio(list(volumes), _MA_MID)
        except ValueError:
            vol_ratio = None

    holding = _holding_fields(
        symbol=symbol,
        as_of=as_of,
        price=price,
        ledger=ledger,
        total_value=total_value,
        benchmark_return=benchmark_return,
    )

    return PositionAnalytics(
        symbol=symbol,
        as_of=as_of,
        market_price=price,
        ma5=ma5,
        ma20=ma20,
        ma60=ma60,
        ma_alignment=_describe_alignment(ma5, ma20, ma60),
        pct_in_52w_range=pct_range,
        volume_vs_ma20=vol_ratio,
        atr20=atr20,
        stop_loss_price=stop_price,
        distance_to_stop_pct=distance,
        holding_days=holding.holding_days,
        cost_basis=holding.cost_basis,
        unrealized_pnl_pct=holding.unrealized_pnl_pct,
        weight_in_portfolio=holding.weight_in_portfolio,
        holding_excess_vs_benchmark=holding.holding_excess_vs_benchmark,
        days_to_tax_free=holding.days_to_tax_free,
        tax_saving_if_wait=holding.tax_saving_if_wait,
    )


def _holding_fields(
    *,
    symbol: Symbol,
    as_of: TradeDate,
    price: Money,
    ledger: Ledger | None,
    total_value: Money | None,
    benchmark_return: Decimal | None,
) -> _HoldingFields:
    """提取持仓相关维度。

    Args:
        symbol: 标的。
        as_of: 基准日。
        price: 现价。
        ledger: 账本。
        total_value: 账户总资产。
        benchmark_return: 基准收益。

    Returns:
        持仓维度；无持仓时返回全默认值。
    """
    if ledger is None:
        return _HoldingFields()
    position = ledger.position(symbol, as_of=as_of)
    if position is None or position.qty <= 0:
        return _HoldingFields()

    cost = position.cost_basis_avg
    pnl_pct = safe_div(price - cost, cost) if cost > 0 else None
    weight = (
        safe_div(price * position.qty, total_value)
        if total_value is not None and total_value > 0
        else None
    )
    excess = (
        pnl_pct - benchmark_return if pnl_pct is not None and benchmark_return is not None else None
    )

    days_left = ledger.days_to_tax_free(symbol, as_of=as_of)
    saving: Money | None = None
    if days_left is not None and position.total_dividend > 0:
        # 现在卖按当前档位缴税，等满一年则免征，差额即为等待的收益
        current_rate = dividend_tax_rate(position.holding_days(as_of))
        saving = (position.total_dividend * current_rate).quantize(Decimal("0.01"))
        if saving <= 0:
            saving = None

    return _HoldingFields(
        holding_days=position.holding_days(as_of),
        cost_basis=cost,
        unrealized_pnl_pct=pnl_pct,
        weight_in_portfolio=weight,
        holding_excess_vs_benchmark=excess,
        days_to_tax_free=days_left,
        tax_saving_if_wait=saving,
    )


def _safe_ma(closes: Sequence[float], window: int) -> float | None:
    """计算均线，数据不足返回 None 而非填 0。

    Args:
        closes: 收盘价序列。
        window: 窗口。

    Returns:
        均线值；数据不足时 None。
    """
    try:
        return moving_average(list(closes), window)
    except ValueError:
        return None


def _describe_alignment(ma5: float | None, ma20: float | None, ma60: float | None) -> str:
    """描述均线排列。

    Args:
        ma5: 5 日均线。
        ma20: 20 日均线。
        ma60: 60 日均线。

    Returns:
        "多头排列" / "空头排列" / "纠缠"；数据不足时为空串。
    """
    if ma5 is None or ma20 is None or ma60 is None:
        return ""
    if ma5 > ma20 > ma60:
        return "多头排列"
    if ma5 < ma20 < ma60:
        return "空头排列"
    return "纠缠"
