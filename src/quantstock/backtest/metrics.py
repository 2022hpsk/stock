"""绩效指标。

规范见 docs/03-功能规格.md F6.2、docs/08-差距分析与设计补强.md D2。

**TWR 与 MWR 必须同时报告**：
- TWR 剔除资金进出影响，衡量**策略能力**；
- MWR/IRR 衡量**实际赚了多少钱**。

只报其中一个都是误导——入金会被 MWR 计入"收益"，而 TWR 又看不出实际盈亏规模。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from quantstock.infra.types import Money, TradeDate

__all__ = [
    "DrawdownInfo",
    "PerformanceStats",
    "annualized_return",
    "compute_performance",
    "max_drawdown",
    "sharpe_ratio",
    "time_weighted_return",
]

TRADING_DAYS_PER_YEAR = 252
_MIN_SAMPLES = 2


@dataclass(frozen=True, slots=True)
class DrawdownInfo:
    """最大回撤信息。"""

    max_drawdown: float
    """最大回撤（负值）。"""
    peak_date: TradeDate | None
    trough_date: TradeDate | None
    recovery_date: TradeDate | None
    """回补到前高的日期；尚未回补时为 None。"""
    duration_days: int
    """从峰到谷的交易日数。"""


@dataclass(frozen=True, slots=True)
class PerformanceStats:
    """完整绩效指标。"""

    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_loss_ratio: float
    trading_days: int
    twr: float
    """时间加权收益率——衡量策略能力。"""
    mwr: float
    """资金加权收益率（IRR 近似）——衡量实际赚了多少。"""


def _returns_from_values(values: Sequence[float]) -> list[float]:
    """由净值序列算日收益率。

    Args:
        values: 净值序列。

    Returns:
        日收益率序列。
    """
    return [values[i] / values[i - 1] - 1.0 for i in range(1, len(values)) if values[i - 1] > 0]


def annualized_return(total_return: float, trading_days: int) -> float:
    """年化收益率。

    Args:
        total_return: 区间总收益率。
        trading_days: 区间交易日数。

    Returns:
        年化收益率；不足一天或本金归零时返回 0。
    """
    if trading_days <= 0 or total_return <= -1.0:
        return 0.0
    years = trading_days / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return 0.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def sharpe_ratio(returns: Sequence[float], *, risk_free_rate: float = 0.0) -> float:
    """夏普比率（年化）。

    Args:
        returns: 日收益率序列。
        risk_free_rate: 年化无风险利率。

    Returns:
        夏普比率；样本不足或无波动时返回 0。
    """
    if len(returns) < _MIN_SAMPLES:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = [r - daily_rf for r in returns]
    stdev = statistics.stdev(excess)
    if stdev == 0:
        return 0.0
    return statistics.fmean(excess) / stdev * math.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(returns: Sequence[float], *, risk_free_rate: float = 0.0) -> float:
    """索提诺比率（年化）。

    只惩罚下行波动——上涨的波动不是风险，这比夏普更贴合投资者的真实感受。

    Args:
        returns: 日收益率序列。
        risk_free_rate: 年化无风险利率。

    Returns:
        索提诺比率；无下行波动时返回 0。
    """
    if len(returns) < _MIN_SAMPLES:
        return 0.0
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess = [r - daily_rf for r in returns]
    downside = [r for r in excess if r < 0]
    if not downside:
        return 0.0
    downside_dev = math.sqrt(sum(r**2 for r in downside) / len(excess))
    if downside_dev == 0:
        return 0.0
    return statistics.fmean(excess) / downside_dev * math.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(  # noqa: C901 - 峰谷回补是一次线性扫描，拆分会让状态变量散落
    values: Sequence[float], dates: Sequence[TradeDate] | None = None
) -> DrawdownInfo:
    """最大回撤及其持续期。

    Args:
        values: 净值序列。
        dates: 对应日期，用于标注峰谷时点。

    Returns:
        回撤信息。
    """
    if not values:
        return DrawdownInfo(0.0, None, None, None, 0)

    peak = values[0]
    peak_idx = 0
    worst = 0.0
    worst_peak_idx = 0
    worst_trough_idx = 0

    for i, value in enumerate(values):
        if value > peak:
            peak = value
            peak_idx = i
        elif peak > 0:
            drawdown = value / peak - 1.0
            if drawdown < worst:
                worst = drawdown
                worst_peak_idx = peak_idx
                worst_trough_idx = i

    recovery_idx: int | None = None
    if worst < 0:
        peak_value = values[worst_peak_idx]
        for i in range(worst_trough_idx + 1, len(values)):
            if values[i] >= peak_value:
                recovery_idx = i
                break

    def at(idx: int | None) -> TradeDate | None:
        if idx is None or dates is None or idx >= len(dates):
            return None
        return dates[idx]

    return DrawdownInfo(
        max_drawdown=worst,
        peak_date=at(worst_peak_idx) if worst < 0 else None,
        trough_date=at(worst_trough_idx) if worst < 0 else None,
        recovery_date=at(recovery_idx),
        duration_days=worst_trough_idx - worst_peak_idx if worst < 0 else 0,
    )


def time_weighted_return(
    values: Sequence[float], cash_flows: Sequence[float] | None = None
) -> float:
    """时间加权收益率（TWR）。

    在每次资金进出处切段，各段收益率连乘——这样入金本身不会被算成收益。

    Args:
        values: 每期期末总资产。
        cash_flows: 每期发生的净资金流入（正为入金）。长度须与 values 一致。

    Returns:
        区间 TWR。

    Raises:
        ValueError: 序列长度不一致。
    """
    if len(values) < _MIN_SAMPLES:
        return 0.0
    flows = list(cash_flows) if cash_flows is not None else [0.0] * len(values)
    if len(flows) != len(values):
        msg = f"净值与资金流长度不一致：{len(values)} vs {len(flows)}"
        raise ValueError(msg)

    cumulative = 1.0
    for i in range(1, len(values)):
        # 期初资本 = 上期期末 + 本期流入，这样流入的钱不计入本期收益
        start_capital = values[i - 1] + flows[i]
        if start_capital <= 0:
            continue
        cumulative *= values[i] / start_capital
    return cumulative - 1.0


def money_weighted_return(
    *,
    initial_value: Money,
    final_value: Money,
    cash_flows: Sequence[tuple[int, Money]],
    total_days: int,
    max_iterations: int = 100,
) -> float:
    """资金加权收益率（IRR 近似，年化）。

    用二分法求解使净现值为零的贴现率。相比 TWR，它反映"你实际赚了多少"——
    在低点大额入金会拉高 MWR，这正是真实收益的一部分。

    Args:
        initial_value: 期初总资产。
        final_value: 期末总资产。
        cash_flows: ``(距期初的天数, 净流入金额)`` 列表。
        total_days: 区间总天数。
        max_iterations: 二分迭代次数。

    Returns:
        年化 MWR；无法求解时返回 0。
    """
    if total_days <= 0 or initial_value <= 0:
        return 0.0

    def npv(rate: float) -> float:
        """给定年化贴现率下的净现值。"""
        total = float(initial_value)
        for day, amount in cash_flows:
            total += float(amount) / ((1.0 + rate) ** (day / 365.0))
        return float(total - float(final_value) / ((1.0 + rate) ** (total_days / 365.0)))

    low, high = -0.99, 10.0
    if npv(low) * npv(high) > 0:
        return 0.0
    for _ in range(max_iterations):
        mid = (low + high) / 2
        if npv(low) * npv(mid) <= 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2


def compute_performance(
    *,
    values: Sequence[float],
    dates: Sequence[TradeDate] | None = None,
    cash_flows: Sequence[float] | None = None,
    trade_pnls: Sequence[float] = (),
    risk_free_rate: float = 0.0,
) -> PerformanceStats:
    """计算全套绩效指标。

    Args:
        values: 每个交易日的总资产。
        dates: 对应日期。
        cash_flows: 每日净资金流入，用于 TWR。
        trade_pnls: 各笔已平仓交易的盈亏，用于胜率与盈亏比。
        risk_free_rate: 年化无风险利率。

    Returns:
        绩效指标。

    Raises:
        ValueError: 净值序列为空。
    """
    if not values:
        msg = "净值序列为空，无法计算绩效"
        raise ValueError(msg)

    returns = _returns_from_values(values)
    total = values[-1] / values[0] - 1.0 if values[0] > 0 else 0.0
    trading_days = len(values)
    dd = max_drawdown(values, dates)

    annual = annualized_return(total, trading_days)
    volatility = (
        statistics.stdev(returns) * math.sqrt(TRADING_DAYS_PER_YEAR)
        if len(returns) >= _MIN_SAMPLES
        else 0.0
    )
    calmar = annual / abs(dd.max_drawdown) if dd.max_drawdown < 0 else 0.0

    wins = [p for p in trade_pnls if p > 0]
    losses = [p for p in trade_pnls if p < 0]
    win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    avg_win = statistics.fmean(wins) if wins else 0.0
    avg_loss = abs(statistics.fmean(losses)) if losses else 0.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0

    twr = time_weighted_return(values, cash_flows)
    net_flow = sum(cash_flows) if cash_flows else 0.0
    mwr = money_weighted_return(
        initial_value=Decimal(str(values[0])),
        final_value=Decimal(str(values[-1])),
        cash_flows=[(trading_days // 2, Decimal(str(net_flow)))] if net_flow else [],
        total_days=trading_days,
    )

    return PerformanceStats(
        total_return=total,
        annualized_return=annual,
        annualized_volatility=volatility,
        sharpe=sharpe_ratio(returns, risk_free_rate=risk_free_rate),
        sortino=sortino_ratio(returns, risk_free_rate=risk_free_rate),
        calmar=calmar,
        max_drawdown=dd.max_drawdown,
        max_drawdown_duration=dd.duration_days,
        win_rate=win_rate,
        profit_loss_ratio=pl_ratio,
        trading_days=trading_days,
        twr=twr,
        mwr=mwr,
    )
