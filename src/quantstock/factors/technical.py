"""技术因子。

规范见 docs/03-功能规格.md F2.1。

**全部函数只接受"截至 T 日（含）"的序列，返回 T 日的因子值**——
函数签名上就不可能看到未来数据（红线 R2）。调用方负责保证传入的序列已按 PIT 截断。

价格序列必须是**后复权**（红线 R4），否则除权会被误当成暴跌。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "atr",
    "bias",
    "drawdown_from_peak",
    "ema",
    "macd",
    "momentum",
    "moving_average",
    "position_in_range",
    "realized_volatility",
    "reversal",
    "rsi",
    "volume_ratio",
]

_TRADING_DAYS_PER_YEAR = 252
_MIN_RETURN_SAMPLES = 2
"""计算样本标准差所需的最少收益率样本数。"""


def _tail(values: Sequence[float], window: int) -> Sequence[float]:
    """取序列末尾 window 个元素。

    Args:
        values: 序列。
        window: 窗口长度。

    Returns:
        末尾切片。

    Raises:
        ValueError: 窗口非正或数据不足。
    """
    if window <= 0:
        msg = f"窗口必须为正，收到 {window}"
        raise ValueError(msg)
    if len(values) < window:
        msg = f"数据不足：需要 {window} 条，仅有 {len(values)} 条"
        raise ValueError(msg)
    return values[-window:]


def moving_average(closes: Sequence[float], window: int) -> float:
    """简单移动平均。

    Args:
        closes: 后复权收盘价序列，最后一个为 T 日。
        window: 窗口长度。

    Returns:
        T 日的 MA 值。

    Raises:
        ValueError: 数据不足。
    """
    tail = _tail(closes, window)
    return sum(tail) / window


def ema(closes: Sequence[float], window: int) -> float:
    """指数移动平均。

    用全部可用数据递推而非只用窗口内数据——EMA 本就是无限记忆的。

    Args:
        closes: 后复权收盘价序列。
        window: 窗口长度，决定平滑系数。

    Returns:
        T 日的 EMA 值。

    Raises:
        ValueError: 数据不足。
    """
    if window <= 0:
        msg = f"窗口必须为正，收到 {window}"
        raise ValueError(msg)
    if not closes:
        msg = "价格序列为空"
        raise ValueError(msg)
    alpha = 2.0 / (window + 1)
    result = closes[0]
    for price in closes[1:]:
        result = alpha * price + (1 - alpha) * result
    return result


def momentum(closes: Sequence[float], window: int, *, skip_recent: int = 0) -> float:
    """动量：过去 window 个交易日的累计收益。

    ``skip_recent`` 用于剔除最近若干日——经典动量因子（12-1 动量）会跳过最近 1 个月，
    因为短期存在反转效应，不跳过会让动量与反转互相抵消。

    Args:
        closes: 后复权收盘价序列。
        window: 回看窗口。
        skip_recent: 跳过最近的交易日数。

    Returns:
        累计收益率。

    Raises:
        ValueError: 数据不足或参数非法。
    """
    if skip_recent < 0:
        msg = f"skip_recent 不能为负，收到 {skip_recent}"
        raise ValueError(msg)
    needed = window + skip_recent + 1
    if len(closes) < needed:
        msg = f"数据不足：需要 {needed} 条，仅有 {len(closes)} 条"
        raise ValueError(msg)
    end = len(closes) - skip_recent - 1
    start = end - window
    base = closes[start]
    if base <= 0:
        msg = "基期价格必须为正"
        raise ValueError(msg)
    return closes[end] / base - 1.0


def reversal(closes: Sequence[float], window: int) -> float:
    """短期反转：动量取反。

    Args:
        closes: 后复权收盘价序列。
        window: 回看窗口。

    Returns:
        反转因子值（越大表示近期跌得越多）。
    """
    return -momentum(closes, window)


def realized_volatility(closes: Sequence[float], window: int, *, annualize: bool = True) -> float:
    """已实现波动率。

    Args:
        closes: 后复权收盘价序列。
        window: 计算收益率的窗口（需要 window+1 个价格点）。
        annualize: 是否年化。

    Returns:
        波动率。

    Raises:
        ValueError: 数据不足。
    """
    prices = _tail(closes, window + 1)
    returns = [prices[i] / prices[i - 1] - 1.0 for i in range(1, len(prices)) if prices[i - 1] > 0]
    if len(returns) < _MIN_RETURN_SAMPLES:
        msg = "有效收益率样本不足"
        raise ValueError(msg)
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    vol = math.sqrt(variance)
    return vol * math.sqrt(_TRADING_DAYS_PER_YEAR) if annualize else vol


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    window: int = 20,
) -> float:
    """平均真实波幅。

    止损位用 ATR 而非固定百分比，是为了让止损宽度自适应个股波动——
    对高波动股用固定 8% 止损会被正常震荡打掉。

    Args:
        highs: 最高价序列。
        lows: 最低价序列。
        closes: 收盘价序列。三者长度必须一致。
        window: 窗口长度。

    Returns:
        ATR 值。

    Raises:
        ValueError: 序列长度不一致或数据不足。
    """
    if not (len(highs) == len(lows) == len(closes)):
        msg = f"高低收序列长度必须一致：{len(highs)}/{len(lows)}/{len(closes)}"
        raise ValueError(msg)
    if len(closes) < window + 1:
        msg = f"数据不足：需要 {window + 1} 条，仅有 {len(closes)} 条"
        raise ValueError(msg)

    true_ranges = [
        max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        for i in range(len(closes) - window, len(closes))
    ]
    return sum(true_ranges) / window


def rsi(closes: Sequence[float], window: int = 14) -> float:
    """相对强弱指标。

    Args:
        closes: 后复权收盘价序列。
        window: 窗口长度。

    Returns:
        RSI 值（0~100）。全程无下跌时返回 100。

    Raises:
        ValueError: 数据不足。
    """
    prices = _tail(closes, window + 1)
    gains = 0.0
    losses = 0.0
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    if losses == 0:
        return 100.0
    rs = (gains / window) / (losses / window)
    return 100.0 - 100.0 / (1.0 + rs)


def macd(
    closes: Sequence[float],
    *,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float, float]:
    """MACD 指标。

    Args:
        closes: 后复权收盘价序列。
        fast: 快线周期。
        slow: 慢线周期。
        signal: 信号线周期。

    Returns:
        ``(dif, dea, macd_hist)``。

    Raises:
        ValueError: 数据不足或参数非法。
    """
    if fast >= slow:
        msg = f"快线周期必须小于慢线周期：{fast} >= {slow}"
        raise ValueError(msg)
    if len(closes) < slow:
        msg = f"数据不足：需要 {slow} 条，仅有 {len(closes)} 条"
        raise ValueError(msg)

    # DEA 是 DIF 的 EMA，需要逐日重算 DIF 序列
    difs: list[float] = []
    for end in range(slow, len(closes) + 1):
        window = closes[:end]
        difs.append(ema(window, fast) - ema(window, slow))
    dif = difs[-1]
    dea = ema(difs, signal) if len(difs) > 1 else dif
    return dif, dea, (dif - dea) * 2


def bias(closes: Sequence[float], window: int) -> float:
    """乖离率：收盘价相对均线的偏离程度。

    Args:
        closes: 后复权收盘价序列。
        window: 均线窗口。

    Returns:
        乖离率。

    Raises:
        ValueError: 数据不足或均线为零。
    """
    ma = moving_average(closes, window)
    if ma == 0:
        msg = "均线值为零，无法计算乖离率"
        raise ValueError(msg)
    return closes[-1] / ma - 1.0


def volume_ratio(volumes: Sequence[float], window: int = 20) -> float:
    """量比：当日成交量相对过去 window 日均量。

    Args:
        volumes: 成交量序列。
        window: 均量窗口（不含当日）。

    Returns:
        量比。均量为零时返回 0。

    Raises:
        ValueError: 数据不足。
    """
    if len(volumes) < window + 1:
        msg = f"数据不足：需要 {window + 1} 条，仅有 {len(volumes)} 条"
        raise ValueError(msg)
    baseline = sum(volumes[-window - 1 : -1]) / window
    if baseline == 0:
        return 0.0
    return volumes[-1] / baseline


def position_in_range(closes: Sequence[float], window: int = 252) -> float:
    """价格在过去 window 日高低区间中的位置分位。

    直接用于建议解释支柱②的"处于 52 周区间 34% 分位"。

    Args:
        closes: 后复权收盘价序列。
        window: 区间长度，252 约为一年。

    Returns:
        分位 0~1；区间无波动时返回 0.5。

    Raises:
        ValueError: 数据不足。
    """
    tail = _tail(closes, window)
    lowest, highest = min(tail), max(tail)
    if highest == lowest:
        return 0.5
    return (tail[-1] - lowest) / (highest - lowest)


def drawdown_from_peak(closes: Sequence[float], window: int | None = None) -> float:
    """相对区间最高价的回撤。

    移动止损用它——相对**持仓期最高价**而非成本价，锁住已有浮盈。

    Args:
        closes: 后复权收盘价序列。
        window: 回看窗口；None 表示用全部序列。

    Returns:
        回撤（负值或 0）。

    Raises:
        ValueError: 序列为空。
    """
    if not closes:
        msg = "价格序列为空"
        raise ValueError(msg)
    tail = closes if window is None else _tail(closes, window)
    peak = max(tail)
    if peak <= 0:
        return 0.0
    return tail[-1] / peak - 1.0
