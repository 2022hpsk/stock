"""因子处理流水线与有效性检验。

规范见 docs/03-功能规格.md F2.4、F2.5。

统一处理链：去极值 → 缺失填充 → 标准化 → 中性化。

**样本清洗是重中之重**：入场日涨停（无法买入）的样本若计入 IC，
会严重高估因子有效性——很多"神因子"本质上只是在预测涨停
（见 docs/08-差距分析与设计补强.md A2）。
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from quantstock.factors.types import ICStats, LayerStats
from quantstock.infra.types import Symbol

_MIN_SAMPLES = 2
"""计算相关性与标准差所需的最少样本数。"""

__all__ = [
    "LabelSample",
    "build_labels",
    "compute_ic",
    "layer_backtest",
    "neutralize",
    "rank_pct",
    "standardize",
    "winsorize",
]


def winsorize(values: Mapping[Symbol, float], *, n_mad: float = 3.0) -> dict[Symbol, float]:
    """MAD 去极值。

    用中位数绝对偏差而非标准差：标准差本身会被极端值污染，
    结果是"越极端的异常值越不容易被识别为异常值"。

    Args:
        values: 原始因子值。
        n_mad: 保留区间为中位数 ± n_mad × MAD。

    Returns:
        截断后的因子值。

    Raises:
        ValueError: n_mad 非正。
    """
    if n_mad <= 0:
        msg = f"n_mad 必须为正，收到 {n_mad}"
        raise ValueError(msg)
    if not values:
        return {}

    series = list(values.values())
    median = statistics.median(series)
    mad = statistics.median([abs(v - median) for v in series])
    if mad == 0:
        return dict(values)

    # 1.4826 使 MAD 在正态分布下与标准差同尺度
    scale = 1.4826 * mad * n_mad
    lower, upper = median - scale, median + scale
    return {sym: min(max(v, lower), upper) for sym, v in values.items()}


def fill_missing(
    values: Mapping[Symbol, float | None],
    *,
    groups: Mapping[Symbol, str] | None = None,
) -> dict[Symbol, float]:
    """缺失值填充。

    有行业分组时用行业中位数，否则用全体中位数——
    用 0 填充会把缺失样本错误地放到分布中间偏离真实位置的地方。

    Args:
        values: 含缺失（None）的因子值。
        groups: 标的到行业的映射。

    Returns:
        填充后的因子值。全部缺失时返回空字典。
    """
    present: dict[Symbol, float] = {s: v for s, v in values.items() if v is not None}
    if not present:
        return {}

    overall: float = statistics.median(present.values())
    group_median: dict[str, float] = {}
    if groups:
        buckets: dict[str, list[float]] = {}
        for known_sym, known_value in present.items():
            buckets.setdefault(groups.get(known_sym, ""), []).append(known_value)
        group_median = {g: statistics.median(vs) for g, vs in buckets.items()}

    filled: dict[Symbol, float] = {}
    for sym, raw in values.items():
        if raw is not None:
            filled[sym] = raw
            continue
        group = groups.get(sym, "") if groups else ""
        filled[sym] = group_median.get(group, overall)
    return filled


def standardize(values: Mapping[Symbol, float]) -> dict[Symbol, float]:
    """横截面 z-score 标准化。

    Args:
        values: 因子值。

    Returns:
        标准化后的值；样本不足 2 个或方差为零时全部返回 0。
    """
    if len(values) < _MIN_SAMPLES:
        return dict.fromkeys(values, 0.0)
    series = list(values.values())
    mean = statistics.fmean(series)
    stdev = statistics.stdev(series)
    if stdev == 0:
        return dict.fromkeys(values, 0.0)
    return {sym: (v - mean) / stdev for sym, v in values.items()}


def rank_pct(values: Mapping[Symbol, float]) -> dict[Symbol, float]:
    """横截面分位（0~1）。

    直接用于建议解释："momentum_60d 处于全市场 87% 分位"。

    Args:
        values: 因子值。

    Returns:
        各标的的分位；单个标的时返回 0.5。
    """
    if not values:
        return {}
    if len(values) == 1:
        return dict.fromkeys(values, 0.5)
    ordered = sorted(values.items(), key=lambda kv: kv[1])
    denom = len(ordered) - 1
    return {sym: idx / denom for idx, (sym, _) in enumerate(ordered)}


def neutralize(
    values: Mapping[Symbol, float],
    *,
    groups: Mapping[Symbol, str],
) -> dict[Symbol, float]:
    """行业中性化：减去所属行业均值。

    不做中性化时，"选出高动量股"很可能只是"选出了当期最强的行业"——
    组合风险会集中在少数行业上而不自知。

    Args:
        values: 因子值。
        groups: 标的到行业的映射。

    Returns:
        中性化后的值。
    """
    if not values:
        return {}
    buckets: dict[str, list[float]] = {}
    for sym, value in values.items():
        buckets.setdefault(groups.get(sym, ""), []).append(value)
    means = {g: statistics.fmean(vs) for g, vs in buckets.items()}
    return {sym: v - means[groups.get(sym, "")] for sym, v in values.items()}


# --------------------------------------------------------------------- 标签
@dataclass(frozen=True, slots=True)
class LabelSample:
    """一个训练/检验样本。"""

    symbol: Symbol
    factor_value: float
    forward_return: float
    excluded_reason: str = ""
    """非空表示该样本被剔除，值为剔除原因。"""

    @property
    def usable(self) -> bool:
        """该样本是否可用于 IC 计算。"""
        return not self.excluded_reason


def build_labels(
    *,
    factor_values: Mapping[Symbol, float],
    forward_returns: Mapping[Symbol, float],
    unbuyable_at_entry: frozenset[Symbol] = frozenset(),
    unsellable_at_exit: frozenset[Symbol] = frozenset(),
    exclude_unbuyable: bool = True,
    exclude_unsellable: bool = True,
) -> list[LabelSample]:
    """构造带清洗标记的样本集（F2.5）。

    **``exclude_unbuyable`` 是最关键的一条**：入场日涨停或停牌的标的根本买不到，
    把它们的未来收益计入 IC 会严重高估因子有效性。
    实践中很多"惊人有效"的因子，去掉这类样本后 IC 会腰斩甚至归零。

    Args:
        factor_values: T 日的因子值。
        forward_returns: 未来 N 日收益。
        unbuyable_at_entry: 入场日涨停/停牌，无法买入的标的。
        unsellable_at_exit: 出场日跌停/停牌，无法卖出的标的。
        exclude_unbuyable: 是否剔除无法买入的样本。
        exclude_unsellable: 是否剔除无法卖出的样本。

    Returns:
        全部样本（含被剔除的，带 ``excluded_reason`` 便于对比）。
    """
    samples: list[LabelSample] = []
    for symbol, factor in factor_values.items():
        forward = forward_returns.get(symbol)
        if forward is None:
            continue
        reason = ""
        if exclude_unbuyable and symbol in unbuyable_at_entry:
            reason = "入场日涨停或停牌，无法买入"
        elif exclude_unsellable and symbol in unsellable_at_exit:
            reason = "出场日跌停或停牌，无法卖出"
        samples.append(
            LabelSample(
                symbol=symbol,
                factor_value=factor,
                forward_return=forward,
                excluded_reason=reason,
            )
        )
    return samples


# --------------------------------------------------------------------- 检验
def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman 秩相关。

    用秩相关而非皮尔逊相关：因子值分布常有厚尾，秩相关对异常值稳健得多。

    Args:
        xs: 序列一。
        ys: 序列二，长度须与 xs 一致。

    Returns:
        秩相关系数；样本不足或无变异时返回 0。
    """
    if len(xs) != len(ys) or len(xs) < _MIN_SAMPLES:
        return 0.0
    rank_x = _ranks(xs)
    rank_y = _ranks(ys)
    mean_x = statistics.fmean(rank_x)
    mean_y = statistics.fmean(rank_y)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in rank_x)
    var_y = sum((b - mean_y) ** 2 for b in rank_y)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _ranks(values: Sequence[float]) -> list[float]:
    """计算秩，并列取平均秩。

    Args:
        values: 数值序列。

    Returns:
        各元素的秩。
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def compute_ic(name: str, periods: Sequence[Sequence[LabelSample]]) -> ICStats:
    """计算多期 RankIC 统计。

    只使用 ``usable`` 的样本——被清洗标记剔除的样本不参与。

    Args:
        name: 因子名。
        periods: 每期一个样本列表。

    Returns:
        IC 统计结果。

    Raises:
        ValueError: 没有任何一期有足够样本。
    """
    ics: list[float] = []
    for samples in periods:
        usable = [s for s in samples if s.usable]
        if len(usable) < _MIN_SAMPLES:
            continue
        ics.append(
            _spearman(
                [s.factor_value for s in usable],
                [s.forward_return for s in usable],
            )
        )
    if not ics:
        msg = "没有任何一期有足够的可用样本，无法计算 IC"
        raise ValueError(msg)

    mean = statistics.fmean(ics)
    stdev = statistics.stdev(ics) if len(ics) > 1 else 0.0
    ir = mean / stdev if stdev > 0 else 0.0
    t_stat = ir * math.sqrt(len(ics)) if stdev > 0 else 0.0
    return ICStats(
        name=name,
        periods=len(ics),
        ic_mean=mean,
        ic_std=stdev,
        ir=ir,
        positive_rate=sum(1 for ic in ics if ic > 0) / len(ics),
        t_stat=t_stat,
    )


def layer_backtest(name: str, samples: Sequence[LabelSample], *, layers: int = 5) -> LayerStats:
    """分层回测：按因子值分组，比较各组的平均收益。

    单调性是比 IC 更直观的有效性判据——只有头尾两组有差异、
    中间乱序的因子多半是噪声。

    Args:
        name: 因子名。
        samples: 样本列表。
        layers: 分组数。

    Returns:
        分层结果，``mean_returns`` 按因子值从低到高排列。

    Raises:
        ValueError: 分组数非法或可用样本不足。
    """
    if layers < _MIN_SAMPLES:
        msg = f"分组数必须 ≥ 2，收到 {layers}"
        raise ValueError(msg)
    usable = sorted((s for s in samples if s.usable), key=lambda s: s.factor_value)
    if len(usable) < layers:
        msg = f"可用样本 {len(usable)} 少于分组数 {layers}"
        raise ValueError(msg)

    size = len(usable) / layers
    means: list[float] = []
    for i in range(layers):
        lo = round(i * size)
        hi = round((i + 1) * size)
        group = usable[lo:hi] or usable[lo : lo + 1]
        means.append(statistics.fmean(s.forward_return for s in group))

    return LayerStats(
        name=name,
        layers=layers,
        mean_returns=tuple(means),
        long_short_return=means[-1] - means[0],
    )
