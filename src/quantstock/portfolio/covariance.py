"""协方差估计：Ledoit-Wolf 收缩（A4）。

规范见 docs/08-差距分析与设计补强.md A4。

**为什么不能直接用样本协方差**：个人账户持仓通常 ≤12 只，而估计一个 N×N
协方差矩阵需要远多于 N 的观测。样本协方差在这种小样本下极不稳定——
最小特征值方向的估计误差最大，而均值方差优化恰好会**重仓押注**那个方向。
结果是优化器把全部资金压在估计误差上。

Ledoit-Wolf 的做法：把样本协方差朝一个结构化目标（这里用"等相关"矩阵）
收缩，收缩强度由数据本身解析地定出来，不需要交叉验证。

即便如此，docs/08 的结论仍是：**个人账户默认用波动率倒数而非均值方差**。
收缩协方差是给风险平价与相关性诊断用的，不是为了把均值方差扶正。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

__all__ = [
    "CovarianceEstimate",
    "correlation_from_covariance",
    "ledoit_wolf_shrinkage",
    "sample_covariance",
]

MIN_OBSERVATIONS = 2


@dataclass(frozen=True, slots=True)
class CovarianceEstimate:
    """协方差估计结果。"""

    matrix: tuple[tuple[float, ...], ...]
    shrinkage: float
    """收缩强度 0~1。接近 1 说明样本量相对维度太少，样本协方差几乎不可用。"""
    n_assets: int
    n_observations: int

    @property
    def is_heavily_shrunk(self) -> bool:
        """是否收缩得很厉害。

        超过 0.5 意味着一半以上的信息来自结构化先验而非数据——
        此时任何依赖协方差细节的优化都该被怀疑。
        """
        return self.shrinkage > 0.5  # noqa: PLR2004 - 一半就是一半

    def variance_of(self, index: int) -> float:
        """取某资产的方差。

        Args:
            index: 资产序号。

        Returns:
            方差。
        """
        return self.matrix[index][index]

    def explain(self) -> str:
        """人类可读说明。

        Returns:
            说明文本。
        """
        note = "（收缩强度高，协方差细节不可尽信）" if self.is_heavily_shrunk else ""
        return (
            f"{self.n_assets} 只标的 × {self.n_observations} 期观测，"
            f"收缩强度 {self.shrinkage:.2f}{note}"
        )


def _demean(returns: Sequence[Sequence[float]]) -> list[list[float]]:
    """按资产去均值。

    Args:
        returns: ``N × T`` 收益矩阵。

    Returns:
        去均值后的矩阵。
    """
    return [[v - fmean(row) for v in row] for row in returns]


def sample_covariance(returns: Sequence[Sequence[float]]) -> list[list[float]]:
    """样本协方差矩阵。

    Args:
        returns: ``N × T`` 收益矩阵，每行是一只标的的逐期收益。

    Returns:
        ``N × N`` 协方差矩阵。

    Raises:
        ValueError: 观测数不足或各行长度不一致。
    """
    if not returns:
        msg = "收益矩阵为空"
        raise ValueError(msg)
    if len({len(row) for row in returns}) != 1:
        msg = "各标的的收益序列长度必须一致"
        raise ValueError(msg)

    n_obs = len(returns[0])
    if n_obs < MIN_OBSERVATIONS:
        msg = f"至少需要 {MIN_OBSERVATIONS} 期观测，收到 {n_obs}"
        raise ValueError(msg)

    centred = _demean(returns)
    n_assets = len(returns)
    denominator = n_obs - 1
    return [
        [
            sum(centred[i][t] * centred[j][t] for t in range(n_obs)) / denominator
            for j in range(n_assets)
        ]
        for i in range(n_assets)
    ]


def _equicorrelation_target(sample: Sequence[Sequence[float]]) -> list[list[float]]:
    """构造"等相关"收缩目标。

    对角线保留各自方差，非对角线用**平均相关系数**重建。这个目标比
    "对角矩阵"更好：它保留了"资产之间确实相关"这个真实结构，
    只是不相信每一对的相关系数都被准确估计了。

    Args:
        sample: 样本协方差矩阵。

    Returns:
        目标矩阵。
    """
    n = len(sample)
    stds = [math.sqrt(sample[i][i]) if sample[i][i] > 0 else 0.0 for i in range(n)]

    correlations = [
        sample[i][j] / (stds[i] * stds[j])
        for i in range(n)
        for j in range(i + 1, n)
        if stds[i] > 0 and stds[j] > 0
    ]
    mean_corr = fmean(correlations) if correlations else 0.0

    return [
        [sample[i][i] if i == j else mean_corr * stds[i] * stds[j] for j in range(n)]
        for i in range(n)
    ]


def ledoit_wolf_shrinkage(
    returns: Sequence[Sequence[float]], *, forced_shrinkage: float | None = None
) -> CovarianceEstimate:
    """Ledoit-Wolf 收缩协方差估计。

    收缩强度 ``δ`` 由样本自身解析给出：估计误差越大（观测越少、维度越高），
    δ 越接近 1，结果越靠近结构化目标。

    Args:
        returns: ``N × T`` 收益矩阵。
        forced_shrinkage: 强制指定收缩强度，用于测试与敏感性分析。

    Returns:
        协方差估计。

    Raises:
        ValueError: 输入不合法，或 ``forced_shrinkage`` 越界。
    """
    if forced_shrinkage is not None and not 0.0 <= forced_shrinkage <= 1.0:
        msg = f"收缩强度必须在 0~1，收到 {forced_shrinkage}"
        raise ValueError(msg)

    sample = sample_covariance(returns)
    target = _equicorrelation_target(sample)
    n_assets = len(sample)
    n_obs = len(returns[0])

    if forced_shrinkage is not None:
        delta = forced_shrinkage
    else:
        delta = _optimal_shrinkage(returns, sample, target, n_assets, n_obs)

    matrix = tuple(
        tuple(delta * target[i][j] + (1 - delta) * sample[i][j] for j in range(n_assets))
        for i in range(n_assets)
    )
    return CovarianceEstimate(
        matrix=matrix, shrinkage=delta, n_assets=n_assets, n_observations=n_obs
    )


def _optimal_shrinkage(
    returns: Sequence[Sequence[float]],
    sample: Sequence[Sequence[float]],
    target: Sequence[Sequence[float]],
    n_assets: int,
    n_obs: int,
) -> float:
    """解析求最优收缩强度。

    ``δ* = (π̂ − ρ̂) / γ̂ / T``：

    - ``π̂``：样本协方差各元素的渐近方差之和——样本有多不可靠；
    - ``ρ̂``：样本与**目标**估计误差之间的协方差。目标本身也是从同一批数据
      估出来的，两者的误差是相关的；
    - ``γ̂``：样本与目标的 Frobenius 距离——目标有多"偏"。

    **``ρ̂`` 不能省**。省掉它会让 δ 系统性偏大，实测下来 8 只标的的场景里，
    15 期和 500 期观测都会算出 δ=1.0——那不是 Ledoit-Wolf，
    那是"永远只用等相关矩阵、完全不看数据"。这个错误没有任何症状：
    结果矩阵看起来完全正常，只是里面一点样本信息都没有。

    Args:
        returns: 收益矩阵。
        sample: 样本协方差。
        target: 收缩目标。
        n_assets: 资产数。
        n_obs: 观测数。

    Returns:
        0~1 的收缩强度。
    """
    centred = _demean(returns)
    stds = [math.sqrt(sample[i][i]) if sample[i][i] > 0 else 0.0 for i in range(n_assets)]
    mean_corr = _mean_correlation(sample, stds, n_assets)

    # π̂_ij：√T·s_ij 的渐近方差
    pi_matrix = [
        [_asymptotic_variance(centred[i], centred[j], sample[i][j], n_obs) for j in range(n_assets)]
        for i in range(n_assets)
    ]
    pi = sum(sum(row) for row in pi_matrix)

    # ρ̂：对角项直接取 π̂_ii；非对角项来自目标里 √(s_ii·s_jj) 对样本的依赖
    rho = sum(pi_matrix[i][i] for i in range(n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j or stds[i] <= 0 or stds[j] <= 0:
                continue
            cov_ii_ij = _asymptotic_covariance(
                centred[i], centred[i], centred[i], centred[j], sample[i][i], sample[i][j], n_obs
            )
            cov_jj_ij = _asymptotic_covariance(
                centred[j], centred[j], centred[i], centred[j], sample[j][j], sample[i][j], n_obs
            )
            rho += (mean_corr / 2) * (
                (stds[j] / stds[i]) * cov_ii_ij + (stds[i] / stds[j]) * cov_jj_ij
            )

    gamma = sum(
        (sample[i][j] - target[i][j]) ** 2 for i in range(n_assets) for j in range(n_assets)
    )
    if gamma <= 0:
        return 0.0
    return max(0.0, min(1.0, (pi - rho) / gamma / n_obs))


def _mean_correlation(
    sample: Sequence[Sequence[float]], stds: Sequence[float], n_assets: int
) -> float:
    """平均相关系数。

    Args:
        sample: 样本协方差。
        stds: 各资产标准差。
        n_assets: 资产数。

    Returns:
        平均相关系数；无有效资产对时为 0。
    """
    values = [
        sample[i][j] / (stds[i] * stds[j])
        for i in range(n_assets)
        for j in range(i + 1, n_assets)
        if stds[i] > 0 and stds[j] > 0
    ]
    return fmean(values) if values else 0.0


def _asymptotic_variance(
    left: Sequence[float], right: Sequence[float], covariance: float, n_obs: int
) -> float:
    """``√T·s_ij`` 的渐近方差。

    Args:
        left: 资产 i 的去均值收益。
        right: 资产 j 的去均值收益。
        covariance: 对应的样本协方差。
        n_obs: 观测数。

    Returns:
        渐近方差。
    """
    return sum((left[t] * right[t] - covariance) ** 2 for t in range(n_obs)) / n_obs


def _asymptotic_covariance(
    a: Sequence[float],
    b: Sequence[float],
    c: Sequence[float],
    d: Sequence[float],
    cov_ab: float,
    cov_cd: float,
    n_obs: int,
) -> float:
    """``√T·s_ab`` 与 ``√T·s_cd`` 的渐近协方差。

    Args:
        a: 第一对的左序列。
        b: 第一对的右序列。
        c: 第二对的左序列。
        d: 第二对的右序列。
        cov_ab: 第一对的样本协方差。
        cov_cd: 第二对的样本协方差。
        n_obs: 观测数。

    Returns:
        渐近协方差。
    """
    return sum((a[t] * b[t] - cov_ab) * (c[t] * d[t] - cov_cd) for t in range(n_obs)) / n_obs


def correlation_from_covariance(
    covariance: Sequence[Sequence[float]],
) -> list[list[float]]:
    """由协方差矩阵算相关系数矩阵。

    供"因子间相关性上限"与"持仓相关性诊断"使用——相关性过高等同重复下注。

    Args:
        covariance: 协方差矩阵。

    Returns:
        相关系数矩阵。方差为 0 的资产对应行列取 0。
    """
    n = len(covariance)
    stds = [math.sqrt(covariance[i][i]) if covariance[i][i] > 0 else 0.0 for i in range(n)]
    return [
        [
            covariance[i][j] / (stds[i] * stds[j]) if stds[i] > 0 and stds[j] > 0 else 0.0
            for j in range(n)
        ]
        for i in range(n)
    ]
