"""研究纪律：试验记录与过拟合防御（A5）。

规范见 docs/08-差距分析与设计补强.md A5。

**问题**：如果试了 200 组参数，最优那组的 Sharpe 天然虚高。哪怕全是随机噪声，
200 次抽样里的最大值也会显著大于 0。样本外必然衰减——这不是运气不好，
是选择过程本身制造的偏差。

**防御四件套**：

1. **记录每一次尝试**（``trials.jsonl``，只追加）。只留最优结果等于把
   "试了多少次"这个关键信息扔掉，而后面三项全都依赖它。
2. **Deflated Sharpe Ratio**：按试验次数折减，得到"扣除选择偏差后真实
   Sharpe 仍大于 0"的概率。低于 0.95 表示"这个 Sharpe 用随机噪声就能试出来"。
3. **PBO**（过拟合概率，用 CSCV 计算）：枚举所有"一半时间当样本内、
   另一半当样本外"的切分，看样本内最优的那组在样本外落到中位数以下的比例。
   PBO > 0.5 意味着"选最优"这个动作还不如随机选。
4. **参数高原**：最优参数邻域的平均表现必须仍为正。尖峰意味着换个市场环境
   就完全失效。

DSR < 0.95 或 PBO > 0.5 的策略**禁止进入实盘候选池**，由 ``admission_check`` 强制。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from statistics import NormalDist, fmean, stdev
from typing import Any

from quantstock.infra.clock import now
from quantstock.infra.errors import StrategyError
from quantstock.infra.logging import get_logger

__all__ = [
    "DSR_FLOOR",
    "PBO_CEILING",
    "AdmissionVerdict",
    "Trial",
    "TrialLog",
    "admission_check",
    "deflated_sharpe_ratio",
    "dsr_z_score",
    "expected_max_sharpe",
    "parameter_plateau",
    "probability_of_backtest_overfitting",
]

_log = get_logger(__name__)

DSR_FLOOR = 0.95
"""DSR 准入下限。

DSR 是**概率**：扣除多重检验的选择偏差后，真实 Sharpe 仍大于 0 的概率。
0.95 是文献里的标准门槛，也就是通常意义上的 95% 置信。

docs/08 早先写的是"DSR < 0 禁止入池"。那个表述来自把 DSR 当成一个可正可负的
比率，但标准定义下它是概率、恒在 0~1。曾经试过把它映射成以 0 为分界的形式，
实测发现 ``2Φ(z)−1`` 在 |z| 超过 3 之后就完全饱和，60 期与 2000 期样本得到
一模一样的 −1.000000，分辨率全部丢失。所以这里回到标准定义，
门槛改用 0.95，并同步修正了文档。想看未饱和的量纲请用 ``dsr_z_score``。
"""

PBO_CEILING = 0.5
"""PBO 准入上限。高于它说明"选样本内最优"还不如随机选。"""

_EULER_MASCHERONI = 0.5772156649015329
_NORMAL = NormalDist()
MIN_TRIALS_FOR_PBO = 4
"""PBO 至少需要的试验数。样本太少时切分出来的组没有统计意义。"""


@dataclass(frozen=True, slots=True)
class Trial:
    """一次回测尝试。

    ``params`` 与全部指标都要留下——只记最优结果会让 DSR/PBO 无法计算，
    而那两个数字正是判断"这个策略是真的好还是试出来的"的唯一依据。
    """

    trial_id: str
    strategy: str
    params: dict[str, Any]
    sharpe: float
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    turnover: float = 0.0
    n_periods: int = 0
    """样本期数，用于 DSR 的方差修正。"""
    skew: float = 0.0
    kurtosis: float = 3.0
    """收益分布的偏度与峰度。DSR 对非正态收益的修正依赖它们。"""
    segment: str = "train"
    """``train`` / ``validation`` / ``test``。test 段每个策略只允许跑一次。"""
    note: str = ""
    created_at: str = ""

    def to_json(self) -> dict[str, Any]:
        """转成可落盘的字典。

        Returns:
            JSON 结构。
        """
        return {
            "trial_id": self.trial_id,
            "strategy": self.strategy,
            "params": self.params,
            "sharpe": self.sharpe,
            "annual_return": self.annual_return,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "n_periods": self.n_periods,
            "skew": self.skew,
            "kurtosis": self.kurtosis,
            "segment": self.segment,
            "note": self.note,
            "created_at": self.created_at or now().isoformat(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Trial:
        """从字典恢复。

        Args:
            payload: JSON 结构。

        Returns:
            试验记录。
        """
        return cls(
            trial_id=str(payload["trial_id"]),
            strategy=str(payload["strategy"]),
            params=dict(payload.get("params", {})),
            sharpe=float(payload["sharpe"]),
            annual_return=float(payload.get("annual_return", 0.0)),
            max_drawdown=float(payload.get("max_drawdown", 0.0)),
            turnover=float(payload.get("turnover", 0.0)),
            n_periods=int(payload.get("n_periods", 0)),
            skew=float(payload.get("skew", 0.0)),
            kurtosis=float(payload.get("kurtosis", 3.0)),
            segment=str(payload.get("segment", "train")),
            note=str(payload.get("note", "")),
            created_at=str(payload.get("created_at", "")),
        )


class TrialLog:
    """试验流水，只追加不修改。

    与账本同一原则：删掉失败的尝试会让 DSR 系统性偏乐观，
    而那恰好是这套机制要防的事。
    """

    def __init__(self, path: Path) -> None:
        """初始化。

        Args:
            path: 流水文件路径，通常是 ``var/research/trials.jsonl``。
        """
        self._path = Path(path)

    @property
    def path(self) -> Path:
        """流水文件路径。"""
        return self._path

    def append(self, trial: Trial) -> None:
        """追加一条试验记录。

        Args:
            trial: 试验记录。
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trial.to_json(), ensure_ascii=False) + "\n")

    def __iter__(self) -> Iterator[Trial]:
        """遍历全部记录。

        Yields:
            试验记录。损坏的行跳过并告警——一行坏数据不该让整份研究记录不可读。
        """
        if not self._path.exists():
            return
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                yield Trial.from_json(json.loads(line))
            except (json.JSONDecodeError, KeyError, ValueError):
                _log.warning("trial_log_corrupt_line", path=str(self._path))

    def for_strategy(self, strategy: str, *, segment: str | None = None) -> list[Trial]:
        """取某策略的试验记录。

        Args:
            strategy: 策略名。
            segment: 只取该数据段；None 表示全部。

        Returns:
            试验记录列表。
        """
        return [
            t for t in self if t.strategy == strategy and (segment is None or t.segment == segment)
        ]

    def count(self, strategy: str, *, segment: str | None = None) -> int:
        """试验次数。

        Args:
            strategy: 策略名。
            segment: 数据段。

        Returns:
            次数。
        """
        return len(self.for_strategy(strategy, segment=segment))

    def test_segment_used(self, strategy: str) -> bool:
        """Test 段是否已经用过。

        **测试集一次性使用**：跑完即锁定，重跑需换新的时间段。
        反复在同一测试集上调参，测试集就变成了第二个训练集。

        Args:
            strategy: 策略名。

        Returns:
            已用过则 True。
        """
        return self.count(strategy, segment="test") > 0


def expected_max_sharpe(n_trials: int, *, variance: float = 1.0) -> float:
    """N 次独立试验中最大 Sharpe 的期望值（零真实信号假设下）。

    这是 DSR 的核心：**哪怕全是噪声**，试 200 次里的最大值也会显著大于 0。
    用极值分布的近似式估计这个"白捡的"高度。

    Args:
        n_trials: 试验次数。
        variance: 试验间 Sharpe 的方差。

    Returns:
        期望最大值。``n_trials <= 1`` 时为 0。
    """
    if n_trials <= 1:
        return 0.0
    scale = math.sqrt(variance)
    # Bailey & López de Prado 的近似：结合 Gumbel 极值分布的两项
    term_a = (1 - _EULER_MASCHERONI) * _NORMAL.inv_cdf(1 - 1 / n_trials)
    term_b = _EULER_MASCHERONI * _NORMAL.inv_cdf(1 - 1 / (n_trials * math.e))
    return scale * (term_a + term_b)


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_periods: int,
    trial_variance: float = 1.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """折减后的 Sharpe 比率。

    先算出"零真实信号假设下，试 N 次能白捡到的最大 Sharpe"作为基准，
    把观测值减去它，再除以 Sharpe 估计量的标准误得到 z 值，返回 ``Φ(z)``。

    Args:
        observed_sharpe: 观测到的 Sharpe（样本内最优那组）。
        n_trials: 试验次数。
        n_periods: 样本期数。
        trial_variance: 试验间 Sharpe 的方差。
        skew: 收益分布偏度。
        kurtosis: 收益分布峰度（正态为 3）。

    Returns:
        0~1 的概率。**≥ 0.95 才算通过**；接近 0 表示这个 Sharpe
        用随机噪声就能试出来。

    Raises:
        ValueError: 样本期数不足。
    """
    return _NORMAL.cdf(
        dsr_z_score(
            observed_sharpe,
            n_trials=n_trials,
            n_periods=n_periods,
            trial_variance=trial_variance,
            skew=skew,
            kurtosis=kurtosis,
        )
    )


def dsr_z_score(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_periods: int,
    trial_variance: float = 1.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """DSR 背后的 z 值。

    概率形式在 |z| > 3 之后就饱和了（``Φ(3.5)`` 与 ``Φ(30)`` 打印出来都是 1.0），
    比较两个都"极显著"的策略时毫无分辨率。z 值不饱和，做敏感性分析时用它。

    Args:
        observed_sharpe: 观测到的 Sharpe。
        n_trials: 试验次数。
        n_periods: 样本期数。
        trial_variance: 试验间 Sharpe 的方差。
        skew: 收益分布偏度。
        kurtosis: 收益分布峰度。

    Returns:
        z 值。正得越多越可信。

    Raises:
        ValueError: 样本期数不足。
    """
    if n_periods < 2:  # noqa: PLR2004 - 标准差至少需要两个观测
        msg = f"计算 DSR 至少需要 2 个样本期，收到 {n_periods}"
        raise ValueError(msg)

    benchmark = expected_max_sharpe(n_trials, variance=trial_variance)
    variance = (1 - skew * observed_sharpe + (kurtosis - 1) / 4 * observed_sharpe**2) / (
        n_periods - 1
    )
    if variance <= 0:
        # 极端偏度下修正项可能为负，退回未修正的标准误而不是返回 NaN
        variance = 1 / (n_periods - 1)
    return (observed_sharpe - benchmark) / math.sqrt(variance)


def probability_of_backtest_overfitting(
    returns_matrix: Sequence[Sequence[float]], *, n_splits: int = 8
) -> float:
    """过拟合概率 PBO，用组合对称交叉验证（CSCV）计算。

    做法（López de Prado & Bailey）：把时间轴切成 S 个等长子集，
    枚举所有把 S/2 个子集当样本内、其余当样本外的组合。每个组合里：

    1. 在样本内挑出表现最好的那组参数；
    2. 看它在**样本外**的相对排名 ``w``；
    3. ``w ≤ 0.5``（即落到样本外中位数以下）就记一次"过拟合"。

    PBO 就是这个比例。它回答的问题是：**"挑样本内最优"这个动作本身，
    有没有比随机挑更好？** PBO > 0.5 表示还不如随机挑。

    **需要每组参数的完整收益序列而不是单个 Sharpe**——只有标量的话
    根本切不出样本内/样本外，那样算出来的"PBO"是自欺欺人。

    Args:
        returns_matrix: ``N × T`` 矩阵，每行是一组参数的逐期收益。
        n_splits: 时间轴切分数 S，必须为偶数。

    Returns:
        0~1 的过拟合概率。

    Raises:
        ValueError: 参数组数或期数不足，或 ``n_splits`` 非偶数。
    """
    if n_splits % 2 or n_splits < 2:  # noqa: PLR2004 - S 必须是 ≥2 的偶数
        msg = f"n_splits 必须是不小于 2 的偶数，收到 {n_splits}"
        raise ValueError(msg)

    matrix = [list(row) for row in returns_matrix]
    n_trials = len(matrix)
    if n_trials < MIN_TRIALS_FOR_PBO:
        msg = f"计算 PBO 至少需要 {MIN_TRIALS_FOR_PBO} 组参数，收到 {n_trials}"
        raise ValueError(msg)
    if len({len(row) for row in matrix}) != 1:
        msg = "各组参数的收益序列长度必须一致"
        raise ValueError(msg)

    n_periods = len(matrix[0])
    if n_periods < n_splits * 2:
        msg = f"期数 {n_periods} 不足以切成 {n_splits} 份（每份至少 2 期）"
        raise ValueError(msg)

    block = n_periods // n_splits
    blocks = [list(range(i * block, (i + 1) * block)) for i in range(n_splits)]

    overfit = 0
    total = 0
    for combo in combinations(range(n_splits), n_splits // 2):
        in_idx = [i for b in combo for i in blocks[b]]
        out_idx = [i for b in range(n_splits) if b not in combo for i in blocks[b]]

        in_scores = [_sharpe(row, in_idx) for row in matrix]
        out_scores = [_sharpe(row, out_idx) for row in matrix]

        best = max(range(n_trials), key=lambda n: in_scores[n])
        # 相对排名：样本外比它差的组数占比。1 表示样本外也最好，0 表示最差
        worse = sum(1 for n in range(n_trials) if n != best and out_scores[n] < out_scores[best])
        relative_rank = worse / (n_trials - 1)

        total += 1
        if relative_rank <= 0.5:  # noqa: PLR2004 - 中位数就是 0.5
            overfit += 1

    return overfit / total if total else 0.0


def _sharpe(series: Sequence[float], indices: Sequence[int]) -> float:
    """给定期数子集上的 Sharpe。

    Args:
        series: 完整收益序列。
        indices: 要用的期序号。

    Returns:
        Sharpe；波动为 0 时返回 0（无信息，不该被判为最优）。
    """
    values = [series[i] for i in indices]
    if len(values) < 2:  # noqa: PLR2004 - 标准差至少需要两个观测
        return 0.0
    spread = stdev(values)
    return fmean(values) / spread if spread > 0 else 0.0


@dataclass(frozen=True, slots=True)
class AdmissionVerdict:
    """实盘候选池准入结论。"""

    admitted: bool
    dsr: float
    pbo: float
    n_trials: int
    reasons: tuple[str, ...] = ()
    plateau_ok: bool = True

    def explain(self) -> str:
        """人类可读结论。

        Returns:
            结论文本。
        """
        verdict = "允许进入实盘候选池" if self.admitted else "禁止进入实盘候选池"
        detail = f"DSR={self.dsr:.3f}，PBO={self.pbo:.2f}，试验 {self.n_trials} 次"
        blockers = "；".join(self.reasons)
        return f"{verdict}（{detail}）" + (f"：{blockers}" if blockers else "")


def parameter_plateau(neighbourhood: Iterable[float], *, require_positive: bool = True) -> bool:
    """参数高原检验。

    最优参数邻域（默认 ±20%）的**平均**表现必须仍然为正。
    尖峰意味着参数稍微一动就失效，换个市场环境同样会失效——
    那种"最优参数"通常是拟合噪声的产物。

    Args:
        neighbourhood: 邻域内各组参数的表现。
        require_positive: 是否要求均值为正。

    Returns:
        通过则 True。邻域为空时按不通过——没检验过不等于通过。
    """
    values = list(neighbourhood)
    if not values:
        return False
    return fmean(values) > 0 if require_positive else True


def admission_check(
    trials: Sequence[Trial],
    *,
    returns_matrix: Sequence[Sequence[float]] | None = None,
    plateau: Sequence[float] | None = None,
) -> AdmissionVerdict:
    """实盘候选池准入检查（A5 强制门槛）。

    Args:
        trials: 该策略的全部试验记录。**必须是全部**，删掉失败的尝试
            会让 DSR 系统性偏乐观。
        returns_matrix: 与 ``trials`` 一一对应的逐期收益矩阵，用于 CSCV 算 PBO。
            缺省时跳过 PBO 检查并在理由中说明——**上线前必须补做**。
        plateau: 最优参数邻域的表现，用于参数高原检验。

    Returns:
        准入结论。

    Raises:
        StrategyError: 试验记录为空。
    """
    if not trials:
        msg = "没有任何试验记录，无法判断是否过拟合。禁止入池。"
        raise StrategyError(msg)

    sharpes = [t.sharpe for t in trials]
    best = max(trials, key=lambda t: t.sharpe)
    variance = stdev(sharpes) ** 2 if len(sharpes) > 1 else 1.0

    dsr = deflated_sharpe_ratio(
        best.sharpe,
        n_trials=len(trials),
        n_periods=max(best.n_periods, 2),
        trial_variance=max(variance, 1e-6),
        skew=best.skew,
        kurtosis=best.kurtosis,
    )

    reasons: list[str] = []
    pbo = 1.0
    pbo_computed = returns_matrix is not None
    if returns_matrix is not None:
        pbo = probability_of_backtest_overfitting(returns_matrix)
    else:
        # 没算不等于合格。缺省值取最悲观的 1.0，而不是让"没做检验"悄悄通过
        reasons.append("未提供逐期收益矩阵，PBO 无法计算——上线前必须补做")

    plateau_ok = parameter_plateau(plateau) if plateau is not None else False
    if plateau is None:
        reasons.append("未提供参数邻域表现，参数高原检验未做")
    elif not plateau_ok:
        reasons.append("参数高原检验不通过：最优参数邻域均值非正，疑似参数尖峰")

    if dsr < DSR_FLOOR:
        reasons.append(f"DSR={dsr:.3f} < {DSR_FLOOR}，该 Sharpe 用随机噪声即可试出")
    if pbo_computed and pbo > PBO_CEILING:
        reasons.append(f"PBO={pbo:.2f} > {PBO_CEILING}，选样本内最优不如随机选")

    admitted = dsr >= DSR_FLOOR and pbo_computed and pbo <= PBO_CEILING and plateau_ok
    if not admitted:
        _log.warning(
            "strategy_admission_denied",
            strategy=best.strategy,
            dsr=round(dsr, 4),
            pbo=round(pbo, 4),
            trials=len(trials),
        )
    return AdmissionVerdict(
        admitted=admitted,
        dsr=dsr,
        pbo=pbo,
        n_trials=len(trials),
        reasons=tuple(reasons),
        plateau_ok=plateau_ok,
    )


@dataclass
class TrialRecorder:
    """试验记录器，便于在参数扫描循环里顺手记录。

    刻意做成"记录是默认动作"：忘记记录的代价是 DSR 偏乐观，
    而那种错误没有任何症状。
    """

    log: TrialLog
    strategy: str
    segment: str = "train"
    _counter: int = field(default=0, init=False)

    def record(self, params: dict[str, Any], *, sharpe: float, **metrics: float) -> Trial:
        """记录一次尝试。

        Args:
            params: 本次参数。
            sharpe: Sharpe 比率。
            **metrics: 其它指标。

        Returns:
            写入的试验记录。
        """
        self._counter += 1
        trial = Trial(
            trial_id=f"{self.strategy}-{self.segment}-{self._counter:04d}",
            strategy=self.strategy,
            params=params,
            sharpe=sharpe,
            annual_return=metrics.get("annual_return", 0.0),
            max_drawdown=metrics.get("max_drawdown", 0.0),
            turnover=metrics.get("turnover", 0.0),
            n_periods=int(metrics.get("n_periods", 0)),
            skew=metrics.get("skew", 0.0),
            kurtosis=metrics.get("kurtosis", 3.0),
            segment=self.segment,
        )
        self.log.append(trial)
        return trial
