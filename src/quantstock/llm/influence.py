"""LLM 对打分的有界影响（红线 LR1、LR2）。

规范见 docs/10-大模型集成规格.md 第六节。

```
final_score = base_score × (1 + α × conviction_adjustment)
α = llm.max_influence（默认 0.15，硬上限 0.20）
adjustment ∈ [-1, 1]
```

**整个 LLM 集成的安全性都压在这个式子上**，所以它被单独放在一个模块里，
配一组属性测试，并且是 `llm` 包中唯一产出数值的地方。可以这样读它：

- LLM 完全失效（API 挂了、解析失败、预算超了）→ adjustment = 0
  → final = base → 系统退化为纯量化，**照常出建议**；
- LLM 出现幻觉给出极端值 → 最多让打分偏移 α（默认 15%），
  **不可能**凭空造出一笔交易，也不可能把一个负分翻成正分；
- 风控、组合约束、硬闸全部在这个式子的下游，**不受任何影响**。

α 的硬上限 0.20 由 ``LLMConfig`` 强制，这里再独立校验一次——
安全边界值得多一道冗余，何况这道校验只有一行。
"""

from __future__ import annotations

from dataclasses import dataclass

from quantstock.infra.errors import LLMError

__all__ = [
    "HARD_INFLUENCE_CAP",
    "InfluenceResult",
    "apply_conviction",
    "apply_exposure",
]

HARD_INFLUENCE_CAP = 0.20
"""α 的硬上限。配置超过它直接拒绝，不是截断——静默截断会让用户以为设置生效了。"""


@dataclass(frozen=True, slots=True)
class InfluenceResult:
    """一次影响施加的结果。

    保留全部中间量是为了界面的 🤖 展开视图（红线 LR8）：用户点开要能看到
    "原始分 0.62，LLM 调节 +0.4，α=0.15，最终分 0.657"，而不是只看到一个结果。
    """

    base_score: float
    adjustment: float
    alpha: float
    final_score: float
    applied: bool
    """LLM 是否实际参与。False 表示降级为纯量化。"""
    note: str = ""

    @property
    def delta(self) -> float:
        """绝对偏移量。"""
        return self.final_score - self.base_score

    @property
    def relative_delta(self) -> float:
        """相对偏移比例。base 为 0 时返回 0。"""
        return self.delta / self.base_score if self.base_score else 0.0

    def explain(self) -> str:
        """人类可读说明，供界面展开视图。

        Returns:
            说明文本。
        """
        if not self.applied:
            return f"未使用 LLM（{self.note or '已关闭'}），打分保持 {self.base_score:.4f}"
        return (
            f"🤖 LLM 调节 {self.adjustment:+.2f}（α={self.alpha:.2f}）："
            f"{self.base_score:.4f} → {self.final_score:.4f}"
            f"（{self.relative_delta:+.2%}）"
        )


def _validate_alpha(alpha: float) -> None:
    """校验 α 在硬上限内。

    Args:
        alpha: 影响系数。

    Raises:
        LLMError: α 为负或超过硬上限。
    """
    if alpha < 0:
        msg = "LLM 影响系数 α 不能为负"
        raise LLMError(msg, alpha=alpha)
    if alpha > HARD_INFLUENCE_CAP:
        msg = f"LLM 影响系数 α={alpha} 超过硬上限 {HARD_INFLUENCE_CAP}（红线 LR2）"
        raise LLMError(msg, alpha=alpha, cap=HARD_INFLUENCE_CAP)


def apply_conviction(
    base_score: float,
    adjustment: float | None,
    *,
    alpha: float,
    note: str = "",
) -> InfluenceResult:
    """把 L2 研判的调节值施加到打分上。

    ``adjustment`` 为 None 时表示 LLM 没有输出（关闭、失效、缓存未命中、
    校验不通过），此时原样返回 base——这条路径必须始终可用，
    它是"LLM 挂了系统照常工作"的实现。

    Args:
        base_score: 纯量化打分。
        adjustment: LLM 调节值 ``[-1, 1]``；None 表示不使用 LLM。
        alpha: 影响系数。
        note: 未施加时的原因说明。

    Returns:
        影响结果。

    Raises:
        LLMError: α 越界。
    """
    _validate_alpha(alpha)

    if adjustment is None:
        return InfluenceResult(
            base_score=base_score,
            adjustment=0.0,
            alpha=alpha,
            final_score=base_score,
            applied=False,
            note=note or "本次无 LLM 输出",
        )

    # 越界值裁剪而非报错：schemas 已经拦过一道，这里是纵深防御。
    # 到这一步还越界说明有代码路径绕过了契约校验，裁剪能保证影响仍然有界
    clamped = max(-1.0, min(1.0, adjustment))
    return InfluenceResult(
        base_score=base_score,
        adjustment=clamped,
        alpha=alpha,
        final_score=base_score * (1 + alpha * clamped),
        applied=True,
    )


def apply_exposure(
    base_exposure: float,
    adjustment: float | None,
    *,
    alpha: float,
    note: str = "",
) -> InfluenceResult:
    """把 L2 市场研判施加到总仓位中枢上。

    **只能减不能加**（这是与 ``apply_conviction`` 的关键差别）：
    正向调节被截断为 0。原因是不对称的——让模型基于新闻情绪把总仓位
    从 60% 推到 70%，是在最容易被情绪裹挟的时候加杠杆；而让它在风险
    信号密集时把仓位降下来，错了也只是少赚。

    Args:
        base_exposure: 纯量化的总仓位中枢。
        adjustment: LLM 调节值；None 表示不使用。
        alpha: 影响系数。
        note: 未施加时的原因说明。

    Returns:
        影响结果。

    Raises:
        LLMError: α 越界。
    """
    _validate_alpha(alpha)

    if adjustment is None:
        return InfluenceResult(
            base_score=base_exposure,
            adjustment=0.0,
            alpha=alpha,
            final_score=base_exposure,
            applied=False,
            note=note or "本次无 LLM 输出",
        )

    clamped = min(0.0, max(-1.0, adjustment))
    applied_note = "" if clamped == adjustment else "正向调节已截断——LLM 只能减仓不能加仓"
    return InfluenceResult(
        base_score=base_exposure,
        adjustment=clamped,
        alpha=alpha,
        final_score=base_exposure * (1 + alpha * clamped),
        applied=True,
        note=applied_note,
    )
