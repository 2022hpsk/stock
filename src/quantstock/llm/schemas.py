"""LLM 任务的结构化输入输出契约（红线 LR5）。

规范见 docs/10-大模型集成规格.md 第六节。

**全部用 pydantic 而非 dataclass**，与项目其它层相反。理由：这些对象是从
**模型返回的自由文本**里解析出来的，是不折不扣的外部输入。校验失败必须报错，
而不是让模型自由发挥或猜测——"猜一个合理值"正是幻觉进入决策的入口。

`extra="forbid"`：模型多吐一个字段就报错。这不是吹毛求疵——多出来的字段
通常意味着模型没有按契约作答，此时它给的其它字段同样不可信。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "CONVICTION_BOUND",
    "ExplanationOutput",
    "Factor",
    "IntelClassification",
    "LLMTaskInput",
    "MarketJudgement",
    "PositionJudgement",
    "RiskLevel",
]

CONVICTION_BOUND = 1.0
"""``conviction_adjustment`` 的取值边界。经 α 限幅后才影响打分。"""

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


class _Strict(BaseModel):
    """所有 LLM 契约的基类。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LLMTaskInput(_Strict):
    """任务输入的公共部分。

    ``materials`` 是**唯一允许的信息来源**：系统提示词会强制模型只依据它作答，
    材料不足时必须输出 ``insufficient_evidence`` 而不是调用先验知识。
    这是防训练集泄漏的主要结构性防御（红线 LR4）。
    """

    task_id: str
    as_of: str
    """决策时点。回测中用相对表述，见 ``strip_absolute_dates``。"""
    materials: dict[str, str] = Field(default_factory=dict)
    """材料 ID → 内容。输出里的 ``evidence_ref`` 必须指向这里存在的 key。"""

    def material_ids(self) -> frozenset[str]:
        """全部材料 ID。

        Returns:
            ID 集合，供反幻觉校验比对。
        """
        return frozenset(self.materials)


class Factor(_Strict):
    """一条研判要点。"""

    statement: str = Field(min_length=1)
    evidence_ref: str = Field(min_length=1)
    """引用的材料 ID。指向不存在的材料 → 整个输出作废（反幻觉校验）。"""

    def __str__(self) -> str:
        """渲染成带引用的一行。"""
        return f"{self.statement}（依据 {self.evidence_ref}）"


class IntelClassification(_Strict):
    """L1 情报理解的输出。

    只在规则/词典未命中时才调用（``only_if_rules_miss``），大幅省钱。
    """

    event_type: str | None = None
    """事件类型的字符串值。None 表示模型也无法归类。"""
    domain: str | None = None
    sentiment: float = Field(default=0.0, ge=-1.0, le=1.0)
    key_points: list[str] = Field(default_factory=list, max_length=5)
    insufficient_evidence: bool = False

    @model_validator(mode="after")
    def _neutralize_when_insufficient(self) -> IntelClassification:
        """材料不足时强制中性。

        Returns:
            校验后的自身。
        """
        if self.insufficient_evidence:
            object.__setattr__(self, "sentiment", 0.0)
            object.__setattr__(self, "event_type", None)
        return self


class PositionJudgement(_Strict):
    """L2 研判归纳的输出。

    **``conviction_adjustment`` 是 LLM 唯一的数值出口**（docs/10 第六节）。
    这个设计的价值在失效时才看得出来：模型挂了、解析失败、或者出现幻觉，
    最坏结果也只是这一个数字变成 0 或被 α 限幅，凭空造不出一笔交易。
    """

    symbol_ref: str = Field(min_length=1)
    """匿名模式下为代号（``标的A``），实盘为真实 Symbol。"""
    positive_factors: list[Factor] = Field(default_factory=list, max_length=8)
    negative_factors: list[Factor] = Field(default_factory=list, max_length=8)
    conflicts: list[str] = Field(default_factory=list, max_length=5)
    """证据间的矛盾点。这一项常常比结论本身更有价值。"""
    risk_level: RiskLevel = "MEDIUM"
    conviction_adjustment: float = Field(default=0.0, ge=-CONVICTION_BOUND, le=CONVICTION_BOUND)
    falsification: list[str] = Field(default_factory=list, max_length=5)
    """证伪条件，直接进四支柱④。"""
    insufficient_evidence: bool = False

    @model_validator(mode="after")
    def _zero_adjustment_when_insufficient(self) -> PositionJudgement:
        """材料不足时强制不调节。

        模型说"材料不够"却仍给出方向性调节，是自相矛盾——
        这种情况下的调节值几乎必然来自先验知识而非材料。

        Returns:
            校验后的自身。
        """
        if self.insufficient_evidence:
            object.__setattr__(self, "conviction_adjustment", 0.0)
        return self

    def evidence_refs(self) -> frozenset[str]:
        """输出中引用的全部材料 ID。

        Returns:
            ID 集合。
        """
        return frozenset(f.evidence_ref for f in (*self.positive_factors, *self.negative_factors))


class MarketJudgement(_Strict):
    """L2 市场环境定性的输出。"""

    regime: Literal["RISK_ON", "NEUTRAL", "RISK_OFF"] = "NEUTRAL"
    drivers: list[Factor] = Field(default_factory=list, max_length=6)
    risk_level: RiskLevel = "MEDIUM"
    exposure_adjustment: float = Field(default=0.0, ge=-CONVICTION_BOUND, le=CONVICTION_BOUND)
    """对总仓位中枢的建议调节。同样经 α 限幅，且只能减不能加（见 influence 模块）。"""
    insufficient_evidence: bool = False

    @model_validator(mode="after")
    def _zero_when_insufficient(self) -> MarketJudgement:
        """材料不足时强制不调节。

        Returns:
            校验后的自身。
        """
        if self.insufficient_evidence:
            object.__setattr__(self, "exposure_adjustment", 0.0)
        return self

    def evidence_refs(self) -> frozenset[str]:
        """引用的材料 ID。

        Returns:
            ID 集合。
        """
        return frozenset(d.evidence_ref for d in self.drivers)


class ExplanationOutput(_Strict):
    """L3 解释生成的输出。

    **对决策的影响为零**——它只把已经确定的结构化结论组织成中文行文。
    因此这里没有任何数值字段，这是刻意的：解释环节不该有能改变数字的通道。
    """

    verdict: str = Field(min_length=1, max_length=200)
    """一句话结论。"""
    narrative: list[str] = Field(default_factory=list, max_length=12)
    """分段行文。"""

    def as_text(self) -> str:
        """拼成完整文本。

        Returns:
            结论 + 行文。
        """
        return "\n".join([self.verdict, *self.narrative])
