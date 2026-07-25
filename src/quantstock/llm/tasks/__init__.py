"""LLM 三类任务：L1 情报理解、L2 研判归纳、L3 解释生成。

见 docs/10-大模型集成规格.md 第二节。三类任务对决策的影响是递减的：
L1 进有界软因子，L2 走唯一数值出口 conviction_adjustment，L3 影响为零。
"""

from quantstock.llm.tasks.explain import ExplainOutcome, ExplainTask
from quantstock.llm.tasks.intel_classify import ClassifyOutcome, IntelClassifyTask
from quantstock.llm.tasks.judge import JudgeOutcome, MarketJudgeTask, PositionJudgeTask

__all__ = [
    "ClassifyOutcome",
    "ExplainOutcome",
    "ExplainTask",
    "IntelClassifyTask",
    "JudgeOutcome",
    "MarketJudgeTask",
    "PositionJudgeTask",
]
