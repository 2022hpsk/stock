"""反幻觉校验（红线 LR5）。

规范见 docs/10-大模型集成规格.md 第六节。

三层校验，任一层不通过**整个输出作废**，降级为"本次不使用 LLM"：

1. **结构化解析**——pydantic 契约，见 ``schemas``；
2. **引用存在性**——``evidence_ref`` 必须指向真实存在的材料 ID；
3. **数值范围**——``conviction_adjustment`` 等落在契约范围内。

第 2 层是这里最重要的一条。模型编造一个看似合理的材料编号来支撑结论，
是幻觉最典型也最危险的形态：结论读起来有理有据，而依据根本不存在。
一旦允许"引用不存在的材料"，四支柱解释就从证据链退化成了作文。

**作废而非修补**：不要试图丢掉坏引用保留其余部分。模型在一条上编造，
其余条目的可信度同样存疑。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from quantstock.infra.errors import LLMOutputInvalidError
from quantstock.infra.logging import get_logger
from quantstock.llm.schemas import MarketJudgement, PositionJudgement

__all__ = [
    "ValidationOutcome",
    "extract_json",
    "parse_output",
    "validate_evidence_refs",
]

_log = get_logger(__name__)

_T = TypeVar("_T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """校验结果。"""

    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        """便于直接用于条件判断。"""
        return self.ok


def extract_json(raw: str) -> dict[str, Any]:
    """从模型回复里抽出 JSON 对象。

    模型常常在 JSON 外面裹一层 ```json 代码块，或者在前面加一句
    "好的，这是分析结果："。这里做**有限的**宽容：剥代码块、取第一个花括号
    到最后一个花括号。再多的猜测就越界了——那属于"让模型自由发挥"。

    Args:
        raw: 模型原始回复。

    Returns:
        解析出的字典。

    Raises:
        LLMOutputInvalidError: 无法解析出 JSON 对象。
    """
    text = raw.strip()
    if match := _FENCE.search(text):
        text = match.group(1).strip()

    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        msg = "模型回复中找不到 JSON 对象"
        raise LLMOutputInvalidError(msg, preview=raw[:200])

    try:
        loaded = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        msg = "模型回复不是合法 JSON"
        raise LLMOutputInvalidError(msg, error=str(exc), preview=raw[:200]) from exc

    if not isinstance(loaded, dict):
        msg = "模型回复的 JSON 不是对象"
        raise LLMOutputInvalidError(msg, preview=raw[:200])
    return loaded


def parse_output(raw: str, schema: type[_T]) -> _T:
    """解析并校验模型输出。

    Args:
        raw: 模型原始回复。
        schema: 目标 pydantic 契约。

    Returns:
        校验通过的对象。

    Raises:
        LLMOutputInvalidError: 解析或校验失败。
    """
    payload = extract_json(raw)
    try:
        return schema.model_validate(payload)
    except ValidationError as exc:
        msg = f"模型输出不符合 {schema.__name__} 契约"
        raise LLMOutputInvalidError(msg, errors=exc.errors(include_url=False)) from exc


def validate_evidence_refs(refs: frozenset[str], materials: Mapping[str, Any]) -> ValidationOutcome:
    """校验引用的材料 ID 全部真实存在。

    Args:
        refs: 输出中引用的 ID。
        materials: 提供给模型的材料。

    Returns:
        校验结果。
    """
    if fabricated := sorted(refs - set(materials)):
        return ValidationOutcome(
            ok=False,
            reason=f"引用了不存在的材料 {fabricated}，判定为幻觉，整个输出作废",
        )
    return ValidationOutcome(ok=True)


def validate_judgement(
    judgement: PositionJudgement | MarketJudgement, materials: Mapping[str, Any]
) -> ValidationOutcome:
    """对研判输出做完整的反幻觉校验。

    Args:
        judgement: 研判输出。
        materials: 提供给模型的材料。

    Returns:
        校验结果。
    """
    refs = judgement.evidence_refs()
    if not (outcome := validate_evidence_refs(refs, materials)):
        _log.warning("llm_hallucination_detected", reason=outcome.reason)
        return outcome

    # 说"材料不足"却给出方向性判断是自相矛盾——pydantic 已强制归零，
    # 这里再确认一次，因为归零逻辑一旦被改动，静默失效不会有任何症状
    if judgement.insufficient_evidence:
        value = (
            judgement.conviction_adjustment
            if isinstance(judgement, PositionJudgement)
            else judgement.exposure_adjustment
        )
        if value != 0.0:
            return ValidationOutcome(
                ok=False, reason="声明材料不足却给出了非零调节值，输出自相矛盾"
            )
    return ValidationOutcome(ok=True)
