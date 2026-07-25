"""L1 情报理解任务。

规范见 docs/10-大模型集成规格.md 第二节。

**只在规则/词典未命中时调用**（``only_if_rules_miss``）。这不只是省钱：
规则分类是确定性的、可回测的，同一条 2019 年的公告今天和明年重跑必须得到
同一个分类。能用规则解决的就不该交给会随模型版本漂移的东西。

成本上的差别也很大：L1 按**情报条目**做，每条只算一次、全市场共享，
而不是按"每天 × 每股"——后者是几个数量级的差距。
"""

from __future__ import annotations

from dataclasses import dataclass

from quantstock.llm.client import LLMClient, TaskCall
from quantstock.llm.prompts import INTEL_CLASSIFY_SYSTEM, PROMPT_VERSION, render_materials
from quantstock.llm.schemas import IntelClassification

__all__ = ["ClassifyOutcome", "IntelClassifyTask"]

TASK_ID = "intel_classify"


@dataclass(frozen=True, slots=True)
class ClassifyOutcome:
    """一次 L1 分类的结果。"""

    classification: IntelClassification | None
    classifier_tag: str
    """``llm:<model>`` 或 ``rule``。LLM 产出必须可识别（红线 I-R3）。"""
    reason: str = ""
    cache_key: str = ""
    from_cache: bool = False
    cost_usd: float = 0.0

    @property
    def used_llm(self) -> bool:
        """本次是否用上了 LLM。"""
        return self.classification is not None


class IntelClassifyTask:
    """情报分类任务。"""

    def __init__(
        self,
        client: LLMClient,
        *,
        model_id: str,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        """初始化。

        Args:
            client: LLM 客户端。
            model_id: 模型 ID。
            prompt_version: 提示词版本。
        """
        self._client = client
        self._model_id = model_id
        self._prompt_version = prompt_version

    def build_call(self, *, item_id: str, title: str, body: str = "") -> TaskCall:
        """构造一次调用，供 ``backfill`` 批量预计算复用。

        Args:
            item_id: 情报 ID，作为材料编号。
            title: 标题。
            body: 正文。

        Returns:
            任务调用。
        """
        materials = {item_id: f"{title}\n{body}".strip()}
        return TaskCall(
            task_id=TASK_ID,
            prompt_version=self._prompt_version,
            model_id=self._model_id,
            system=INTEL_CLASSIFY_SYSTEM,
            user=render_materials(materials),
            payload={"materials": materials},
            max_tokens=1024,
        )

    def run(self, *, item_id: str, title: str, body: str = "") -> ClassifyOutcome:
        """对一条情报做分类。

        Args:
            item_id: 情报 ID。
            title: 标题。
            body: 正文。

        Returns:
            分类结果。失败时 ``classification`` 为 None，上层保留规则分类。
        """
        call = self.build_call(item_id=item_id, title=title, body=body)
        result = self._client.complete(call, IntelClassification)

        if result.output is None:
            return ClassifyOutcome(
                classification=None,
                classifier_tag="rule",
                reason=result.reason,
                cache_key=result.cache_key,
                from_cache=result.from_cache,
                cost_usd=result.cost_usd,
            )

        classification = result.output
        assert isinstance(classification, IntelClassification)  # noqa: S101 - complete 保证契约

        if classification.insufficient_evidence:
            # 模型自己说材料不足，那就别用它的分类
            return ClassifyOutcome(
                classification=None,
                classifier_tag="rule",
                reason="模型判定材料不足，保留规则分类",
                cache_key=result.cache_key,
                from_cache=result.from_cache,
                cost_usd=result.cost_usd,
            )

        return ClassifyOutcome(
            classification=classification,
            classifier_tag=f"llm:{self._model_id}",
            cache_key=result.cache_key,
            from_cache=result.from_cache,
            cost_usd=result.cost_usd,
        )
