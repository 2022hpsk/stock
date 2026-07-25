"""L3 解释生成任务。

规范见 docs/10-大模型集成规格.md 第二节。

**对决策的影响为零**——输入是已经确定的结构化结论，输出只是行文。
因此这个任务没有任何数值出口，失败时直接回退到规则拼接的文本，
用户看到的内容会朴素一些，但信息一条不少。

规则回退不是权宜之计，而是**基线**：LLM 行文只是让同样的内容更易读。
如果去掉 LLM 之后解释就变得看不懂，说明结构化结论本身就不完整。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from quantstock.llm.client import LLMClient, TaskCall
from quantstock.llm.prompts import EXPLAIN_SYSTEM, PROMPT_VERSION, render_materials
from quantstock.llm.schemas import ExplanationOutput

__all__ = ["ExplainOutcome", "ExplainTask"]

TASK_ID = "explain"


@dataclass(frozen=True, slots=True)
class ExplainOutcome:
    """一次解释生成的结果。"""

    verdict: str
    narrative: tuple[str, ...]
    llm_generated: bool
    """是否由 LLM 行文。界面据此打 🤖 标（红线 LR8）。"""
    reason: str = ""
    cache_key: str = ""
    cost_usd: float = 0.0

    def as_lines(self) -> list[str]:
        """渲染成文本行。

        Returns:
            行列表，LLM 生成时首行带标记。
        """
        head = f"{'🤖 ' if self.llm_generated else ''}{self.verdict}"
        return [head, *self.narrative]


class ExplainTask:
    """解释生成任务。"""

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

    def run(
        self,
        *,
        verdict: str,
        pillars: dict[str, Sequence[str]],
        as_of: str,
    ) -> ExplainOutcome:
        """把结构化结论组织成中文说明。

        Args:
            verdict: 已经确定的一句话结论。**LLM 不得改动它**。
            pillars: 支柱名 → 要点列表。
            as_of: 决策时点。

        Returns:
            解释结果。失败时回退为规则拼接。
        """
        materials = {name: "；".join(points) for name, points in pillars.items() if points}
        fallback = _rule_based(pillars)

        if not self._client.enabled:
            return ExplainOutcome(
                verdict=verdict,
                narrative=fallback,
                llm_generated=False,
                reason="LLM 已关闭，使用规则行文",
            )

        call = TaskCall(
            task_id=TASK_ID,
            prompt_version=self._prompt_version,
            model_id=self._model_id,
            system=EXPLAIN_SYSTEM,
            user=(
                f"决策时点：{as_of}\n"
                f"已确定的结论（不得改动）：{verdict}\n\n"
                f"{render_materials(materials)}"
            ),
            payload={"verdict": verdict, "as_of": as_of, "materials": materials},
        )
        result = self._client.complete(call, ExplanationOutput)

        if result.output is None:
            return ExplainOutcome(
                verdict=verdict,
                narrative=fallback,
                llm_generated=False,
                reason=result.reason,
                cache_key=result.cache_key,
                cost_usd=result.cost_usd,
            )

        output = result.output
        assert isinstance(output, ExplanationOutput)  # noqa: S101 - complete 保证契约

        # 结论用我们自己的，不用模型回填的那份：模型可能"顺手润色"了它，
        # 而结论必须与结构化结果逐字一致
        return ExplainOutcome(
            verdict=verdict,
            narrative=tuple(output.narrative),
            llm_generated=True,
            cache_key=result.cache_key,
            cost_usd=result.cost_usd,
        )


def _rule_based(pillars: dict[str, Sequence[str]]) -> tuple[str, ...]:
    """规则拼接的解释文本。

    结论不在这里重复——调用方已经把它放在 ``ExplanationOutcome.verdict`` 里。

    Args:
        pillars: 支柱要点。

    Returns:
        文本行。
    """
    lines: list[str] = []
    for name, points in pillars.items():
        if not points:
            continue
        lines.append(f"{name}：")
        lines.extend(f"  · {p}" for p in points)
    return tuple(lines)
