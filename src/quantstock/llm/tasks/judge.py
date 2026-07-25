"""L2 研判归纳任务。

规范见 docs/10-大模型集成规格.md 第六节。

L2 是 LLM 唯一能影响数字的地方，因此这里的每一步都在收窄它的影响面：

1. 材料先经匿名化与日期脱敏（回测中）；
2. 输出经 pydantic 契约校验（越界即作废）；
3. 输出经反幻觉校验（引用不存在的材料即作废）；
4. 通过的调节值再经 α 限幅才落到打分上。

任一步失败都返回 ``adjustment=None``，上层退化为纯量化——
**这条降级路径必须始终可用**，它是"LLM 挂了系统照常出建议"的实现。
"""

from __future__ import annotations

from dataclasses import dataclass

from quantstock.infra.logging import get_logger
from quantstock.infra.types import Symbol
from quantstock.llm.anonymize import Anonymizer, strip_absolute_dates
from quantstock.llm.client import LLMClient, TaskCall
from quantstock.llm.influence import InfluenceResult, apply_conviction, apply_exposure
from quantstock.llm.prompts import (
    MARKET_JUDGE_SYSTEM,
    POSITION_JUDGE_SYSTEM,
    PROMPT_VERSION,
    render_materials,
)
from quantstock.llm.schemas import MarketJudgement, PositionJudgement
from quantstock.llm.validate import validate_judgement

__all__ = ["JudgeOutcome", "MarketJudgeTask", "PositionJudgeTask"]

_log = get_logger(__name__)

POSITION_TASK_ID = "position_judge"
MARKET_TASK_ID = "market_judge"


@dataclass(frozen=True, slots=True)
class JudgeOutcome:
    """一次研判的完整结果。"""

    judgement: PositionJudgement | MarketJudgement | None
    influence: InfluenceResult
    reason: str = ""
    cache_key: str = ""
    from_cache: bool = False
    model_id: str = ""
    prompt_version: str = ""
    cost_usd: float = 0.0

    @property
    def used_llm(self) -> bool:
        """本次是否实际用上了 LLM。日报据此决定是否打 🤖 标（红线 LR8）。"""
        return self.judgement is not None and self.influence.applied

    def falsification(self) -> tuple[str, ...]:
        """研判给出的证伪条件，直接进四支柱④。

        Returns:
            证伪条件；无研判时为空。
        """
        if isinstance(self.judgement, PositionJudgement):
            return tuple(self.judgement.falsification)
        return ()

    def conflicts(self) -> tuple[str, ...]:
        """证据间的矛盾点。

        Returns:
            矛盾点；无研判时为空。
        """
        if isinstance(self.judgement, PositionJudgement):
            return tuple(self.judgement.conflicts)
        return ()


class PositionJudgeTask:
    """对单个标的做 L2 研判。"""

    def __init__(
        self,
        client: LLMClient,
        *,
        alpha: float,
        model_id: str,
        anonymize: bool = False,
        strip_dates: bool = False,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        """初始化。

        Args:
            client: LLM 客户端。
            alpha: 影响系数。
            model_id: 模型 ID。
            anonymize: 是否匿名化标的。回测应开启（红线 LR4）。
            strip_dates: 是否脱敏绝对日期。回测应开启。
            prompt_version: 提示词版本。
        """
        self._client = client
        self._alpha = alpha
        self._model_id = model_id
        self._anonymize = anonymize
        self._strip_dates = strip_dates
        self._prompt_version = prompt_version

    def run(
        self,
        symbol: Symbol,
        *,
        base_score: float,
        materials: dict[str, str],
        as_of: str,
        anonymizer: Anonymizer | None = None,
    ) -> JudgeOutcome:
        """执行一次研判。

        Args:
            symbol: 标的。
            base_score: 纯量化打分。
            materials: 材料 ID → 内容。
            as_of: 决策时点的字符串表述。
            anonymizer: 匿名化器。开启匿名化时必需，由调用方在一个决策周期内复用。

        Returns:
            研判结果。
        """
        prepared, symbol_ref = self._prepare(symbol, materials, anonymizer)
        call = TaskCall(
            task_id=POSITION_TASK_ID,
            prompt_version=self._prompt_version,
            model_id=self._model_id,
            system=POSITION_JUDGE_SYSTEM,
            user=(
                f"标的标识：{symbol_ref}\n"
                f"决策时点：{as_of}\n"
                f"量化基准分：{base_score:.4f}\n\n"
                f"{render_materials(prepared)}"
            ),
            payload={"symbol_ref": symbol_ref, "as_of": as_of, "materials": prepared},
        )

        result = self._client.complete(call, PositionJudgement)
        if result.output is None:
            return JudgeOutcome(
                judgement=None,
                influence=apply_conviction(base_score, None, alpha=self._alpha, note=result.reason),
                reason=result.reason,
                cache_key=result.cache_key,
                from_cache=result.from_cache,
                cost_usd=result.cost_usd,
            )

        judgement = result.output
        assert isinstance(judgement, PositionJudgement)  # noqa: S101 - 契约由 complete 保证

        if not (outcome := validate_judgement(judgement, prepared)):
            # 反幻觉校验不通过 → 整个输出作废，退化为纯量化
            _log.warning("position_judge_rejected", symbol=str(symbol), reason=outcome.reason)
            return JudgeOutcome(
                judgement=None,
                influence=apply_conviction(
                    base_score, None, alpha=self._alpha, note=outcome.reason
                ),
                reason=outcome.reason,
                cache_key=result.cache_key,
                from_cache=result.from_cache,
                cost_usd=result.cost_usd,
            )

        return JudgeOutcome(
            judgement=judgement,
            influence=apply_conviction(
                base_score, judgement.conviction_adjustment, alpha=self._alpha
            ),
            cache_key=result.cache_key,
            from_cache=result.from_cache,
            model_id=result.model_id or self._model_id,
            prompt_version=result.prompt_version or self._prompt_version,
            cost_usd=result.cost_usd,
        )

    def _prepare(
        self, symbol: Symbol, materials: dict[str, str], anonymizer: Anonymizer | None
    ) -> tuple[dict[str, str], str]:
        """按配置做匿名化与日期脱敏。

        Args:
            symbol: 标的。
            materials: 原始材料。
            anonymizer: 匿名化器。

        Returns:
            ``(处理后的材料, 标的标识)``。
        """
        prepared = dict(materials)
        if self._strip_dates:
            prepared = {k: strip_absolute_dates(v) for k, v in prepared.items()}

        if not self._anonymize:
            return prepared, str(symbol)

        mapper = anonymizer or Anonymizer()
        label = mapper.label_of(symbol)
        prepared = {k: mapper.anonymize(v).text for k, v in prepared.items()}
        return prepared, label


class MarketJudgeTask:
    """对市场环境做 L2 研判。"""

    def __init__(
        self,
        client: LLMClient,
        *,
        alpha: float,
        model_id: str,
        strip_dates: bool = False,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        """初始化。

        Args:
            client: LLM 客户端。
            alpha: 影响系数。
            model_id: 模型 ID。
            strip_dates: 是否脱敏绝对日期。
            prompt_version: 提示词版本。
        """
        self._client = client
        self._alpha = alpha
        self._model_id = model_id
        self._strip_dates = strip_dates
        self._prompt_version = prompt_version

    def run(self, *, base_exposure: float, materials: dict[str, str], as_of: str) -> JudgeOutcome:
        """执行一次市场研判。

        Args:
            base_exposure: 纯量化的总仓位中枢。
            materials: 材料。
            as_of: 决策时点。

        Returns:
            研判结果。仓位调节**只能减不能加**。
        """
        prepared = (
            {k: strip_absolute_dates(v) for k, v in materials.items()}
            if self._strip_dates
            else dict(materials)
        )
        call = TaskCall(
            task_id=MARKET_TASK_ID,
            prompt_version=self._prompt_version,
            model_id=self._model_id,
            system=MARKET_JUDGE_SYSTEM,
            user=(
                f"决策时点：{as_of}\n"
                f"量化总仓位中枢：{base_exposure:.4f}\n\n"
                f"{render_materials(prepared)}"
            ),
            payload={"as_of": as_of, "materials": prepared},
        )

        result = self._client.complete(call, MarketJudgement)
        if result.output is None:
            return JudgeOutcome(
                judgement=None,
                influence=apply_exposure(
                    base_exposure, None, alpha=self._alpha, note=result.reason
                ),
                reason=result.reason,
                cache_key=result.cache_key,
                from_cache=result.from_cache,
                cost_usd=result.cost_usd,
            )

        judgement = result.output
        assert isinstance(judgement, MarketJudgement)  # noqa: S101 - 契约由 complete 保证

        if not (outcome := validate_judgement(judgement, prepared)):
            _log.warning("market_judge_rejected", reason=outcome.reason)
            return JudgeOutcome(
                judgement=None,
                influence=apply_exposure(
                    base_exposure, None, alpha=self._alpha, note=outcome.reason
                ),
                reason=outcome.reason,
                cache_key=result.cache_key,
                from_cache=result.from_cache,
                cost_usd=result.cost_usd,
            )

        return JudgeOutcome(
            judgement=judgement,
            influence=apply_exposure(
                base_exposure, judgement.exposure_adjustment, alpha=self._alpha
            ),
            cache_key=result.cache_key,
            from_cache=result.from_cache,
            model_id=result.model_id or self._model_id,
            prompt_version=result.prompt_version or self._prompt_version,
            cost_usd=result.cost_usd,
        )
