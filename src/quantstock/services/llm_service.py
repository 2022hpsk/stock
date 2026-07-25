"""大模型服务：按配置装配客户端与任务，暴露用量与缓存状态。

CLI 与界面共用同一实现。

**总开关在这里生效**：``llm.enabled=false`` 时构造出的客户端是 ``off`` 模式，
所有任务照常可调用、只是一律返回"不使用 LLM"。上层代码**不需要**到处写
``if llm_enabled``——那种写法迟早会漏掉一处，而漏掉的那处就是 LR2 的破口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quantstock.config.settings import Settings
from quantstock.infra.errors import LLMError
from quantstock.infra.logging import get_logger
from quantstock.llm.anonymize import Anonymizer
from quantstock.llm.budget import BudgetGuard
from quantstock.llm.cache import LLMCache
from quantstock.llm.client import LLMClient, LLMMode, TaskCall
from quantstock.llm.influence import HARD_INFLUENCE_CAP
from quantstock.llm.prompts import PROMPT_VERSION
from quantstock.llm.protocols import LLMProvider
from quantstock.llm.schemas import PositionJudgement
from quantstock.llm.tasks import (
    ExplainTask,
    IntelClassifyTask,
    MarketJudgeTask,
    PositionJudgeTask,
)

__all__ = ["PROMPT_VERSION", "LLMService", "LLMStatus"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LLMStatus:
    """LLM 模块状态，供 ``quantstock llm status`` 与仪表盘。"""

    enabled: bool
    mode: str
    alpha: float
    prompt_version: str
    cached_entries: int
    cached_cost_usd: float
    daily_spent_usd: float
    monthly_spent_usd: float
    degraded: bool
    degraded_reason: str

    @property
    def message(self) -> str:
        """一行摘要。"""
        if not self.enabled:
            return "LLM 已关闭，系统运行在纯量化模式（功能完整）"
        state = "已降级" if self.degraded else "正常"
        return (
            f"LLM {self.mode} 模式，α={self.alpha:.2f}，提示词 {self.prompt_version}，"
            f"缓存 {self.cached_entries} 条，当日花费 ${self.daily_spent_usd:.4f}（{state}）"
        )


class LLMService:
    """大模型服务。"""

    def __init__(
        self,
        settings: Settings,
        *,
        provider: LLMProvider | None = None,
        in_backtest: bool = False,
    ) -> None:
        """初始化。

        Args:
            settings: 运行期配置。
            provider: 供应商实现。未提供且模式为 ``live`` 时所有调用降级为不使用。
            in_backtest: 是否处于回测中。为 True 时强制 ``replay``（红线 LR3）。

        Raises:
            LLMError: 配置的 α 超过硬上限。
        """
        config = settings.config.llm
        if config.max_influence > HARD_INFLUENCE_CAP:
            msg = f"llm.max_influence={config.max_influence} 超过硬上限 {HARD_INFLUENCE_CAP}"
            raise LLMError(msg)

        self._settings = settings
        self._config = config
        self._in_backtest = in_backtest

        # 回测强制 replay：不是"建议"，是引擎层面的硬约束
        mode: LLMMode = "replay" if in_backtest and config.mode != "off" else config.mode
        if in_backtest and config.mode == "live":
            _log.warning("llm_mode_forced_to_replay", configured=config.mode)

        self._budget = BudgetGuard(
            daily_usd=config.daily_budget_usd,
            monthly_usd=config.monthly_budget_usd,
            usage_dir=settings.var_dir / "llm_usage",
        )
        self._client = LLMClient(
            mode=mode,
            provider=provider,
            cache_dir=settings.var_dir / "llm_cache",
            temperature=config.temperature,
            rate_limit_per_min=config.rate_limit_per_min,
            budget=self._budget,
            in_backtest=in_backtest,
        )

    @property
    def client(self) -> LLMClient:
        """底层客户端。"""
        return self._client

    @property
    def enabled(self) -> bool:
        """LLM 是否会产出结果。"""
        return self._config.enabled and self._client.enabled

    @property
    def alpha(self) -> float:
        """影响系数。"""
        return self._config.max_influence

    def model_for(self, task: str) -> str:
        """取某任务配置的模型 ID。

        Args:
            task: 任务名。

        Returns:
            模型 ID。未配置任务时用 main 档。
        """
        tier = self._config.tasks.get(task)
        mapping = {
            "fast": self._config.model_fast,
            "main": self._config.model_main,
            "deep": self._config.model_deep,
        }
        return mapping[tier.model] if tier else self._config.model_main

    def task_enabled(self, task: str) -> bool:
        """某任务是否启用。

        Args:
            task: 任务名。

        Returns:
            启用则 True。总开关关闭时一律 False。
        """
        if not self.enabled:
            return False
        tier = self._config.tasks.get(task)
        return tier.enabled if tier else False

    def position_judge(self) -> PositionJudgeTask:
        """构造 L2 个股研判任务。

        Returns:
            任务实例。回测中自动开启匿名化与日期脱敏（红线 LR4）。
        """
        return PositionJudgeTask(
            self._client,
            alpha=self.alpha,
            model_id=self.model_for("position_judge"),
            anonymize=self._in_backtest and self._config.anonymize_in_backtest,
            strip_dates=self._config.strip_absolute_dates,
        )

    def market_judge(self) -> MarketJudgeTask:
        """构造 L2 市场研判任务。

        Returns:
            任务实例。
        """
        return MarketJudgeTask(
            self._client,
            alpha=self.alpha,
            model_id=self.model_for("market_judge"),
            strip_dates=self._config.strip_absolute_dates,
        )

    def intel_classify(self) -> IntelClassifyTask:
        """构造 L1 情报分类任务。

        Returns:
            任务实例。
        """
        return IntelClassifyTask(self._client, model_id=self.model_for("intel_classify"))

    def explain(self) -> ExplainTask:
        """构造 L3 解释生成任务。

        Returns:
            任务实例。
        """
        return ExplainTask(self._client, model_id=self.model_for("explain"))

    def anonymizer(self) -> Anonymizer:
        """构造一个匿名化器。

        一个决策周期用一个实例——同一标的在整批材料里必须是同一个代号。

        Returns:
            匿名化器。
        """
        return Anonymizer()

    def status(self) -> LLMStatus:
        """当前状态。

        Returns:
            状态快照。
        """
        cache = self._client.cache
        state = self._budget.state
        return LLMStatus(
            enabled=self.enabled,
            mode=self._client.mode,
            alpha=self.alpha,
            prompt_version=PROMPT_VERSION,
            cached_entries=cache.count() if cache else 0,
            cached_cost_usd=cache.total_cost() if cache else 0.0,
            daily_spent_usd=state.daily_usd,
            monthly_spent_usd=state.monthly_usd,
            degraded=state.degraded,
            degraded_reason=state.degraded_reason,
        )

    def cache_dir(self) -> Path:
        """缓存目录。"""
        return self._settings.var_dir / "llm_cache"

    def cache(self) -> LLMCache | None:
        """缓存实例。"""
        return self._client.cache

    def backfill(self, calls: list[TaskCall]) -> tuple[int, float]:
        """批量预计算 L2 研判缓存。

        回测前跑一次，之后回测走 ``replay`` 零成本、可重复。

        Args:
            calls: 待预计算的调用。

        Returns:
            ``(成功条数, 累计费用)``。

        Raises:
            LLMError: 非 live 模式。
        """
        return self._client.backfill(calls, PositionJudgement)

    def coverage(self, calls: list[TaskCall]) -> float:
        """这批调用的缓存覆盖率。

        回测前用它检查缓存备齐了没有——命中率低意味着大量决策点根本没有
        LLM 输出，此时"含 LLM 的 A/B 回测"实际在比较两条几乎相同的路径。

        Args:
            calls: 调用列表。

        Returns:
            覆盖率 0~1。
        """
        return self._client.coverage(calls)

    def param_hash_parts(self) -> dict[str, str]:
        """进入 ``param_hash`` 的 LLM 相关部分（红线 R6）。

        提示词版本与模型 ID 必须进 param_hash：**改提示词等同于改策略**。

        Returns:
            键值对。
        """
        if not self.enabled:
            return {"llm": "off"}
        return {
            "llm_mode": self._client.mode,
            "llm_alpha": f"{self.alpha:.4f}",
            "llm_prompt_version": PROMPT_VERSION,
            "llm_model_main": self._config.model_main,
            "llm_model_fast": self._config.model_fast,
        }
