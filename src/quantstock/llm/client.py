"""带缓存、限流、预算与三模式的 LLM 客户端。

规范见 docs/10-大模型集成规格.md 4.2。

三种模式：

===========  ================================================  ==================
模式         行为                                              用于
===========  ================================================  ==================
``off``      完全不调用，走纯规则/纯量化路径                   基线对照、故障降级
``live``     实际调用 API，结果写缓存                          实盘、``llm backfill``
``replay``   **只读缓存**；未命中即视为"该条无 LLM 输出"       **回测唯一允许**
===========  ================================================  ==================

**回测强制 ``replay``**：由回测引擎在启动时设置。任何代码路径试图在回测中
发起真实调用都会抛 ``LLMLiveCallInBacktestError``（红线 LR3）。

这里的关键设计是 ``complete()`` **永远不抛业务异常**（回测越界除外）：
API 挂了、解析失败、预算超了，一律返回 ``None`` 并记录原因。
上层拿到 None 就走纯量化路径。把 LLM 故障做成异常会让它有能力中断
整条决策链——那正好违背了"LLM 是增强项"这个前提。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel

from quantstock.infra.clock import now
from quantstock.infra.errors import (
    LLMError,
    LLMLiveCallInBacktestError,
    LLMOutputInvalidError,
)
from quantstock.infra.logging import get_logger
from quantstock.infra.retry import RateLimiter
from quantstock.llm.budget import BudgetGuard, UsageRecord, estimate_cost
from quantstock.llm.cache import LLMCache, compute_cache_key
from quantstock.llm.protocols import (
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    LLMProvider,
)
from quantstock.llm.validate import parse_output

__all__ = ["LLMClient", "LLMMode", "TaskCall", "TaskResult"]

_log = get_logger(__name__)

LLMMode = Literal["off", "live", "replay"]

_T = TypeVar("_T", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class TaskCall:
    """一次任务调用的描述。"""

    task_id: str
    prompt_version: str
    model_id: str
    system: str
    user: str
    payload: dict[str, Any]
    """进入缓存键的规范化输入。**必须只含决定输出的内容**——
    把时间戳之类每次都变的东西放进来会让缓存永不命中。"""
    max_tokens: int = 2048

    def to_request(self, *, temperature: float) -> CompletionRequest:
        """转成供应商请求。

        Args:
            temperature: 采样温度。

        Returns:
            补全请求。
        """
        return CompletionRequest(
            model=self.model_id,
            system=self.system,
            messages=(ChatMessage(role="user", content=self.user),),
            temperature=temperature,
            max_tokens=self.max_tokens,
        )


@dataclass(frozen=True, slots=True)
class TaskResult:
    """一次任务调用的结果。"""

    output: BaseModel | None
    """校验通过的结构化输出；None 表示本次不使用 LLM。"""
    cache_key: str
    from_cache: bool = False
    reason: str = ""
    """未产出结果时的原因，会进入日报与界面。"""
    cost_usd: float = 0.0
    model_id: str = ""
    prompt_version: str = ""

    @property
    def used_llm(self) -> bool:
        """本次是否实际用上了 LLM。"""
        return self.output is not None


class LLMClient:
    """LLM 客户端。"""

    def __init__(
        self,
        *,
        mode: LLMMode = "off",
        provider: LLMProvider | None = None,
        cache_dir: Path | None = None,
        temperature: float = 0.0,
        rate_limit_per_min: int = 20,
        budget: BudgetGuard | None = None,
        in_backtest: bool = False,
    ) -> None:
        """初始化。

        Args:
            mode: 运行模式。
            provider: 供应商实现。``live`` 模式下必需。
            cache_dir: 缓存目录。
            temperature: 采样温度。默认 0 以最大化确定性。
            rate_limit_per_min: 每分钟调用上限。
            budget: 预算闸门。
            in_backtest: 是否处于回测中。为 True 时任何实时调用都会抛异常。

        Raises:
            LLMLiveCallInBacktestError: 回测中却配置了 ``live`` 模式。
        """
        if in_backtest and mode == "live":
            msg = "回测必须使用 replay 模式，实时调用会引入不可复现性与未来信息（红线 LR3）"
            raise LLMLiveCallInBacktestError(msg, mode=mode)

        self._mode: LLMMode = mode
        self._provider = provider
        self._cache = LLMCache(cache_dir) if cache_dir else None
        self._temperature = temperature
        self._limiter = RateLimiter(rate_per_min=rate_limit_per_min)
        self._budget = budget
        self._in_backtest = in_backtest

    @property
    def mode(self) -> LLMMode:
        """当前模式。"""
        return self._mode

    @property
    def cache(self) -> LLMCache | None:
        """缓存实例。"""
        return self._cache

    @property
    def enabled(self) -> bool:
        """是否会产出 LLM 结果。"""
        return self._mode != "off"

    def cache_key_for(self, call: TaskCall) -> str:
        """计算某次调用的缓存键。

        Args:
            call: 任务调用。

        Returns:
            缓存键。
        """
        return compute_cache_key(
            task_id=call.task_id,
            prompt_version=call.prompt_version,
            model_id=call.model_id,
            temperature=self._temperature,
            payload=call.payload,
        )

    def complete(self, call: TaskCall, schema: type[_T]) -> TaskResult:
        """执行一次任务调用。

        **不抛业务异常**：任何失败都返回 ``output=None`` 与原因，
        上层据此走纯量化路径。

        Args:
            call: 任务调用。
            schema: 期望的输出契约。

        Returns:
            任务结果。

        Raises:
            LLMLiveCallInBacktestError: 回测中试图发起实时调用。
        """
        cache_key = self.cache_key_for(call)

        if self._mode == "off":
            return TaskResult(output=None, cache_key=cache_key, reason="LLM 已关闭")

        if (cached := self._read_cache(call, cache_key, schema)) is not None:
            return cached

        if self._mode == "replay":
            # 回放模式下未命中不是错误：它意味着这个决策点没有预计算过，
            # 按"该条无 LLM 输出"处理，回测照常继续
            _log.info("llm_replay_miss", task_id=call.task_id, cache_key=cache_key[:12])
            return TaskResult(
                output=None,
                cache_key=cache_key,
                reason="回放模式下缓存未命中，本次不使用 LLM",
            )

        return self._call_live(call, cache_key, schema)

    def _read_cache(self, call: TaskCall, cache_key: str, schema: type[_T]) -> TaskResult | None:
        """尝试读缓存并解析。

        Args:
            call: 任务调用。
            cache_key: 缓存键。
            schema: 输出契约。

        Returns:
            任务结果；未命中时 None。
        """
        if self._cache is None:
            return None
        entry = self._cache.get(call.task_id, cache_key)
        if entry is None:
            return None

        response = CompletionResponse.from_payload(entry.response)
        try:
            output = parse_output(response.text, schema)
        except LLMOutputInvalidError as exc:
            # 缓存里存着一份当时就没通过校验的响应。按不使用处理，
            # 不要在这里重新调用——回放模式的全部价值就是不发生实时调用
            _log.warning("llm_cached_output_invalid", task_id=call.task_id, error=str(exc))
            return TaskResult(
                output=None,
                cache_key=cache_key,
                from_cache=True,
                reason=f"缓存中的输出未通过校验：{exc}",
            )

        return TaskResult(
            output=output,
            cache_key=cache_key,
            from_cache=True,
            cost_usd=0.0,
            model_id=entry.model_id,
            prompt_version=entry.prompt_version,
        )

    def _call_live(self, call: TaskCall, cache_key: str, schema: type[_T]) -> TaskResult:
        """实际调用 API。

        Args:
            call: 任务调用。
            cache_key: 缓存键。
            schema: 输出契约。

        Returns:
            任务结果。

        Raises:
            LLMLiveCallInBacktestError: 回测中走到了这里。
        """
        if self._in_backtest:
            msg = "回测中禁止发起实时 LLM 调用（红线 LR3）"
            raise LLMLiveCallInBacktestError(msg, task_id=call.task_id)

        if self._provider is None:
            return TaskResult(output=None, cache_key=cache_key, reason="未配置 LLM 供应商")

        if self._budget is not None and not self._budget.can_spend():
            return TaskResult(
                output=None,
                cache_key=cache_key,
                reason=f"超出费用预算，已降级为纯量化：{self._budget.state.degraded_reason}",
            )

        self._limiter.acquire()
        started = time.monotonic()
        try:
            response = self._provider.complete(call.to_request(temperature=self._temperature))
        except Exception as exc:
            _log.warning("llm_call_failed", task_id=call.task_id, error=str(exc))
            return TaskResult(
                output=None, cache_key=cache_key, reason=f"调用失败：{type(exc).__name__}"
            )

        latency = int((time.monotonic() - started) * 1000)
        cost = estimate_cost(
            call.model_id,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
        if self._budget is not None:
            self._budget.record(
                UsageRecord(
                    task_id=call.task_id,
                    model_id=call.model_id,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=cost,
                    at=now(),
                )
            )

        # 无论是否通过校验都写缓存：一份没通过校验的响应同样是"当时发生了什么"
        # 的证据，回放时按不使用处理，但排查提示词问题时它是唯一线索
        if self._cache is not None:
            self._cache.put(
                self._cache.make_entry(
                    cache_key=cache_key,
                    task_id=call.task_id,
                    model_id=call.model_id,
                    prompt_version=call.prompt_version,
                    temperature=self._temperature,
                    request=dict(call.to_request(temperature=self._temperature).payload()),
                    response=dict(response.payload()),
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    cost_usd=cost,
                    latency_ms=latency,
                )
            )

        try:
            output = parse_output(response.text, schema)
        except LLMOutputInvalidError as exc:
            _log.warning("llm_output_invalid", task_id=call.task_id, error=str(exc))
            return TaskResult(
                output=None,
                cache_key=cache_key,
                reason=f"输出未通过校验，本次不使用 LLM：{exc}",
                cost_usd=cost,
            )

        return TaskResult(
            output=output,
            cache_key=cache_key,
            cost_usd=cost,
            model_id=call.model_id,
            prompt_version=call.prompt_version,
        )

    def backfill(self, calls: list[TaskCall], schema: type[_T]) -> tuple[int, float]:
        """批量预计算并写入缓存。

        回测前跑一次，之后回测走 ``replay`` 零成本、可重复。

        Args:
            calls: 待预计算的调用。
            schema: 输出契约。

        Returns:
            ``(成功条数, 累计费用)``。

        Raises:
            LLMError: 非 ``live`` 模式下调用。
        """
        if self._mode != "live":
            msg = "backfill 必须在 live 模式下运行"
            raise LLMError(msg, mode=self._mode)

        done = 0
        cost = 0.0
        for call in calls:
            key = self.cache_key_for(call)
            if self._cache is not None and self._cache.has(call.task_id, key):
                continue  # 断点续传：已算过的跳过
            result = self.complete(call, schema)
            cost += result.cost_usd
            if result.used_llm:
                done += 1
        _log.info("llm_backfill_done", calls=len(calls), succeeded=done, cost_usd=round(cost, 4))
        return done, cost

    def coverage(self, calls: list[TaskCall]) -> float:
        """这批调用的缓存覆盖率。

        Args:
            calls: 调用列表。

        Returns:
            覆盖率 0~1。无缓存目录时为 0。
        """
        if self._cache is None or not calls:
            return 0.0 if self._cache is None else 1.0
        hits = sum(1 for c in calls if self._cache.has(c.task_id, self.cache_key_for(c)))
        return hits / len(calls)
