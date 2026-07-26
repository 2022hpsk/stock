"""P16 大模型页（docs/09 第三节）。

用量与预算、缓存覆盖、提示词版本、各任务启停。

这个页面存在的意义是**让 LLM 的影响力可见且可关**（红线 LR2）。
所以它显示的都是"约束"而不是"能力"：α 的当前值与硬上限、当日/当月已花多少、
缓存覆盖到哪天、降级了没有。

**没有下单相关的任何接口**。LLM 的输出不得直接成为下单方向、数量、价格
（红线 LR1），界面上也就不该有从这个页面通向执行的路径。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from quantstock.web.deps import AuthDep

__all__ = ["router"]

router = APIRouter(prefix="/api/llm", tags=["llm"])

TASKS = ("position_judge", "market_judge", "intel_classify", "explain")
"""界面上要逐个显示启停与模型的任务。"""


@router.get("/status")
async def llm_status(state: AuthDep) -> dict[str, Any]:
    """当前状态、用量与预算。"""
    s = state.llm.status()
    return {
        "enabled": s.enabled,
        "mode": s.mode,
        "alpha": s.alpha,
        "prompt_version": s.prompt_version,
        "cached_entries": s.cached_entries,
        "cached_cost_usd": s.cached_cost_usd,
        "daily_spent_usd": s.daily_spent_usd,
        "monthly_spent_usd": s.monthly_spent_usd,
        "degraded": s.degraded,
        "degraded_reason": s.degraded_reason,
        "message": s.message,
        "cache_dir": str(state.llm.cache_dir()),
        "tasks": [
            {
                "name": name,
                "enabled": state.llm.task_enabled(name),
                "model": state.llm.model_for(name),
            }
            for name in TASKS
        ],
        # 改提示词等同于改策略，所以提示词版本与模型 ID 都进 param_hash（红线 R6）。
        # 审计页靠它区分"同样的参数为什么给出了不同建议"
        "param_hash_parts": state.llm.param_hash_parts(),
    }
