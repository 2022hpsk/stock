"""P2 每日建议页（docs/09 第三节）。

生成建议、查看历史计划、展开四支柱解释。

界面上最要紧的一条：**被否决的候选与解释不完整被剔除的条目必须一起返回**。
只显示"建议买入这 5 只"的界面会让人误以为系统只看好这 5 只，
而实际上可能有 20 只被风控挡掉了——那 20 条才是了解系统在想什么的关键。
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from quantstock.infra.money import money
from quantstock.infra.types import Symbol
from quantstock.web.deps import AuthDep, WriteDep
from quantstock.web.serializers import serialize_plan

__all__ = ["router"]

router = APIRouter(prefix="/api/advisor", tags=["advisor"])


class AdviseRequest(BaseModel):
    """生成建议请求。"""

    as_of: str | None = Field(default=None, description="决策日；空表示今日")
    tier: str | None = Field(default=None, description="候选池档位；空表示用配置值")
    symbols: list[str] = Field(default_factory=list, description="显式候选池")
    total_value: str | None = Field(default=None, description="账户总资产；空表示由账本推导")
    cash: str | None = Field(default=None, description="可用资金")
    exposure: str | None = Field(default=None, description="总仓位中枢 0~1；空表示由策略给出")
    save: bool = Field(default=True, description="是否落盘")


@router.post("/advise")
async def advise(body: AdviseRequest, state: WriteDep) -> dict[str, Any]:
    """生成今日建议。

    金额参数走**字符串**而不是 float：JSON 的数字是 IEEE 754 双精度，
    ``100000.1`` 传过来已经不精确了，再拿去算仓位会一路带着误差（红线 R1）。
    """
    state.events.publish("tasks", "progress", task="advisor.advise", stage="scoring")
    result = state.advisor.advise(
        as_of=dt.date.fromisoformat(body.as_of) if body.as_of else None,
        universe=[Symbol(s) for s in body.symbols] or None,
        tier=body.tier,
        total_value=money(body.total_value) if body.total_value else None,
        cash=money(body.cash) if body.cash else None,
        exposure=Decimal(body.exposure) if body.exposure else None,
        save=body.save,
    )

    payload = {
        "plan": serialize_plan(result.plan),
        "saved_to": result.saved_to,
        "summary": result.summary,
        "llm_used": result.llm_used,
        # 打分的前后对照是 LLM 影响力的唯一可视化依据（红线 LR2 有界影响）：
        # 界面要能显示 0.62 → 0.66 这样的具体调整量，而不是笼统一句"AI 参与了"
        "base_scores": {str(k): v for k, v in result.base_scores.items()},
        "final_scores": {str(k): v for k, v in result.final_scores.items()},
        "llm_notes": {str(k): v for k, v in result.llm_notes.items()},
        "skipped": [{"symbol": str(s), "reason": r} for s, r in result.skipped],
    }
    state.events.publish(
        "tasks",
        "done",
        task="advisor.advise",
        plan_id=str(result.plan.plan_id),
        intents=len(result.plan.intents),
    )
    return payload


@router.get("/dates")
async def plan_dates(state: AuthDep) -> dict[str, Any]:
    """有计划的日期列表。"""
    return {"dates": [d.isoformat() for d in state.advisor.store.list_dates()]}


@router.get("/plan/{trade_date}")
async def latest_plan(trade_date: str, state: AuthDep) -> dict[str, Any]:
    """取某日最新的交易计划。

    Raises:
        HTTPException: 该日没有计划。
    """
    plan = state.advisor.store.latest(dt.date.fromisoformat(trade_date))
    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"{trade_date} 没有已保存的交易计划"
        )
    return serialize_plan(plan)


@router.get("/plan/{trade_date}/{plan_id}")
async def plan_detail(trade_date: str, plan_id: str, state: AuthDep) -> dict[str, Any]:
    """按 ID 取交易计划。审计页复现当天建议时用。"""
    plan = state.advisor.store.load(dt.date.fromisoformat(trade_date), plan_id)  # type: ignore[arg-type]
    return serialize_plan(plan)
