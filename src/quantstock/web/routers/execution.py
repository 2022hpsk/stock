"""P3 执行页（docs/09 第三节）。

逐单确认、价格漂移复核、通道选择、提交、撤单。

这是**整个界面里唯一能动真钱的地方**，所以约束最密：

- 未出现在 ``decisions`` 里的意图一律按跳过处理，服务层已保证。界面不必、
  也不应该"帮用户默认接受剩下的"；
- 跳过必须选原因（枚举，非自由输入）。复盘要按原因分组统计人工干预到底是
  帮忙还是添乱（docs/08 D3）；
- 真实通道需要 ``live=true`` **且**确认码（红线 R5）；
- ``var/HALT`` 存在时一律拒绝，由服务层的 ``HaltSwitch`` 兜底——
  界面把按钮变灰只是提升体验，不是安全措施。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from quantstock.infra.money import money
from quantstock.infra.types import Money, Symbol
from quantstock.services.execution_service import ConfirmationDecision, SkipReason
from quantstock.web.deps import AuthDep, WriteDep
from quantstock.web.serializers import serialize_preview

__all__ = ["router"]

router = APIRouter(prefix="/api/execution", tags=["execution"])


class DecisionModel(BaseModel):
    """对单条意图的人工决定。"""

    intent_id: str
    accepted: bool
    adjusted_qty: int | None = Field(default=None, description="改量；空表示按原数量")
    skip_reason: str | None = Field(
        default=None,
        description=(
            "跳过原因，跳过时必填。取值："
            "disagree_logic / cash_reserved / bad_timing / other_info / other"
        ),
    )
    skip_note: str = ""


class ExecuteRequest(BaseModel):
    """执行请求。"""

    trade_date: str
    plan_id: str
    decisions: list[DecisionModel]
    prices: dict[str, str] = Field(
        default_factory=dict, description="标的 → 当前价（字符串，红线 R1）"
    )
    confirmed_by: str = Field(default="ui", description="确认人，写入审计")
    live: bool = Field(default=False, description="使用真实资金通道")
    confirmation_code: str = Field(default="", description="真实通道的二次确认码")


class PreviewRequest(BaseModel):
    """预检请求。"""

    trade_date: str
    plan_id: str
    prices: dict[str, str] = Field(default_factory=dict)


def _prices(raw: dict[str, str]) -> dict[Symbol, Money]:
    """把字符串价格转成 Decimal。

    Args:
        raw: 标的 → 价格字符串。

    Returns:
        标的 → 金额。
    """
    return {Symbol(k): money(v) for k, v in raw.items()}


@router.get("/status")
async def execution_status(state: AuthDep) -> dict[str, Any]:
    """当前通道与桥接目录。"""
    bridge = state.execution.bridge_dir()
    return {
        "broker": state.execution.broker_name,
        "bridge_dir": str(bridge) if bridge else None,
        "readonly": state.readonly,
    }


@router.post("/preview")
async def preview(body: PreviewRequest, state: AuthDep) -> dict[str, Any]:
    """执行前预检：漂移复核、硬闸校验、急停状态。

    Raises:
        HTTPException: 计划不存在。
    """
    plan = state.advisor.store.load(dt.date.fromisoformat(body.trade_date), body.plan_id)  # type: ignore[arg-type]
    if plan is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="计划不存在")
    return serialize_preview(state.execution.preview(plan, _prices(body.prices)))


@router.post("/execute")
async def execute(body: ExecuteRequest, state: WriteDep) -> dict[str, Any]:
    """按逐单决定执行计划。

    Raises:
        HTTPException: 真实通道缺少确认码，或跳过未给原因。
    """
    if body.live and not body.confirmation_code.strip():
        # 真实通道必须二次确认（红线 R5）。这里是**后端**校验——
        # 只在前端拦是防误点，不是防绕过
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="真实资金通道必须提供二次确认码",
        )

    decisions: list[ConfirmationDecision] = []
    for d in body.decisions:
        if not d.accepted and not d.skip_reason:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"意图 {d.intent_id} 被跳过但未选择原因",
            )
        decisions.append(
            ConfirmationDecision(
                intent_id=d.intent_id,
                accepted=d.accepted,
                adjusted_qty=d.adjusted_qty,
                skip_reason=SkipReason(d.skip_reason) if d.skip_reason else None,
                skip_note=d.skip_note,
            )
        )

    plan = state.advisor.store.load(dt.date.fromisoformat(body.trade_date), body.plan_id)  # type: ignore[arg-type]
    report = state.execution.execute(
        plan,
        decisions=decisions,
        current_prices=_prices(body.prices),
        confirmed_by=body.confirmed_by,
        live=body.live,
    )

    payload = {
        "plan_id": str(report.plan_id),
        "trade_date": report.trade_date.isoformat(),
        "executed_at": report.executed_at.isoformat(),
        "broker": report.broker,
        "aborted": report.aborted,
        "abort_reason": report.abort_reason,
        "confirmed_by": report.confirmed_by,
        "submitted": len(report.submitted),
        "skipped": len(report.skipped),
        "skip_reasons": report.skip_reasons(),
        "total_amount": str(report.total_amount),
        "manual_checklist": list(report.manual_checklist),
        "orders": [
            {
                "symbol": str(o.symbol),
                "side": str(getattr(o.side, "value", o.side)),
                "qty": o.qty,
                "price": str(o.price),
                "amount": str(o.amount),
                "status": str(getattr(o.status, "value", o.status)),
                "note": getattr(o, "note", ""),
            }
            for o in report.orders
        ],
        "fills": [
            {
                "fill_id": f.fill_id,
                "order_id": f.order_id,
                "symbol": str(f.symbol),
                "side": str(getattr(f.side, "value", f.side)),
                "qty": f.qty,
                "price": str(f.price),
                "amount": str(f.amount),
                "fee": str(f.fee),
                "filled_at": f.filled_at.isoformat(),
            }
            for f in report.fills
        ],
    }
    state.events.publish(
        "orders",
        "executed",
        **{k: payload[k] for k in ("plan_id", "submitted", "skipped", "aborted")},
    )
    return payload


@router.post("/cancel-all")
async def cancel_all(state: WriteDep) -> dict[str, Any]:
    """撤销所有未成交委托。"""
    count = state.execution.cancel_all()
    state.events.publish("orders", "cancelled", count=count)
    return {"cancelled": count}


@router.get("/skip-reasons")
async def skip_reasons() -> dict[str, Any]:
    """可选的跳过原因。

    界面必须用这个列表渲染下拉框，**不能让用户自由输入**——
    自由文本没法分组统计，人工干预价值分析就成了一堆读不出结论的字符串。
    """
    labels = {
        "disagree_logic": "不认同策略逻辑",
        "cash_reserved": "资金另有安排",
        "bad_timing": "认为时机不对",
        "other_info": "已有其他渠道信息",
        "other": "其他",
    }
    return {
        "reasons": [{"value": r.value, "label": labels.get(r.value, r.value)} for r in SkipReason]
    }
