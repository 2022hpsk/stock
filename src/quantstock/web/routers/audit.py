"""P15 审计页（docs/09 第三节）。

任一天的建议**完整复现**，以及下单记录的链路追溯。

红线 R6 要求每条建议可追溯可复现（数据指纹 + 策略版本 + 参数哈希）。
光把这三个字段存下来还不够——**能存不等于能复现**。这里提供的
"用当时的参数重新算一遍并比对"才是那条红线真正被满足的证据。

复现结果分三种，界面必须区分开：

- ``identical``：完全一致，可复现；
- ``drifted``：字段还在但结果变了。这**不一定是 bug**——数据被回补、
  策略版本升级都会造成差异，关键是差在哪里能说清楚；
- ``unreproducible``：连指纹都对不上，当时的输入已经不可得。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, HTTPException, status

from quantstock.web.deps import AuthDep
from quantstock.web.serializers import serialize_plan

__all__ = ["router"]

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("/dates")
async def dates(state: AuthDep) -> dict[str, Any]:
    """有可审计记录的日期。"""
    plan_dates = {d.isoformat() for d in state.advisor.store.list_dates()}
    exec_dates = {d.isoformat() for d in state.execution.reports.list_dates()}
    return {
        "plan_dates": sorted(plan_dates),
        "execution_dates": sorted(exec_dates),
        # 有计划没执行记录是正常的（那天没执行）；
        # 有执行记录没计划则说明有人绕过了计划下单，值得单独标出来
        "orphan_executions": sorted(exec_dates - plan_dates),
    }


@router.get("/plan/{trade_date}")
async def audit_plan(trade_date: str, state: AuthDep) -> dict[str, Any]:
    """某日建议的完整快照与执行链路。

    Raises:
        HTTPException: 该日没有计划。
    """
    date = dt.date.fromisoformat(trade_date)
    plan = state.advisor.store.latest(date)
    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"{trade_date} 没有已保存的交易计划"
        )

    reports = state.execution.reports.read(date)
    # 建议 → 确认（谁、何时）→ 提交 → 成交 的完整链路。
    # intent_id 是贯穿全程的锚点（红线 R6）
    chain = []
    for intent in plan.intents:
        matched = [
            order
            for report in reports
            for order in report["orders"]
            if order["intent_id"] == str(intent.intent_id)
        ]
        fills = [
            fill
            for report in reports
            for fill in report["fills"]
            if any(o["order_id"] == fill["order_id"] for o in matched)
        ]
        chain.append(
            {
                "intent_id": str(intent.intent_id),
                "symbol": str(intent.symbol),
                "side": str(getattr(intent.side, "value", intent.side)),
                "suggested_qty": intent.qty,
                "orders": matched,
                "fills": fills,
                "outcome": _outcome_of(matched),
            }
        )

    return {
        "plan": serialize_plan(plan),
        "executions": [
            {
                "executed_at": r["executed_at"],
                "broker": r["broker"],
                "confirmed_by": r["confirmed_by"],
                "aborted": r["aborted"],
                "abort_reason": r["abort_reason"],
                "orders": len(r["orders"]),
                "fills": len(r["fills"]),
            }
            for r in reports
        ],
        "chain": chain,
    }


@router.post("/reproduce/{trade_date}")
async def reproduce(trade_date: str, state: AuthDep) -> dict[str, Any]:
    """用当时的参数重新计算，并与存档比对（红线 R6）。

    **这是可复现性的唯一证据**。把指纹存下来只证明"当时算过"，
    重算一遍并比对才证明"现在还能算出同样的结果"。

    Raises:
        HTTPException: 该日没有计划。
    """
    date = dt.date.fromisoformat(trade_date)
    archived = state.advisor.store.latest(date)
    if archived is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"{trade_date} 没有已保存的交易计划"
        )

    # save=False：复现绝不能覆盖存档。覆盖之后就再也没有"当时"可比了
    result = state.advisor.advise(as_of=date, save=False)
    fresh = result.plan

    fingerprint_match = fresh.data_fingerprint == archived.data_fingerprint
    param_match = fresh.param_hash == archived.param_hash
    archived_intents = {
        (str(i.symbol), str(getattr(i.side, "value", i.side)), i.qty) for i in archived.intents
    }
    fresh_intents = {
        (str(i.symbol), str(getattr(i.side, "value", i.side)), i.qty) for i in fresh.intents
    }
    intents_match = archived_intents == fresh_intents

    if fingerprint_match and param_match and intents_match:
        verdict = "identical"
        explain = "完全一致：同样的输入与参数，重算得到同样的建议——可复现性成立"
    elif not fingerprint_match:
        verdict = "unreproducible"
        explain = (
            "数据指纹不一致：当时的行情已被回补或修改，输入本身变了。"
            "这不代表策略有问题，但那天的建议无法再精确复现"
        )
    else:
        verdict = "drifted"
        explain = (
            "输入一致但结果不同。若策略版本或参数哈希也变了，说明是代码/参数改动导致；"
            "两者都没变却结果不同，则说明存在未被记录的随机性或隐藏状态，需要排查"
        )

    return {
        "trade_date": trade_date,
        "verdict": verdict,
        "explain": explain,
        "fingerprint": {
            "archived": archived.data_fingerprint,
            "fresh": fresh.data_fingerprint,
            "match": fingerprint_match,
        },
        "param_hash": {
            "archived": archived.param_hash,
            "fresh": fresh.param_hash,
            "match": param_match,
        },
        "strategy_versions": {
            "archived": dict(archived.strategy_versions),
            "fresh": dict(fresh.strategy_versions),
        },
        "intents": {
            "archived": sorted(f"{s} {side} {q}" for s, side, q in archived_intents),
            "fresh": sorted(f"{s} {side} {q}" for s, side, q in fresh_intents),
            "match": intents_match,
            "only_archived": sorted(
                f"{s} {side} {q}" for s, side, q in archived_intents - fresh_intents
            ),
            "only_fresh": sorted(
                f"{s} {side} {q}" for s, side, q in fresh_intents - archived_intents
            ),
        },
    }


def _outcome_of(orders: list[dict[str, Any]]) -> str:
    """归纳一条意图的最终去向。

    Args:
        orders: 与该意图关联的订单。

    Returns:
        去向描述。
    """
    if not orders:
        return "未执行（当日没有对应的执行记录）"
    statuses = {o["status"] for o in orders}
    if statuses == {"skipped"}:
        reasons = {o.get("skip_reason") or "未记录原因" for o in orders}
        return f"人工跳过：{'、'.join(sorted(reasons))}"
    return f"已提交（{'、'.join(sorted(statuses))}）"
