"""P12 复盘页（docs/09 第三节、docs/08 D3）。

计划-实际偏差、人工干预价值。

界面上要顶住的一个诱惑是**别把噪声画成结论**。跳过三次对了两次，
胜率 67%，图表画出来很好看，但那个数字没有任何意义。所以接口一律
返回 ``has_enough_samples`` 与 ``unpriced_skips``，界面必须显示它们。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter

from quantstock.infra.clock import today
from quantstock.web.deps import AuthDep

__all__ = ["router"]

router = APIRouter(prefix="/api/review", tags=["review"])

DEFAULT_WINDOW_DAYS = 180


@router.get("/dates")
async def dates(state: AuthDep) -> dict[str, Any]:
    """有执行记录的日期。"""
    return {"dates": [d.isoformat() for d in state.review.dates()]}


@router.get("/summary")
async def summary(
    state: AuthDep,
    start: str | None = None,
    end: str | None = None,
    horizon_days: int = 20,
) -> dict[str, Any]:
    """区间复盘。"""
    finish = dt.date.fromisoformat(end) if end else today()
    begin = (
        dt.date.fromisoformat(start) if start else finish - dt.timedelta(days=DEFAULT_WINDOW_DAYS)
    )
    s = state.review.summary(start=begin, end=finish, horizon_days=horizon_days)

    return {
        "start": s.start.isoformat(),
        "end": s.end.isoformat(),
        "plans": s.plans,
        "total_planned": s.total_planned,
        "total_executed": s.total_executed,
        "total_skipped": s.total_skipped,
        "execution_rate": s.execution_rate,
        "sample_count": s.sample_count,
        # 界面必须显示这两个：前者决定该不该信结论，
        # 后者说明统计覆盖了多少比例的干预
        "has_enough_samples": s.has_enough_samples,
        "unpriced_skips": s.unpriced_skips,
        "explain": s.explain(),
        "deviations": [
            {
                "trade_date": d.trade_date,
                "planned": d.planned,
                "executed": d.executed,
                "skipped": d.skipped,
                "aborted": d.aborted,
                "execution_rate": d.execution_rate,
                "planned_amount": str(d.planned_amount),
                "executed_amount": str(d.executed_amount),
                "amount_drift": str(d.amount_drift),
                "needs_attention": d.needs_attention,
                "by_reason": dict(d.by_reason),
                "explain": d.explain(),
            }
            for d in s.deviations
        ],
        "interventions": [
            {
                "reason": i.reason,
                "count": i.count,
                "win_rate": i.win_rate,
                "mean_forgone_return": float(i.mean_forgone_return),
                "total_forgone": str(i.total_forgone),
                "has_enough_samples": i.has_enough_samples,
                # verdict 才是这一页的产出：这类干预到底是帮忙还是添乱
                "verdict": i.verdict,
                "explain": i.explain(),
            }
            for i in s.interventions
        ],
    }


@router.get("/deviation/{trade_date}")
async def deviation(trade_date: str, state: AuthDep) -> dict[str, Any]:
    """某日的计划-实际偏差。"""
    report = state.review.deviation(dt.date.fromisoformat(trade_date))
    if report is None:
        return {"available": False, "message": f"{trade_date} 没有执行记录"}
    return {
        "available": True,
        "trade_date": report.trade_date,
        "planned": report.planned,
        "executed": report.executed,
        "skipped": report.skipped,
        "aborted": report.aborted,
        "execution_rate": report.execution_rate,
        "planned_amount": str(report.planned_amount),
        "executed_amount": str(report.executed_amount),
        "amount_drift": str(report.amount_drift),
        "needs_attention": report.needs_attention,
        "by_reason": dict(report.by_reason),
        "explain": report.explain(),
    }
