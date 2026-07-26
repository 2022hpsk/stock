"""P8 回测页（docs/09 第三节）。

发起回测、查看指标与净值、trials 记录表、过拟合风险。

**界面上最容易被误用的一个功能**，所以这里的接口刻意做了几处"不方便"：

- 每次回测**默认都记入 trials**。界面上没有"这次不算"的开关——
  删掉失败尝试会让 DSR 系统性偏乐观，而 DSR 正是用来判断
  "这个结果是真的好还是试出来的"；
- ``warnings`` 与 ``admission`` 一并返回，界面必须显著展示。
  一次好看的回测不等于策略好。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from quantstock.infra.errors import StrategyError
from quantstock.infra.money import money
from quantstock.infra.types import Symbol
from quantstock.web.deps import AuthDep, WriteDep
from quantstock.web.serializers import serialize_backtest

__all__ = ["router"]

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


class RunRequest(BaseModel):
    """回测请求。"""

    start: str
    end: str
    symbols: list[str] = Field(default_factory=list, description="候选池；空表示按档位解析")
    tier: str = "core"
    initial_cash: str | None = Field(default=None, description="初始资金，字符串（红线 R1）")
    rebalance_days: int = Field(default=5, ge=1, le=60)
    segment: str = Field(
        default="train",
        description=(
            "数据段 train / validation / test。"
            "test 段每个策略只允许跑一次——反复在测试集上调参，它就变成了第二个训练集"
        ),
    )


@router.post("/run")
async def run(body: RunRequest, state: WriteDep) -> dict[str, Any]:
    """在历史区间上回测每日建议逻辑。

    跑的是 ``AdvisorService`` 的同一套打分与组合逻辑，不是另写的回测策略——
    两套逻辑迟早分叉，那时回测结果就不再说明实盘会怎样。
    """
    state.events.publish("tasks", "progress", task="backtest.run", stage="running")
    report = state.backtest.run(
        start=dt.date.fromisoformat(body.start),
        end=dt.date.fromisoformat(body.end),
        universe=[Symbol(s) for s in body.symbols] or None,
        tier=body.tier,
        initial_cash=money(body.initial_cash) if body.initial_cash else None,
        rebalance_days=body.rebalance_days,
        segment=body.segment,
    )
    payload = serialize_backtest(report)
    state.events.publish(
        "tasks",
        "done",
        task="backtest.run",
        trial_id=report.trial_id,
        sharpe=round(report.stats.sharpe, 3),
    )
    return payload


@router.get("/trials")
async def trials(state: AuthDep, strategy: str = "daily_advice") -> dict[str, Any]:
    """全部试验记录。

    界面按 Sharpe 排序展示时必须同时显示**试验次数**——
    试了 200 次挑出来的 Sharpe 2.0 和试了 3 次得到的 Sharpe 2.0，
    含金量差着数量级。
    """
    records = state.backtest.trial_records(strategy)
    return {
        "strategy": strategy,
        "count": len(records),
        "trials": [
            {
                "trial_id": t.trial_id,
                "params": t.params,
                "sharpe": t.sharpe,
                "annual_return": t.annual_return,
                "max_drawdown": t.max_drawdown,
                "turnover": t.turnover,
                "n_periods": t.n_periods,
                "segment": t.segment,
                "note": t.note,
                "created_at": t.created_at,
            }
            for t in records
        ],
    }


@router.get("/admission")
async def admission(state: AuthDep, strategy: str = "daily_advice") -> dict[str, Any]:
    """实盘候选池准入检查（A5 强制门槛）。

    DSR < 0.95 或 PBO > 0.5 时禁止进实盘候选池。界面必须把这个结论
    放在回测结果**旁边**而不是藏在另一个页签里——好看的净值曲线配上
    "该 Sharpe 用随机噪声即可试出"的结论，人才会真的停下来想一想。
    """
    try:
        verdict = state.backtest.admission(strategy)
    except StrategyError as exc:
        return {
            "strategy": strategy,
            "available": False,
            "message": exc.message,
        }
    return {
        "strategy": strategy,
        "available": True,
        "admitted": verdict.admitted,
        "dsr": verdict.dsr,
        "pbo": verdict.pbo,
        "n_trials": verdict.n_trials,
        "reasons": list(verdict.reasons),
        "explain": verdict.explain(),
    }
