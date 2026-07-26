"""P4 数据页（docs/09 第三节）。

数据源健康、覆盖状态、更新任务、K 线浏览。

K 线接口有一处必须显式：**复权口径**。研究用后复权（``hfq``），
下单与展示用不复权（``none``）——红线 R4。接口默认给不复权，
因为界面上的 K 线是给人对着看盘软件核对的，后复权价格对不上。
"""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from quantstock.infra.clock import today
from quantstock.infra.types import Symbol
from quantstock.web.deps import AuthDep, WriteDep

__all__ = ["router"]

router = APIRouter(prefix="/api/data", tags=["data"])

MAX_BARS = 2000
"""单次返回的最大 K 线数。再多前端图表也画不清，白白撑大响应体。"""


class UpdateRequest(BaseModel):
    """行情更新请求。"""

    symbols: list[str] = Field(default_factory=list, description="标的；空表示按档位解析")
    tier: str = Field(default="core", description="候选池档位 core / all")
    start: str | None = Field(default=None, description="起始日；空表示增量续拉")
    end: str | None = Field(default=None, description="结束日；空表示今日")
    sync_instruments: bool = Field(default=False, description="同时同步标的表")


@router.get("/status")
async def data_status(state: AuthDep) -> dict[str, Any]:
    """数据湖状态与各源健康度。"""
    s = state.data.status()
    return {
        "root": str(s.root),
        "symbols": s.symbols,
        "files": s.files,
        "bytes_on_disk": s.bytes_on_disk,
        "latest_date": s.latest_date.isoformat() if s.latest_date else None,
        "instruments": s.instruments,
        "delisted": s.delisted,
        "is_ready": s.is_ready,
        "message": s.message,
        "health": [
            {
                "source": h.name,
                "ok": h.ok,
                "checked_at": h.checked_at.isoformat(),
                "message": h.message,
                "latency_ms": h.latency_ms,
                "consecutive_failures": h.consecutive_failures,
            }
            for h in s.health
        ],
    }


@router.get("/universe")
async def universe(state: AuthDep, tier: str = "core") -> dict[str, Any]:
    """候选池。"""
    symbols = state.data.resolve_universe(tier)
    return {"tier": tier, "symbols": [str(s) for s in symbols], "count": len(symbols)}


@router.post("/update")
async def update(body: UpdateRequest, state: WriteDep) -> dict[str, Any]:
    """拉取并写入行情。

    与 CLI 的 ``quantstock data update`` 走同一个服务方法——
    规格验收 3 要求 UI 与 CLI 结果完全一致，靠的就是不给界面单开一条路径。
    """
    symbols = (
        tuple(Symbol(s) for s in body.symbols)
        if body.symbols
        else state.data.resolve_universe(body.tier)
    )
    state.events.publish(
        "tasks", "progress", task="data.update", stage="fetching", total=len(symbols)
    )

    instruments = state.data.sync_instruments() if body.sync_instruments else 0
    report = state.data.update(
        symbols,
        start=dt.date.fromisoformat(body.start) if body.start else None,
        end=dt.date.fromisoformat(body.end) if body.end else None,
    )

    result = {
        "symbols": report.symbols,
        "bars_written": report.bars_written,
        "start": report.start.isoformat(),
        "end": report.end.isoformat(),
        "source": report.source,
        "failures": list(report.failures),
        "instruments": instruments,
        "summary": report.summary,
    }
    state.events.publish("tasks", "done", task="data.update", **result)
    return result


@router.get("/bars")
async def bars(
    state: AuthDep,
    symbol: Annotated[str, Query(description="标的代码，如 600519.SH")],
    start: str | None = None,
    end: str | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    """K 线数据，供 ECharts 蜡烛图使用。

    返回的价格是**数据湖里存的口径**（研究用后复权），字段里显式标出，
    界面必须把它显示给用户看——不标口径的 K 线图对不上看盘软件时，
    人会以为是数据错了。
    """
    sym = Symbol(symbol)
    capped = min(limit, MAX_BARS)
    finish = dt.date.fromisoformat(end) if end else today()
    # 不给起点时按天数倒推，多留出周末与节假日的余量，
    # 否则 500 根 K 线只会取回约 340 根
    begin = dt.date.fromisoformat(start) if start else finish - dt.timedelta(days=capped * 2)
    history = state.data.read_bars([sym], start=begin, end=finish)
    series = history.get(sym, [])[-capped:]

    return {
        "symbol": symbol,
        "adjust": state.settings.config.data.adjust_for_research,
        "count": len(series),
        "bars": [
            {
                "date": b.trade_date.isoformat(),
                "open": str(b.open),
                "high": str(b.high),
                "low": str(b.low),
                "close": str(b.close),
                "volume": b.volume,
                "amount": str(b.amount) if getattr(b, "amount", None) is not None else None,
            }
            for b in series
        ],
    }
