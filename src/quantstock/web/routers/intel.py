"""P5 情报页（docs/09 第三节）。

情报流、分域摘要、外置导入（拖拽 / 表单 / 批量）、源健康、黑名单。

两条贯穿本模块的红线：

- **I-R4**：进入界面的每条情报都带 ``url`` 与 ``publish_at``。没有出处的
  情报在界面上就是不可核实的传闻，看着像证据、实际上不是；
- **I-R1**：情报不能单独触发买入。所以这里**没有**任何"据此下单"的接口——
  黑名单解除是唯一的写操作，而它只放松单向否决，不产生买入意图。
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel, Field

from quantstock.web.deps import AuthDep, WriteDep
from quantstock.web.serializers import serialize_intel_item

__all__ = ["router"]

router = APIRouter(prefix="/api/intel", tags=["intel"])

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
"""上传文件大小上限。情报是文本，5 MB 已经远超正常范围；
不设上限的话一个误拖的视频文件就能把内存吃满。"""


class NoteRequest(BaseModel):
    """手工录入一条情报。"""

    text: str = Field(min_length=1, description="正文")
    title: str = ""
    domain: str | None = Field(default=None, description="情报域；空表示由分类器推断")
    symbols: list[str] = Field(default_factory=list)
    url: str = ""
    importance: int | None = None
    sentiment: float | None = None
    publish_at: str | None = Field(default=None, description="发布时间；空表示此刻")


class FetchRequest(BaseModel):
    """采集请求。"""

    lookback_days: int = Field(default=7, ge=1, le=90)
    include_inbox: bool = True
    domains: list[str] = Field(default_factory=list)


@router.get("/status")
async def intel_status(state: AuthDep) -> dict[str, Any]:
    """情报模块状态。"""
    s = state.intel.status()
    return {
        "sources": s.sources,
        "inbox_pending": s.inbox_pending,
        "latest_date": s.latest_date.isoformat() if s.latest_date else None,
        "blacklisted": s.blacklisted,
        "message": s.message,
        "inbox_dir": str(state.intel.inbox_dir),
    }


@router.get("/sources")
async def sources(state: AuthDep) -> dict[str, Any]:
    """已注册的情报源与健康度。

    健康检查会真的去探接口，比较慢，所以是单独的接口而不是塞进 ``/status``——
    仪表盘每次刷新都探一遍外网既慢又不礼貌（红线 I-R6）。
    """
    return {
        "sources": [
            {
                "name": s.name,
                "domains": [str(getattr(d, "value", d)) for d in s.domains],
            }
            for s in state.intel.registry.all()
        ]
    }


@router.get("/health")
async def health(state: AuthDep) -> dict[str, Any]:
    """逐源探测可用性。"""
    out = []
    for source in state.intel.registry.all():
        h = source.health_check()
        out.append(
            {
                "source": h.source,
                "ok": h.ok,
                "fetched": h.fetched,
                "error": h.error,
                "latency_ms": h.latency_ms,
            }
        )
    return {"health": out}


@router.post("/fetch")
async def fetch(body: FetchRequest, state: WriteDep) -> dict[str, Any]:
    """采集全域情报并落库。重复执行幂等。"""
    state.events.publish("tasks", "progress", task="intel.fetch", stage="fetching")
    result = state.intel.fetch(
        lookback_days=body.lookback_days,
        include_inbox=body.include_inbox,
    )
    payload = _pipeline_payload(result)
    state.events.publish("tasks", "done", task="intel.fetch", items=payload["items"])
    return payload


@router.get("/digest")
async def digest(
    state: AuthDep,
    trade_date: str | None = None,
    session: str = "post",
    lookback_days: int = 7,
) -> dict[str, Any]:
    """分域摘要。命中持仓的置顶。"""
    d = state.intel.digest(
        trade_date=dt.date.fromisoformat(trade_date) if trade_date else None,
        session=session,
        lookback_days=lookback_days,
    )
    return {
        "trade_date": d.trade_date.isoformat(),
        "generated_at": d.generated_at.isoformat(),
        "session": d.session,
        "by_domain": {
            str(getattr(k, "value", k)): {
                "highlights": list(v.highlights),
                "count": v.count,
                "net_sentiment": v.net_sentiment,
                "llm_generated": v.llm_generated,
                "symbols": [str(s) for s in v.symbols],
                "items": [serialize_intel_item(i) for i in v.items],
            }
            for k, v in d.by_domain.items()
        },
        "top_items": [serialize_intel_item(i) for i in d.top_items],
        "portfolio_alerts": [
            {
                "symbol": str(a.symbol),
                "severity": a.severity,
                "action_hint": a.action_hint,
                "item": serialize_intel_item(a.item),
            }
            for a in d.portfolio_alerts
        ],
        "watchlist_hits": [serialize_intel_item(i) for i in d.watchlist_hits],
        # 分不清"没查到"和"查了没有"是最糟的状态，所以缺失的域必须显式列出
        "missing_domains": [str(getattr(x, "value", x)) for x in d.missing_domains],
        "failed_sources": list(d.failed_sources),
        "lines": state.intel.render_digest(d),
    }


@router.post("/note")
async def note(body: NoteRequest, state: WriteDep) -> dict[str, Any]:
    """手工录入一条情报（外置导入方式二）。"""
    fields: dict[str, object] = {"title": body.title, "url": body.url}
    if body.domain:
        fields["domain"] = body.domain
    if body.symbols:
        fields["symbols"] = body.symbols
    if body.importance is not None:
        fields["importance"] = body.importance
    if body.sentiment is not None:
        fields["sentiment"] = body.sentiment
    if body.publish_at:
        fields["publish_at"] = body.publish_at

    item = state.intel.note(body.text, **fields)
    result = state.intel.ingest([item])
    return {"item": serialize_intel_item(item), **_pipeline_payload(result)}


@router.post("/upload")
async def upload(state: WriteDep, file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
    """拖拽上传一个情报文件（md / txt / json / csv）。

    落到收件箱后立刻扫描入库——**必须走收件箱这条既有路径**，
    而不是直接把内容塞进数据湖。收件箱扫描器会做解析、去重、实体链接、
    重要性打分；绕过它导入的条目在日报里既不参与去重也不参与排序。
    """
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        return {"ok": False, "error": f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024} MB 上限"}

    name = (file.filename or "upload.md").replace("/", "_")
    target = state.intel.inbox_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / name).write_bytes(raw)

    report = state.intel.scan_inbox()
    result = state.intel.ingest(report.items)
    return {
        "ok": True,
        "filename": name,
        "parsed": len(report.items),
        "failed": [{"file": str(f), "reason": r} for f, r in report.failed],
        **_pipeline_payload(result),
    }


@router.post("/inbox/scan")
async def scan_inbox(state: WriteDep) -> dict[str, Any]:
    """扫描收件箱并入库。"""
    report = state.intel.scan_inbox()
    result = state.intel.ingest(report.items)
    return {
        "parsed": len(report.items),
        "failed": [{"file": str(f), "reason": r} for f, r in report.failed],
        **_pipeline_payload(result),
    }


@router.get("/blacklist")
async def blacklist(state: AuthDep) -> dict[str, Any]:
    """当前生效的情报黑名单。

    每条都带触发它的情报 id 与**原文链接**——被禁止买入的标的，
    用户有权一路点回到那条公告本身（红线 I-R4）。
    """
    return {
        "entries": [
            {
                "symbol": str(e.symbol),
                "reason": e.reason,
                "rule": e.rule,
                "triggered_at": e.triggered_at.isoformat(),
                "expires_at": e.expires_at.isoformat(),
                "item_ids": list(e.item_ids),
                "urls": list(e.urls),
            }
            for e in state.intel.blacklist_entries()
        ]
    }


def _pipeline_payload(result: Any) -> dict[str, Any]:  # noqa: ANN401 - 鸭子类型
    """把流水线结果转成响应片段。

    Args:
        result: 流水线结果。

    Returns:
        JSON 片段。
    """
    return {
        "items": len(result.items),
        "merged": result.dedup_result.dropped_count,
        "blacklisted": [str(s) for s in result.blacklisted],
        "summary": result.summary,
    }
