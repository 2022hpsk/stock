"""FastAPI 应用。

规范见 docs/09-可视化界面规格.md。

本层是**薄**客户端：只做请求解析、调用 ``services``、序列化结果，
不含任何业务逻辑（由 import-linter 契约强制：``web`` 只能依赖 ``services`` 与 ``config``）。
"""

from __future__ import annotations

import secrets as secrets_mod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from quantstock.config.settings import Settings, load_settings
from quantstock.infra.errors import QuantStockError
from quantstock.infra.logging import get_logger
from quantstock.web.deps import AppState, AuthDep, StateDep, WriteDep
from quantstock.web.events import parse_channels
from quantstock.web.routers import (
    account_router,
    advisor_router,
    backtest_router,
    data_router,
    execution_router,
    intel_router,
    llm_router,
    review_router,
    risk_router,
)

__all__ = ["create_app"]

_log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
DIST_DIR = Path(__file__).parent / "dist"
"""Vue 构建产物。发版时预编译并随包分发，用户不需要 Node 环境。

存在时优先于 ``static/``——后者是无构建工具时的降级页面，
保证从源码直接运行（还没跑过 ``make ui-build``）时界面也不是白屏。
"""


# --------------------------------------------------------------------- 请求体
class ConfigSaveRequest(BaseModel):
    """配置保存请求。"""

    config: dict[str, Any] = Field(description="完整配置字典")
    dry_run: bool = Field(default=False, description="只校验与预览，不写入")
    changed_by: str = Field(default="ui", description="操作人标识，写入审计日志")


class HaltRequest(BaseModel):
    """急停请求。"""

    reason: str = Field(min_length=1, description="急停原因，必填——事后复盘要靠它")
    by: str = Field(default="ui", description="操作人标识")


class RollbackRequest(BaseModel):
    """配置回滚请求。"""

    version: str = Field(min_length=1, description="备份时间戳")


# --------------------------------------------------------------------- 应用
def create_app(
    *,
    config_dir: Path | str = "config",
    readonly: bool = False,
    settings: Settings | None = None,
) -> FastAPI:
    """构造 FastAPI 应用。

    Args:
        config_dir: 配置目录。
        readonly: 只读模式，适合在手机上查看而不误操作。
        settings: 直接注入的配置，测试用。

    Returns:
        FastAPI 应用实例。
    """
    resolved = settings or load_settings(config_dir)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        _log.info(
            "web_started",
            readonly=readonly,
            broker=resolved.config.execution.broker,
            var_dir=str(resolved.var_dir),
        )
        yield
        _log.info("web_stopped")

    app = FastAPI(
        title="quantstock",
        description="A股/场内基金 个人量化投研与半自动交易系统",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.app_state = AppState(resolved, readonly=readonly)

    @app.exception_handler(QuantStockError)
    async def _handle_domain_error(_request: Request, exc: QuantStockError) -> JSONResponse:
        """把领域异常转成结构化响应，保留 context 供界面展示。"""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "context": {k: str(v) for k, v in exc.context.items()},
            },
        )

    _register_routes(app)
    _register_websocket(app)
    for router in (
        data_router,
        account_router,
        advisor_router,
        execution_router,
        intel_router,
        backtest_router,
        risk_router,
        review_router,
        llm_router,
    ):
        app.include_router(router)

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI) -> None:
    """挂载前端。

    Vue 构建产物优先；没构建过时回退到无依赖的静态页，
    这样"刚 clone 完就 quantstock ui"不会是一片白屏。

    Args:
        app: FastAPI 应用。
    """
    root = DIST_DIR if (DIST_DIR / "index.html").exists() else STATIC_DIR
    if not (root / "index.html").exists():
        return

    if (root / "assets").exists():
        app.mount("/assets", StaticFiles(directory=root / "assets"), name="assets")
    elif root == STATIC_DIR:
        app.mount("/assets", StaticFiles(directory=root), name="assets")

    @app.get("/", include_in_schema=False)
    async def _index() -> FileResponse:
        """返回单页应用入口。"""
        return FileResponse(root / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    async def _spa_fallback(path: str) -> FileResponse:
        """前端路由回退。

        Vue Router 用的是 history 模式，直接刷新 ``/advisor`` 时浏览器会向
        后端请求这个路径。没有这个回退就是 404——用户刷新一次页面就"打不开了"。

        Args:
            path: 请求路径。

        Returns:
            静态文件或单页入口。

        Raises:
            HTTPException: 未知的 API 路径。
        """
        # API 路径必须落回 404，不能被单页入口吞掉。否则拼错的接口名会返回
        # 一段 HTML 而状态码是 200，前端 response.json() 抛出的
        # "Unexpected token <" 跟真正的问题毫无关系，极难定位
        if path.startswith(("api/", "ws")):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"未知路径 /{path}")
        candidate = root / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(root / "index.html")


def _register_websocket(app: FastAPI) -> None:
    """注册 WebSocket 推送端点。

    Args:
        app: FastAPI 应用。
    """

    @app.websocket("/ws")
    async def events(websocket: WebSocket) -> None:
        """事件推送。

        查询参数：
        - ``token``：访问口令。WebSocket 不走 HTTP 头，只能放查询串；
        - ``channels``：逗号分隔的频道，缺省全订阅；
        - ``since``：客户端已收到的最大序号，用于断线补齐。
        """
        state: AppState = websocket.app.state.app_state
        token = websocket.query_params.get("token", "")
        if not token or not secrets_mod.compare_digest(token, state.access_token):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="访问口令无效")
            return

        channels = parse_channels(websocket.query_params.get("channels"))
        since = int(websocket.query_params.get("since") or 0)
        await websocket.accept()

        async with state.events.subscribe(channels) as sub:
            # 先补断线期间错过的，再进入实时推送。顺序反过来会让补发的旧事件
            # 盖住刚到的新事件，进度条会往回跳
            for missed in state.events.replay(channels, since=since):
                await websocket.send_json(missed.to_dict())
            await websocket.send_json(
                {"kind": "ready", "channels": sorted(channels), "seq": state.events.last_seq}
            )
            try:
                while True:
                    event = await sub.queue.get()
                    await websocket.send_json(event.to_dict())
            except WebSocketDisconnect:
                _log.info("ws_disconnected")


def _register_routes(app: FastAPI) -> None:  # noqa: C901 - 路由注册是平铺的声明，拆分反而更难看
    """注册全部 API 路由。

    Args:
        app: FastAPI 应用。
    """

    @app.get("/api/health", tags=["system"])
    async def health(state: StateDep) -> dict[str, Any]:
        """健康检查。不需要认证，供启动探测使用。"""
        status_obj = state.system_service.status()
        return {"ok": status_obj.ok, "version": status_obj.version}

    @app.get("/api/system/status", tags=["system"])
    async def system_status(state: AuthDep) -> dict[str, Any]:
        """系统状态，供仪表盘展示。"""
        s = state.system_service.status()
        return {
            "ok": s.ok,
            "version": s.version,
            "checked_at": s.checked_at,
            "readonly": state.readonly,
            "broker": s.broker,
            "llm": {"enabled": s.llm_enabled, "mode": s.llm_mode},
            "halt": {
                "halted": s.halt.halted,
                "reason": s.halt.reason,
                "halted_at": s.halt.halted_at,
                "halted_by": s.halt.halted_by,
            },
            "components": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in s.components],
        }

    @app.post("/api/system/halt", tags=["system"])
    async def halt(body: HaltRequest, state: WriteDep) -> dict[str, Any]:
        """触发急停。此后所有下单路径一律拒绝，直到显式 resume。"""
        result = state.system_service.halt(reason=body.reason, by=body.by)
        return {"halted": result.halted, "reason": result.reason, "halted_at": result.halted_at}

    @app.post("/api/system/resume", tags=["system"])
    async def resume(state: WriteDep) -> dict[str, Any]:
        """解除急停。"""
        result = state.system_service.resume(by="ui")
        return {"halted": result.halted}

    @app.get("/api/config/schema", tags=["config"])
    async def config_schema(state: AuthDep) -> dict[str, Any]:
        """配置的 JSON Schema。界面表单由它自动生成。"""
        return state.config_service.json_schema()

    @app.get("/api/config", tags=["config"])
    async def get_config(state: AuthDep) -> dict[str, Any]:
        """当前生效配置。"""
        return state.config_service.current_dict()

    @app.post("/api/config/preview", tags=["config"])
    async def preview_config(body: ConfigSaveRequest, state: AuthDep) -> dict[str, Any]:
        """校验并生成 Diff 预览，不写入。"""
        issues = state.config_service.validate(body.config)
        return {
            "valid": not issues,
            "issues": [
                {"location": i.location, "message": i.message, "input": i.input_value}
                for i in issues
            ],
            "diff": state.config_service.preview(body.config) if not issues else "",
        }

    @app.put("/api/config", tags=["config"])
    async def save_config(body: ConfigSaveRequest, state: WriteDep) -> dict[str, Any]:
        """保存配置。校验 → Diff → 备份 → 写入。"""
        result = state.config_service.save(
            body.config, changed_by=body.changed_by, dry_run=body.dry_run
        )
        return {
            "saved": result.saved,
            "diff": result.diff,
            "backup": result.backup_path,
            "issues": [
                {"location": i.location, "message": i.message, "input": i.input_value}
                for i in result.issues
            ],
        }

    @app.get("/api/config/backups", tags=["config"])
    async def list_backups(state: AuthDep) -> dict[str, Any]:
        """可回滚的配置备份版本。"""
        return {"versions": state.config_service.backups()}

    @app.post("/api/config/rollback", tags=["config"])
    async def rollback_config(body: RollbackRequest, state: WriteDep) -> dict[str, Any]:
        """回滚到指定备份版本。"""
        result = state.config_service.rollback(body.version)
        return {"saved": result.saved, "diff": result.diff}

    @app.get("/api/secrets/status", tags=["config"])
    async def secrets_status(state: AuthDep) -> dict[str, bool]:
        """各密钥是否已配置。**只返回布尔值，永不返回明文。**"""
        return state.config_service.secrets_status()
