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
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from quantstock.config.settings import Settings, load_settings
from quantstock.infra.errors import QuantStockError
from quantstock.infra.logging import get_logger
from quantstock.services.config_service import ConfigService
from quantstock.services.system_service import SystemService

__all__ = ["create_app"]

_log = get_logger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class AppState:
    """应用级共享状态。

    Attributes:
        settings: 运行期配置。
        access_token: 本地访问口令。首次启动随机生成并打印到终端。
        readonly: 只读模式，所有写操作被拒绝。
    """

    def __init__(self, settings: Settings, *, readonly: bool = False) -> None:
        """初始化。

        Args:
            settings: 运行期配置。
            readonly: 是否只读模式。
        """
        self.settings = settings
        self.readonly = readonly
        self.access_token = secrets_mod.token_urlsafe(16)
        self.config_service = ConfigService(settings.config_dir, settings.var_dir)
        self.system_service = SystemService(settings)


def _get_state(request: Request) -> AppState:
    """从请求取应用状态。

    Args:
        request: 当前请求。

    Returns:
        应用状态。
    """
    state: AppState = request.app.state.app_state
    return state


StateDep = Annotated[AppState, Depends(_get_state)]


def _require_token(
    state: StateDep,
    x_access_token: Annotated[str | None, Header(alias="X-Access-Token")] = None,
) -> AppState:
    """校验访问口令。

    界面成为下单入口后必须加认证——即使只监听回环地址，
    同机的其它程序也能访问（见 docs/09-可视化界面规格.md 第五节）。

    Args:
        state: 应用状态。
        x_access_token: 请求头中的口令。

    Returns:
        应用状态。

    Raises:
        HTTPException: 口令缺失或错误。
    """
    if not x_access_token or not secrets_mod.compare_digest(x_access_token, state.access_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="访问口令无效")
    return state


AuthDep = Annotated[AppState, Depends(_require_token)]


def _require_writable(state: AuthDep) -> AppState:
    """校验当前允许写操作。

    Args:
        state: 已认证的应用状态。

    Returns:
        应用状态。

    Raises:
        HTTPException: 处于只读模式。
    """
    if state.readonly:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="当前为只读模式，写操作已禁用"
        )
    return state


WriteDep = Annotated[AppState, Depends(_require_writable)]


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

    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

        @app.get("/", include_in_schema=False)
        async def _index() -> FileResponse:
            """返回单页应用入口。"""
            return FileResponse(STATIC_DIR / "index.html")

    return app


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
