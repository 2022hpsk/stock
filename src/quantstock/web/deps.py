"""应用状态与依赖注入。

从 ``app.py`` 拆出来，让各 router 模块能共用而不产生循环导入。

**服务全部惰性构造**。``DataService`` 要建数据源、读标的表，``IntelService``
要装配情报源并加载黑名单——在启动时全建一遍会让 ``quantstock ui`` 明显变慢，
而且用户可能只想看看配置页，根本不碰数据。构造失败也只影响用到它的那个页面，
不至于整个界面起不来。
"""

from __future__ import annotations

import secrets as secrets_mod
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from quantstock.config.settings import Settings
from quantstock.services.account_service import AccountService
from quantstock.services.advisor_service import AdvisorService
from quantstock.services.backtest_service import BacktestService
from quantstock.services.config_service import ConfigService
from quantstock.services.data_service import DataService
from quantstock.services.execution_service import ExecutionService
from quantstock.services.intel_service import IntelService
from quantstock.services.llm_service import LLMService
from quantstock.services.system_service import SystemService
from quantstock.web.events import EventHub

__all__ = ["AppState", "AuthDep", "StateDep", "WriteDep"]


class AppState:
    """应用级共享状态。

    Attributes:
        settings: 运行期配置。
        access_token: 本地访问口令。首次启动随机生成并打印到终端。
        readonly: 只读模式，所有写操作被拒绝。
        events: WebSocket 事件总线。
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
        self.events = EventHub()

        # 这两个轻量且几乎每个页面都要用，直接建
        self.config_service = ConfigService(settings.config_dir, settings.var_dir)
        self.system_service = SystemService(settings)

        self._account: AccountService | None = None
        self._data: DataService | None = None
        self._advisor: AdvisorService | None = None
        self._execution: ExecutionService | None = None
        self._intel: IntelService | None = None
        self._llm: LLMService | None = None
        self._backtest: BacktestService | None = None

    @property
    def account(self) -> AccountService:
        """账户服务。"""
        if self._account is None:
            self._account = AccountService(self.settings)
        return self._account

    @property
    def data(self) -> DataService:
        """数据服务。"""
        if self._data is None:
            self._data = DataService(self.settings)
        return self._data

    @property
    def advisor(self) -> AdvisorService:
        """建议服务。"""
        if self._advisor is None:
            self._advisor = AdvisorService(self.settings, data=self.data)
        return self._advisor

    @property
    def execution(self) -> ExecutionService:
        """执行服务。"""
        if self._execution is None:
            self._execution = ExecutionService(self.settings)
        return self._execution

    @property
    def intel(self) -> IntelService:
        """情报服务。"""
        if self._intel is None:
            self._intel = IntelService(self.settings)
        return self._intel

    @property
    def llm(self) -> LLMService:
        """大模型服务。"""
        if self._llm is None:
            self._llm = LLMService(self.settings)
        return self._llm

    @property
    def backtest(self) -> BacktestService:
        """回测服务。"""
        if self._backtest is None:
            self._backtest = BacktestService(self.settings, data=self.data)
        return self._backtest


def get_state(request: Request) -> AppState:
    """从请求取应用状态。

    Args:
        request: 当前请求。

    Returns:
        应用状态。
    """
    state: AppState = request.app.state.app_state
    return state


StateDep = Annotated[AppState, Depends(get_state)]


def require_token(
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


AuthDep = Annotated[AppState, Depends(require_token)]


def require_writable(state: AuthDep) -> AppState:
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


WriteDep = Annotated[AppState, Depends(require_writable)]
