"""按页面分组的 API 路由。

每个模块对应 docs/09-可视化界面规格.md 里的一到两个页面。拆开是因为
单文件注册几十个路由后，改一个页面要在上千行里翻找——但**分组本身不带
任何业务语义**，路由函数依然只做"解析请求 → 调 services → 序列化"三件事。
"""

from __future__ import annotations

from quantstock.web.routers.advisor import router as advisor_router
from quantstock.web.routers.backtest import router as backtest_router
from quantstock.web.routers.data import router as data_router
from quantstock.web.routers.execution import router as execution_router
from quantstock.web.routers.intel import router as intel_router
from quantstock.web.routers.llm import router as llm_router

__all__ = [
    "advisor_router",
    "backtest_router",
    "data_router",
    "execution_router",
    "intel_router",
    "llm_router",
]
