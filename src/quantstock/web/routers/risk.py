"""P10 风控页（docs/09 第三节）。

规则表、熔断距离、绝对金额硬闸、账户合理性。

**这里没有任何修改阈值的接口**。改阈值走配置页（校验 → Diff → 备份 → 审计），
给风控单开一条"快速调整"的路径，等于把最该留痕的操作做成了最容易
悄悄做掉的操作。

规则表带 ``closable`` 字段，界面据此**根本不渲染** A 类的开关（验收 5）——
画出来再拒绝和根本不画是两回事，前者会让人一直去试。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi import APIRouter

from quantstock.web.deps import AuthDep

__all__ = ["router"]

router = APIRouter(prefix="/api/risk", tags=["risk"])


@router.get("/rules")
async def rules(state: AuthDep) -> dict[str, Any]:
    """全部风控规则。"""
    views = state.risk.rules()
    return {
        "count": len(views),
        "rules": [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "rule_class": r.rule_class,
                "description": r.description,
                # A 类恒为 False。界面不得为 closable=False 的规则渲染开关
                "closable": r.closable,
                "threshold_editable": r.threshold_editable,
                "threshold_key": r.threshold_key,
                "current_threshold": r.current_threshold,
            }
            for r in views
        ],
        "legend": {
            "A": "市场规则与绝对硬闸，不可关闭、不可在界面调整",
            "B": "组合约束，阈值可配，规则本身不可关闭",
            "C": "建议性检查，可关闭",
        },
    }


@router.get("/hard-limits")
async def hard_limits(state: AuthDep) -> dict[str, Any]:
    """绝对金额硬闸（A10/A11）。

    与比例风控**用不同代码路径实现**，形成双保险——共用实现会共同失效。
    比例约束挡不住计算基数出错：总资产算成十倍时，"单票不超过 15%"
    照样会放出一笔十倍大的委托。
    """
    view = state.risk.hard_limits()
    return {
        "enabled": view.enabled,
        "max_single_order_amount": str(view.max_single_order_amount),
        "max_daily_total_amount": str(view.max_daily_total_amount),
        "max_daily_order_count": view.max_daily_order_count,
        "min_account_value_sanity": str(view.min_account_value_sanity),
        "max_account_value_sanity": str(view.max_account_value_sanity),
        "message": view.message,
        # 阈值必须由用户按自己的实际资金规模手工设定，程序不自动推导——
        # 自动推导会被同一个错误数据污染，失去防护意义
        "auto_derived": False,
    }


@router.get("/circuit")
async def circuit(
    state: AuthDep, daily_return: float = 0.0, drawdown_20d: float = 0.0
) -> dict[str, Any]:
    """熔断状态与距各阈值的距离。

    **显示距离而不只显示状态**：状态只有到了才变，而距离能让人提前
    看到自己正在往哪走。
    """
    distance = state.risk.circuit_distance(
        daily_return=Decimal(str(daily_return)), drawdown_20d=Decimal(str(drawdown_20d))
    )
    halt = state.risk.halt_state()
    return {
        **distance,
        "halt": {
            "halted": halt.halted,
            "reason": halt.reason,
            "halted_at": halt.halted_at,
            "halted_by": halt.halted_by,
        },
        "note": "HALTED 不会自动恢复，必须人工解除——自动恢复会让系统在剧烈波动中反复进出",
    }
