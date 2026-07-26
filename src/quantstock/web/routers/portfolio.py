"""P9 组合页（docs/09 第三节）。

当前权重 vs 目标权重、约束满足情况。

目标权重取自**最新一份已保存的交易计划**，而不是现场重算一遍。理由：
组合页要回答的是"按今天的建议，我该调成什么样"，而那份建议已经过了
风控与解释完整性检查；现场重算会得到一个没经过那些关卡的数字，
两个页面显示不同的目标值，用户不知道该信哪个。
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from fastapi import APIRouter

from quantstock.infra.clock import today
from quantstock.infra.types import Money, Symbol
from quantstock.services.advisor_service import constraints_from
from quantstock.web.deps import AuthDep

__all__ = ["router"]

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


def _latest_prices(state: Any, symbols: list[Symbol]) -> dict[Symbol, Money]:  # noqa: ANN401
    """取最新收盘价。

    Args:
        state: 应用状态。
        symbols: 标的。

    Returns:
        标的 → 价格。
    """
    if not symbols:
        return {}
    end = today()
    history = state.data.read_bars(symbols, start=end - dt.timedelta(days=30), end=end)
    return {sym: bars[-1].close for sym, bars in history.items() if bars}


@router.get("/weights")
async def weights(state: AuthDep, trade_date: str | None = None) -> dict[str, Any]:
    """当前权重 vs 目标权重。"""
    date = dt.date.fromisoformat(trade_date) if trade_date else today()
    plan = state.advisor.store.latest(date)
    if plan is None:
        dates = state.advisor.store.list_dates()
        plan = state.advisor.store.latest(dates[-1]) if dates else None

    positions = state.account.positions()
    symbols = list(positions)
    if plan is not None:
        symbols.extend(Symbol(str(i.symbol)) for i in plan.intents)
    prices = _latest_prices(state, list(dict.fromkeys(symbols)))

    summary = state.account.summary(prices)
    total = summary.total_value
    current: dict[str, dict[str, Any]] = {}
    for sym, pos in positions.items():
        price = prices.get(sym)
        value = price * pos.qty if price is not None else Decimal(0)
        current[str(sym)] = {
            "qty": pos.qty,
            "market_value": str(value),
            "weight": float(value / total) if total else 0.0,
            # 缺现价的标的权重会算成 0。必须标出来，否则饼图上它会凭空消失
            "priced": price is not None,
        }

    # 目标 = 当前 + 建议的增减。建议里给的是"要买/卖多少股"，
    # 换算成权重才能和当前权重放在一张图上比
    target: dict[str, dict[str, Any]] = {}
    if plan is not None:
        for intent in plan.intents:
            symbol = Symbol(str(intent.symbol))
            price = prices.get(symbol)
            if price is None:
                continue
            held = positions.get(symbol)
            base_qty = held.qty if held else 0
            side = str(getattr(intent.side, "value", intent.side))
            new_qty = base_qty + (intent.qty if side == "buy" else -intent.qty)
            value = price * max(new_qty, 0)
            target[str(symbol)] = {
                "qty": max(new_qty, 0),
                "market_value": str(value),
                "weight": float(value / total) if total else 0.0,
                "delta_qty": intent.qty if side == "buy" else -intent.qty,
                "side": side,
            }

    universe = sorted(set(current) | set(target))
    rows = [
        {
            "symbol": sym,
            "current_weight": current.get(sym, {}).get("weight", 0.0),
            "target_weight": target.get(sym, {}).get(
                "weight", current.get(sym, {}).get("weight", 0.0)
            ),
            "current_qty": current.get(sym, {}).get("qty", 0),
            "target_qty": target.get(sym, {}).get("qty", current.get(sym, {}).get("qty", 0)),
            "delta_qty": target.get(sym, {}).get("delta_qty", 0),
            "priced": current.get(sym, {}).get("priced", sym in target),
        }
        for sym in universe
    ]
    for row in rows:
        row["weight_drift"] = row["target_weight"] - row["current_weight"]

    return {
        "trade_date": plan.trade_date.isoformat() if plan is not None else None,
        "plan_id": str(plan.plan_id) if plan is not None else None,
        "total_value": str(total),
        "cash": str(summary.cash),
        "cash_weight": float(summary.cash / total) if total else 0.0,
        "rows": sorted(rows, key=lambda r: -abs(float(r["weight_drift"]))),
        "unpriced_symbols": list(summary.unpriced_symbols),
        "is_empty": summary.is_empty,
    }


@router.get("/constraints")
async def constraints(state: AuthDep) -> dict[str, Any]:
    """当前组合对各项约束的满足情况。

    **超限的项要显式列出来**，而不是只给一个"合规/不合规"的总判定——
    知道"哪一条、超了多少"才知道该卖什么。
    """
    positions = state.account.positions()
    prices = _latest_prices(state, list(positions))
    summary = state.account.summary(prices)
    total = summary.total_value

    # 复用 advisor 用的同一套约束。抄一份到这里迟早会分叉，
    # 那时组合页说「合规」而建议页照样拒单，用户不知道该信哪个
    limits = constraints_from(state.settings)
    max_single = limits.max_single_position
    max_holdings = limits.max_holdings
    # 现金下限由权益上限反推，而不是单开一个配置项——两个数各配一遍
    # 迟早会互相矛盾（比如权益上限 90% 配现金下限 20%），那时谁也说不清该听哪个
    min_cash = Decimal(1) - limits.max_equity_exposure

    breaches: list[dict[str, Any]] = []
    if total:
        for sym, pos in positions.items():
            price = prices.get(sym)
            if price is None:
                continue
            weight = price * pos.qty / total
            if weight > max_single:
                breaches.append(
                    {
                        "rule_id": "B01",
                        "symbol": str(sym),
                        "actual": float(weight),
                        "limit": float(max_single),
                        "message": f"{sym} 占比 {weight:.1%} 超过单票上限 {max_single:.1%}",
                    }
                )

    cash_ratio = summary.cash / total if total else Decimal(0)
    if total and cash_ratio < min_cash:
        breaches.append(
            {
                "rule_id": "B04",
                "symbol": "",
                "actual": float(cash_ratio),
                "limit": float(min_cash),
                "message": f"现金占比 {cash_ratio:.1%} 低于下限 {min_cash:.1%}",
            }
        )

    if len(positions) > max_holdings:
        breaches.append(
            {
                "rule_id": "B03",
                "symbol": "",
                "actual": len(positions),
                "limit": max_holdings,
                "message": f"持仓 {len(positions)} 只超过上限 {max_holdings} 只",
            }
        )

    return {
        "satisfied": not breaches,
        "breaches": breaches,
        "limits": {
            "max_single_position": float(max_single),
            "max_holdings": max_holdings,
            "min_cash_ratio": float(min_cash),
        },
        "current": {
            "holdings": len(positions),
            "cash_ratio": float(cash_ratio),
            "total_value": str(total),
        },
    }
