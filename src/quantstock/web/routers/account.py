"""P1 账户页（docs/09 第三节、docs/11-持仓账本规格.md）。

持仓表、批次明细、资金流水、录入与导入。

**没有"修改持仓"的接口**，只有"记一笔流水"（红线 R8）。持仓、批次、
成本全部由流水重放得出，从不就地修改——写错了用一笔反向的 ``ADJUST``
冲正，并写明理由。允许改历史会让"上周的持仓截图"和"今天重放出来的
上周持仓"对不上，而对不上的时候你根本不知道该信哪个。

批次明细必须展示到界面上：红利税按持股期限分三档、"再持有 N 天可免税"
的倒计时，全都要知道每一份股票是哪天买的。只看平均成本算不出这些。
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from quantstock.infra.clock import today
from quantstock.infra.money import money
from quantstock.infra.types import Money, Symbol
from quantstock.web.deps import AuthDep, WriteDep

__all__ = ["router"]

router = APIRouter(prefix="/api/account", tags=["account"])


class TradeRequest(BaseModel):
    """录入一笔成交。"""

    symbol: str
    side: str = Field(description="buy 或 sell")
    qty: int = Field(gt=0, description="数量，正数。方向由 side 决定")
    price: str = Field(description="成交价，字符串（红线 R1）")
    trade_date: str | None = None
    commission: str = "0"
    stamp_tax: str = "0"
    transfer_fee: str = "0"
    note: str = ""


class CashRequest(BaseModel):
    """入金 / 出金。"""

    amount: str = Field(description="金额，正数。方向由接口决定")
    trade_date: str | None = None
    note: str = ""


class ImportRequest(BaseModel):
    """批量导入流水。"""

    rows: list[dict[str, Any]]


def _latest_prices(state: Any, symbols: list[Symbol]) -> dict[Symbol, Money]:  # noqa: ANN401
    """取各持仓的最新收盘价。

    用数据湖里的收盘价而不是实时价：本系统没有行情推送，
    盘中拿到的"实时价"其实是上一个交易日的收盘价，标成实时反而误导。

    Args:
        state: 应用状态。
        symbols: 标的列表。

    Returns:
        标的 → 价格。数据湖里没有的标的不在结果里。
    """
    if not symbols:
        return {}
    end = today()
    history = state.data.read_bars(symbols, start=end - dt.timedelta(days=30), end=end)
    return {sym: bars[-1].close for sym, bars in history.items() if bars}


@router.get("/summary")
async def summary(state: AuthDep) -> dict[str, Any]:
    """账户总览。"""
    account = state.account
    positions = account.positions()
    prices = _latest_prices(state, list(positions))
    s = account.summary(prices)
    return {
        "account_id": s.account_id,
        "as_of": s.as_of.isoformat(),
        "cash": str(s.cash),
        "market_value": str(s.market_value),
        "total_value": str(s.total_value),
        "position_count": s.position_count,
        "realized_pnl": str(s.realized_pnl),
        "unrealized_pnl": str(s.unrealized_pnl),
        "total_fee": str(s.total_fee),
        "total_dividend": str(s.total_dividend),
        "total_dividend_tax": str(s.total_dividend_tax),
        "total_deposit": str(s.total_deposit),
        "total_withdraw": str(s.total_withdraw),
        "transactions": s.transactions,
        "is_empty": s.is_empty,
        # 缺现价的标的必须显式列出：它们按 0 计入市值，
        # 静默的话总资产会莫名少一截，用户会以为是自己算错了
        "unpriced_symbols": list(s.unpriced_symbols),
        "message": s.message,
        "ledger_path": str(account.store.path),
    }


@router.get("/positions")
async def positions(state: AuthDep) -> dict[str, Any]:
    """持仓明细，含批次。"""
    account = state.account
    holdings = account.positions()
    prices = _latest_prices(state, list(holdings))
    countdown = account.tax_countdown()
    moment = today()

    rows = []
    for symbol, pos in holdings.items():
        price = prices.get(symbol)
        market_value = price * pos.qty if price is not None else None
        cost_total = pos.cost_basis_avg * pos.qty
        rows.append(
            {
                "symbol": str(symbol),
                "qty": pos.qty,
                # T+1：可卖量与持仓量不是一回事，界面必须分开显示，
                # 否则用户会按持仓量下卖单然后被券商拒单
                "available_qty": pos.available_qty,
                "frozen_qty": pos.frozen_qty,
                "cost_basis_avg": str(pos.cost_basis_avg),
                "cost_basis_tax": str(pos.cost_basis_tax),
                "cost_total": str(cost_total),
                "market_price": str(price) if price is not None else None,
                "market_value": str(market_value) if market_value is not None else None,
                "unrealized_pnl": str(market_value - cost_total)
                if market_value is not None
                else None,
                "unrealized_pnl_pct": float((market_value - cost_total) / cost_total)
                if market_value is not None and cost_total
                else None,
                "first_open_date": pos.first_open_date.isoformat(),
                "last_trade_date": pos.last_trade_date.isoformat(),
                "holding_days": pos.holding_days(moment),
                "days_to_tax_free": countdown.get(symbol),
                "realized_pnl": str(pos.realized_pnl),
                "total_dividend": str(pos.total_dividend),
                "total_fee": str(pos.total_fee),
                # 批次级明细。只存平均成本算不出红利税分档与免税倒计时
                "lots": [
                    {
                        "lot_id": lot.lot_id,
                        "open_date": lot.open_date.isoformat(),
                        "original_qty": lot.original_qty,
                        "remaining_qty": lot.remaining_qty,
                        "cost_price": str(lot.cost_price),
                        "accrued_dividend": str(lot.accrued_dividend),
                    }
                    for lot in pos.lots
                ],
            }
        )
    # 按市值降序：仓位最重的排最前，这是看持仓表时最先想知道的
    rows.sort(key=lambda r: Decimal(str(r["market_value"] or "0")), reverse=True)
    return {"positions": rows, "count": len(rows)}


@router.get("/transactions")
async def transactions(state: AuthDep, limit: int = 200) -> dict[str, Any]:
    """资金与成交流水，按时间倒序。"""
    records = state.account.transactions(limit=limit)
    return {
        "count": len(records),
        "transactions": [
            {
                "txn_id": t.txn_id,
                "txn_type": t.txn_type.value,
                "trade_date": t.trade_date.isoformat(),
                "occurred_at": t.occurred_at.isoformat(),
                "symbol": str(t.symbol) if t.symbol else None,
                "qty": t.qty,
                "price": str(t.price),
                "amount": str(t.amount),
                "net_cash": str(t.net_cash),
                "total_fee": str(t.total_fee),
                "source": t.source.value,
                "plan_id": str(t.plan_id) if t.plan_id else None,
                "note": t.note,
            }
            for t in records
        ],
    }


@router.post("/trade")
async def record_trade(body: TradeRequest, state: WriteDep) -> dict[str, Any]:
    """录入一笔成交。"""
    txn = state.account.trade(
        symbol=Symbol(body.symbol),
        side=body.side,
        qty=body.qty,
        price=money(body.price),
        trade_date=dt.date.fromisoformat(body.trade_date) if body.trade_date else None,
        commission=money(body.commission),
        stamp_tax=money(body.stamp_tax),
        transfer_fee=money(body.transfer_fee),
        note=body.note,
    )
    return {"txn_id": txn.txn_id, "net_cash": str(txn.net_cash)}


@router.post("/deposit")
async def deposit(body: CashRequest, state: WriteDep) -> dict[str, Any]:
    """入金。"""
    txn = state.account.deposit(
        money(body.amount),
        trade_date=dt.date.fromisoformat(body.trade_date) if body.trade_date else None,
        note=body.note,
    )
    return {"txn_id": txn.txn_id, "net_cash": str(txn.net_cash)}


@router.post("/withdraw")
async def withdraw(body: CashRequest, state: WriteDep) -> dict[str, Any]:
    """出金。"""
    txn = state.account.withdraw(
        money(body.amount),
        trade_date=dt.date.fromisoformat(body.trade_date) if body.trade_date else None,
        note=body.note,
    )
    return {"txn_id": txn.txn_id, "net_cash": str(txn.net_cash)}


@router.post("/import")
async def import_transactions(body: ImportRequest, state: WriteDep) -> dict[str, Any]:
    """批量导入流水（券商对账单）。

    **整批成功或整批失败**：导入一半的对账单比完全没导入更难收拾。
    """
    written = state.account.import_transactions(body.rows)
    return {"written": written, "submitted": len(body.rows)}
