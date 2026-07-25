"""事件驱动回测引擎。

规范见 docs/03-功能规格.md F6.1。

**PIT 由引擎强制保证，不依赖策略自觉**：
- 策略在 T 日只能看到 T 日及之前的 bar；
- 订单在 **T+1** 撮合，且撮合价来自 T+1 的 bar——策略下单时并不知道这个价格。

撮合遵守 A股 真实约束：涨停买不到、跌停卖不掉、停牌不可交易、T+1 不可当日卖出、
单笔成交量不超过当日成交量的一定比例。**默认取最保守假设**——
回测乐观一分，实盘就会失望十分。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from quantstock.account.ledger import Ledger
from quantstock.account.types import Transaction, TxnSource, TxnType
from quantstock.backtest.metrics import PerformanceStats, compute_performance
from quantstock.data.types import Bar
from quantstock.infra.clock import CST
from quantstock.infra.errors import StrategyError
from quantstock.infra.logging import get_logger
from quantstock.infra.money import align_lot, quantize_cny
from quantstock.infra.types import AccountId, AssetType, Money, Side, Symbol, TradeDate
from quantstock.risk.costs import CostModel

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "MarketView",
    "Order",
    "RejectReason",
]

_log = get_logger(__name__)

_DEFAULT_ACCOUNT = AccountId("backtest")


class RejectReason(StrEnum):
    """订单未成交的原因。

    逐条记录而非笼统丢弃——回测报告里"为什么这笔没成交"往往比收益率更有信息量。
    """

    NO_BAR = "no_bar"
    """次日无行情数据。"""
    SUSPENDED = "suspended"
    """停牌（风控 A04）。"""
    LIMIT_UP = "limit_up"
    """涨停无法买入（风控 A02）。"""
    LIMIT_DOWN = "limit_down"
    """跌停无法卖出。"""
    INSUFFICIENT_CASH = "insufficient_cash"
    NOT_SELLABLE = "not_sellable"
    """T+1 未到可卖日或持仓不足（风控 A01）。"""
    BELOW_LOT = "below_lot"
    """不足一手。"""
    VOLUME_CAP = "volume_cap"
    """超过当日成交量占比上限，已缩量或拒单。"""


@dataclass(frozen=True, slots=True)
class Order:
    """回测订单。T 日生成，T+1 撮合。"""

    symbol: Symbol
    side: Side
    qty: int
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Fill:
    """成交记录。"""

    symbol: Symbol
    side: Side
    qty: int
    price: Money
    trade_date: TradeDate
    fee: Money


@dataclass(frozen=True, slots=True)
class Rejection:
    """被拒订单。"""

    symbol: Symbol
    side: Side
    qty: int
    trade_date: TradeDate
    reason: RejectReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """回测参数。

    默认值全部取**保守**一侧：次日开盘价成交、涨跌停完全无法成交、
    单笔不超过当日成交量 5%。宁可低估收益，也不要高估。
    """

    initial_cash: Money = Decimal("1000000")
    fill_price: str = "next_open"
    """``next_open`` 次日开盘价 / ``next_close`` 次日收盘价。"""
    slippage_bps: Decimal = Decimal("5")
    """滑点（基点），买入上浮、卖出下调。"""
    max_volume_pct: Decimal = Decimal("0.05")
    """单笔成交量占当日成交量的上限。"""
    allow_limit_up_buy: bool = False
    """涨停是否可买入。默认 False——排队成交概率极低，假设能买到是常见的回测陷阱。"""
    allow_limit_down_sell: bool = False
    cost_model: CostModel = field(default_factory=CostModel)
    lot_size: int = 100


class MarketView:
    """策略可见的市场数据视图。

    **只暴露 ``as_of`` 及之前的数据**——策略拿不到未来的 bar，
    这是引擎层面的强制保证，不依赖策略自觉遵守（红线 R2）。
    """

    def __init__(self, history: dict[Symbol, list[Bar]], as_of: TradeDate) -> None:
        """初始化。

        Args:
            history: 各标的的完整历史（已按日期升序）。
            as_of: 当前交易日。
        """
        self._history = history
        self._as_of = as_of

    @property
    def as_of(self) -> TradeDate:
        """当前交易日。"""
        return self._as_of

    def bars(self, symbol: Symbol, *, lookback: int | None = None) -> list[Bar]:
        """取某标的截至今日（含）的历史 bar。

        Args:
            symbol: 标的。
            lookback: 只取最近若干根；None 表示全部。

        Returns:
            按日期升序的 bar 列表。
        """
        visible = [b for b in self._history.get(symbol, ()) if b.trade_date <= self._as_of]
        if lookback is not None:
            return visible[-lookback:]
        return visible

    def closes(self, symbol: Symbol, *, lookback: int | None = None) -> list[float]:
        """取收盘价序列，供技术因子直接使用。

        Args:
            symbol: 标的。
            lookback: 只取最近若干个。

        Returns:
            收盘价列表。
        """
        return [float(b.close) for b in self.bars(symbol, lookback=lookback)]

    def latest(self, symbol: Symbol) -> Bar | None:
        """取今日 bar。

        Args:
            symbol: 标的。

        Returns:
            今日 bar；当日无数据时返回 None。
        """
        visible = self.bars(symbol)
        return visible[-1] if visible else None

    @property
    def symbols(self) -> tuple[Symbol, ...]:
        """今日有数据的标的。"""
        return tuple(sorted(s for s in self._history if self.latest(s) is not None))


StrategyFn = Callable[[MarketView, Ledger], Sequence[Order]]
"""策略函数：给定市场视图与当前账本，返回订单列表。"""


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """回测结果。"""

    dates: tuple[TradeDate, ...]
    equity: tuple[float, ...]
    fills: tuple[Fill, ...]
    rejections: tuple[Rejection, ...]
    stats: PerformanceStats
    ledger: Ledger

    @property
    def rejection_summary(self) -> dict[str, int]:
        """各类拒单原因的计数，用于诊断"为什么策略没跑起来"。"""
        summary: dict[str, int] = {}
        for rejection in self.rejections:
            summary[rejection.reason.value] = summary.get(rejection.reason.value, 0) + 1
        return summary


class BacktestEngine:
    """事件驱动回测引擎。"""

    def __init__(
        self,
        *,
        config: BacktestConfig | None = None,
        account_id: AccountId = _DEFAULT_ACCOUNT,
    ) -> None:
        """初始化。

        Args:
            config: 回测参数。
            account_id: 账户标识。
        """
        self._config = config or BacktestConfig()
        self._account_id = account_id

    def run(
        self,
        *,
        strategy: StrategyFn,
        history: dict[Symbol, list[Bar]],
        trading_days: Sequence[TradeDate],
    ) -> BacktestResult:
        """执行回测。

        每个交易日的顺序：
        1. 用**昨日**生成的订单在今日撮合（T+1）；
        2. 结算当日分红；
        3. 记录当日净值；
        4. 让策略看**截至今日**的数据，生成明日的订单。

        这个顺序保证策略永远看不到自己下单后的成交价。

        Args:
            strategy: 策略函数。
            history: 各标的的历史 bar，须按日期升序。
            trading_days: 回测区间的交易日，须升序。

        Returns:
            回测结果。

        Raises:
            StrategyError: 交易日为空或策略抛出异常。
        """
        if not trading_days:
            msg = "回测区间没有任何交易日"
            raise StrategyError(msg)

        ledger = Ledger(self._account_id)
        ledger.apply(
            Transaction(
                txn_id="init-cash",
                account_id=self._account_id,
                txn_type=TxnType.DEPOSIT,
                trade_date=trading_days[0],
                occurred_at=_moment(trading_days[0]),
                net_cash=self._config.initial_cash,
                source=TxnSource.ADJUST,
                note="回测初始资金",
            )
        )

        bar_index = {(bar.symbol, bar.trade_date): bar for bars in history.values() for bar in bars}

        pending: list[Order] = []
        fills: list[Fill] = []
        rejections: list[Rejection] = []
        equity: list[float] = []
        seq = 0

        for today in trading_days:
            # 1. 昨日订单在今日撮合
            for order in pending:
                seq += 1
                outcome = self._execute(
                    order=order,
                    today=today,
                    bar=bar_index.get((order.symbol, today)),
                    ledger=ledger,
                    seq=seq,
                )
                if isinstance(outcome, Fill):
                    fills.append(outcome)
                else:
                    rejections.append(outcome)
            pending = []

            # 2. 记录当日净值（用当日收盘价估值）
            prices = {
                sym: bar.close
                for sym in ledger.positions()
                if (bar := bar_index.get((sym, today))) is not None
            }
            equity.append(float(ledger.state(as_of=today).total_value(prices)))

            # 3. 策略看截至今日的数据，生成明日订单
            view = MarketView(history, today)
            try:
                pending = list(strategy(view, ledger))
            except Exception as exc:
                msg = "策略执行失败"
                raise StrategyError(msg, trade_date=str(today), error=str(exc)) from exc

        stats = compute_performance(
            values=equity,
            dates=list(trading_days),
            trade_pnls=[float(c.realized_pnl) for c in ledger.consumptions],
        )
        _log.info(
            "backtest_finished",
            days=len(trading_days),
            fills=len(fills),
            rejections=len(rejections),
            total_return=round(stats.total_return, 4),
        )
        return BacktestResult(
            dates=tuple(trading_days),
            equity=tuple(equity),
            fills=tuple(fills),
            rejections=tuple(rejections),
            stats=stats,
            ledger=ledger,
        )

    def _execute(  # noqa: C901, PLR0911 - 撮合是一串平铺的 A股 约束判定，拆开反而看不清顺序
        self,
        *,
        order: Order,
        today: TradeDate,
        bar: Bar | None,
        ledger: Ledger,
        seq: int,
    ) -> Fill | Rejection:
        """在今日撮合一笔订单。

        Args:
            order: 待撮合订单。
            today: 今日。
            bar: 今日该标的的 bar。
            ledger: 账本。
            seq: 流水序号，用于生成唯一 txn_id。

        Returns:
            成交或拒单记录。
        """
        cfg = self._config

        def reject(reason: RejectReason, detail: str = "") -> Rejection:
            return Rejection(
                symbol=order.symbol,
                side=order.side,
                qty=order.qty,
                trade_date=today,
                reason=reason,
                detail=detail,
            )

        if bar is None:
            return reject(RejectReason.NO_BAR, "次日无行情数据")
        if bar.is_suspended or bar.volume <= 0:
            return reject(RejectReason.SUSPENDED, "停牌或无成交")
        if order.side is Side.BUY and bar.is_limit_up and not cfg.allow_limit_up_buy:
            return reject(RejectReason.LIMIT_UP, "涨停，排队成交概率极低")
        if order.side is Side.SELL and bar.is_limit_down and not cfg.allow_limit_down_sell:
            return reject(RejectReason.LIMIT_DOWN, "跌停，无法卖出")

        price = self._fill_price(bar, order.side)
        qty = order.qty

        # 成交量约束：单笔不得超过当日成交量的一定比例
        volume_cap = int(Decimal(bar.volume) * cfg.max_volume_pct)
        if qty > volume_cap:
            qty = volume_cap
            if qty <= 0:
                return reject(RejectReason.VOLUME_CAP, f"当日成交量 {bar.volume} 过低")

        if order.side is Side.BUY:
            qty = align_lot(qty, lot_size=cfg.lot_size)
            if qty <= 0:
                return reject(RejectReason.BELOW_LOT, "不足一手")
            affordable = cfg.cost_model.max_affordable_qty(
                cash=ledger.cash, price=price, trade_date=today, lot_size=cfg.lot_size
            )
            qty = min(qty, affordable)
            if qty <= 0:
                return reject(RejectReason.INSUFFICIENT_CASH, f"可用资金 {ledger.cash} 不足一手")
        else:
            position = ledger.position(order.symbol, as_of=today)
            available = position.available_qty if position else 0
            qty = min(qty, available)
            if qty <= 0:
                return reject(RejectReason.NOT_SELLABLE, "T+1 未到可卖日或无持仓")

        amount = quantize_cny(price * qty)
        fees = cfg.cost_model.compute(
            amount=amount,
            side=order.side,
            trade_date=today,
            asset_type=AssetType.STOCK,
        )
        net_cash = -(amount + fees.total) if order.side is Side.BUY else amount - fees.total

        ledger.apply(
            Transaction(
                txn_id=f"bt-{seq:08d}",
                account_id=self._account_id,
                txn_type=TxnType.BUY if order.side is Side.BUY else TxnType.SELL,
                trade_date=today,
                occurred_at=_moment(today),
                symbol=order.symbol,
                qty=qty if order.side is Side.BUY else -qty,
                price=price,
                amount=amount,
                commission=fees.commission,
                stamp_tax=fees.stamp_tax,
                transfer_fee=fees.transfer_fee,
                net_cash=net_cash,
                source=TxnSource.PLAN,
                note=order.reason,
            )
        )
        return Fill(
            symbol=order.symbol,
            side=order.side,
            qty=qty,
            price=price,
            trade_date=today,
            fee=fees.total,
        )

    def _fill_price(self, bar: Bar, side: Side) -> Money:
        """计算含滑点的成交价。

        滑点方向永远对自己不利：买入上浮、卖出下调。

        Args:
            bar: 撮合日的 bar。
            side: 买卖方向。

        Returns:
            成交价。
        """
        base = bar.open if self._config.fill_price == "next_open" else bar.close
        slip = self._config.slippage_bps / Decimal("10000")
        adjusted = base * (1 + slip) if side is Side.BUY else base * (1 - slip)
        # 成交价不能超出当日高低点——否则等于凭空造出一个不存在的价格
        return max(bar.low, min(bar.high, adjusted))


def _moment(day: TradeDate) -> dt.datetime:
    """构造该交易日的收盘时刻。

    Args:
        day: 交易日。

    Returns:
        tz-aware 的时刻。
    """
    return dt.datetime.combine(day, dt.time(15, 0), tzinfo=CST)
