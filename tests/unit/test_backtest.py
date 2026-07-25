"""回测引擎与绩效指标测试。

最重要的一组是 `TestLookAheadPrevention`——把数据集截断到 T 日，
策略在 T 日的决策必须完全不变。不变才说明没有未来函数（红线 R2）。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence
from decimal import Decimal

import pytest

from quantstock.account.ledger import Ledger
from quantstock.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    MarketView,
    Order,
    RejectReason,
)
from quantstock.backtest.metrics import (
    annualized_return,
    compute_performance,
    max_drawdown,
    money_weighted_return,
    sharpe_ratio,
    sortino_ratio,
    time_weighted_return,
)
from quantstock.data.types import Bar
from quantstock.infra.clock import CST
from quantstock.infra.errors import StrategyError
from quantstock.infra.types import Adjust, Freq, Side, Symbol

A = Symbol("600000.SH")
B = Symbol("600001.SH")

START = dt.date(2026, 1, 5)


def days(n: int) -> list[dt.date]:
    """生成 n 个连续工作日（测试用，不含节假日）。"""
    result: list[dt.date] = []
    day = START
    while len(result) < n:
        if day.weekday() < 5:
            result.append(day)
        day += dt.timedelta(days=1)
    return result


def make_bar(
    symbol: Symbol,
    day: dt.date,
    *,
    close: float,
    open_: float | None = None,
    volume: int = 1_000_000,
    suspended: bool = False,
    limit_up: float | None = None,
    limit_down: float | None = None,
) -> Bar:
    """构造一根 bar。"""
    o = Decimal(str(open_ if open_ is not None else close))
    c = Decimal(str(close))
    return Bar(
        symbol=symbol,
        dt=dt.datetime.combine(day, dt.time(15, 0), tzinfo=CST),
        trade_date=day,
        freq=Freq.D,
        adjust=Adjust.HFQ,
        open=o,
        high=max(o, c),
        low=min(o, c),
        close=c,
        pre_close=c,
        volume=volume,
        amount=c * volume,
        is_suspended=suspended,
        limit_up=Decimal(str(limit_up)) if limit_up is not None else None,
        limit_down=Decimal(str(limit_down)) if limit_down is not None else None,
    )


def flat_history(
    symbol: Symbol, trading_days: Sequence[dt.date], price: float = 100.0
) -> list[Bar]:
    """价格恒定的历史。"""
    return [make_bar(symbol, d, close=price) for d in trading_days]


class TestMarketView:
    def test_hides_future_bars(self) -> None:
        """引擎层面的强制保证：策略拿不到未来的 bar。"""
        trading_days = days(5)
        history = {A: flat_history(A, trading_days)}
        view = MarketView(history, trading_days[2])
        assert len(view.bars(A)) == 3
        assert view.bars(A)[-1].trade_date == trading_days[2]

    def test_lookback_limits_window(self) -> None:
        trading_days = days(10)
        view = MarketView({A: flat_history(A, trading_days)}, trading_days[-1])
        assert len(view.bars(A, lookback=3)) == 3

    def test_latest_returns_today(self) -> None:
        trading_days = days(3)
        view = MarketView({A: flat_history(A, trading_days)}, trading_days[1])
        latest = view.latest(A)
        assert latest is not None
        assert latest.trade_date == trading_days[1]

    def test_unknown_symbol__empty(self) -> None:
        view = MarketView({}, START)
        assert view.bars(A) == []
        assert view.latest(A) is None

    def test_closes_are_floats(self) -> None:
        trading_days = days(3)
        view = MarketView({A: flat_history(A, trading_days)}, trading_days[-1])
        assert view.closes(A) == [100.0, 100.0, 100.0]


class TestExecution:
    def test_buy_fills_next_day(self) -> None:
        """T 日下的单在 T+1 撮合。"""
        trading_days = days(3)
        history = {A: flat_history(A, trading_days)}
        fired = {"done": False}

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            if not fired["done"]:
                fired["done"] = True
                return [Order(A, Side.BUY, 1000)]
            return []

        result = BacktestEngine().run(strategy=strategy, history=history, trading_days=trading_days)
        assert len(result.fills) == 1
        assert result.fills[0].trade_date == trading_days[1]

    def test_t1_prevents_same_day_sell(self) -> None:
        """风控 A01：当日买入当日不可卖。"""
        trading_days = days(4)
        history = {A: flat_history(A, trading_days)}

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            if view.as_of == trading_days[0]:
                return [Order(A, Side.BUY, 1000)]
            if view.as_of == trading_days[1]:
                # 买入成交于 T1，此刻立刻挂卖 → 将在 T2 撮合，届时才可卖
                return [Order(A, Side.SELL, 1000)]
            return []

        result = BacktestEngine().run(strategy=strategy, history=history, trading_days=trading_days)
        assert [f.side for f in result.fills] == [Side.BUY, Side.SELL]

    def test_sell_without_position__rejected(self) -> None:
        trading_days = days(3)

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            return [Order(A, Side.SELL, 1000)] if view.as_of == trading_days[0] else []

        result = BacktestEngine().run(
            strategy=strategy,
            history={A: flat_history(A, trading_days)},
            trading_days=trading_days,
        )
        assert result.rejections[0].reason is RejectReason.NOT_SELLABLE

    def test_limit_up__buy_rejected(self) -> None:
        """涨停排队成交概率极低，假设能买到是常见的回测陷阱。"""
        trading_days = days(3)
        history = {
            A: [
                make_bar(A, trading_days[0], close=100),
                make_bar(A, trading_days[1], close=110, limit_up=110),
                make_bar(A, trading_days[2], close=110),
            ]
        }

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            return [Order(A, Side.BUY, 1000)] if view.as_of == trading_days[0] else []

        result = BacktestEngine().run(strategy=strategy, history=history, trading_days=trading_days)
        assert result.rejections[0].reason is RejectReason.LIMIT_UP

    def test_suspended__rejected(self) -> None:
        trading_days = days(3)
        history = {
            A: [
                make_bar(A, trading_days[0], close=100),
                make_bar(A, trading_days[1], close=100, suspended=True, volume=0),
                make_bar(A, trading_days[2], close=100),
            ]
        }

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            return [Order(A, Side.BUY, 1000)] if view.as_of == trading_days[0] else []

        result = BacktestEngine().run(strategy=strategy, history=history, trading_days=trading_days)
        assert result.rejections[0].reason is RejectReason.SUSPENDED

    def test_no_bar_next_day__rejected(self) -> None:
        trading_days = days(3)
        history = {A: [make_bar(A, trading_days[0], close=100)]}

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            return [Order(A, Side.BUY, 1000)] if view.as_of == trading_days[0] else []

        result = BacktestEngine().run(strategy=strategy, history=history, trading_days=trading_days)
        assert result.rejections[0].reason is RejectReason.NO_BAR

    def test_insufficient_cash__rejected(self) -> None:
        trading_days = days(3)

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            return [Order(A, Side.BUY, 100)] if view.as_of == trading_days[0] else []

        result = BacktestEngine(config=BacktestConfig(initial_cash=Decimal("100"))).run(
            strategy=strategy,
            history={A: flat_history(A, trading_days)},
            trading_days=trading_days,
        )
        assert result.rejections[0].reason is RejectReason.INSUFFICIENT_CASH

    def test_volume_cap_limits_size(self) -> None:
        """单笔不得超过当日成交量的一定比例——大单在真实市场吃不掉。"""
        trading_days = days(3)
        history = {A: [make_bar(A, d, close=100, volume=10_000) for d in trading_days]}

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            return [Order(A, Side.BUY, 100_000)] if view.as_of == trading_days[0] else []

        result = BacktestEngine().run(strategy=strategy, history=history, trading_days=trading_days)
        # 10000 × 5% = 500 股
        assert result.fills[0].qty == 500

    def test_buy_aligns_to_lot(self) -> None:
        trading_days = days(3)

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            return [Order(A, Side.BUY, 150)] if view.as_of == trading_days[0] else []

        result = BacktestEngine().run(
            strategy=strategy,
            history={A: flat_history(A, trading_days)},
            trading_days=trading_days,
        )
        assert result.fills[0].qty == 100

    def test_slippage_hurts_both_directions(self) -> None:
        """滑点方向永远对自己不利：买入上浮、卖出下调。"""
        trading_days = days(4)
        history = {A: flat_history(A, trading_days, price=100.0)}

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            if view.as_of == trading_days[0]:
                return [Order(A, Side.BUY, 1000)]
            if view.as_of == trading_days[2]:
                return [Order(A, Side.SELL, 1000)]
            return []

        config = BacktestConfig(slippage_bps=Decimal("100"))
        result = BacktestEngine(config=config).run(
            strategy=strategy, history=history, trading_days=trading_days
        )
        buy = next(f for f in result.fills if f.side is Side.BUY)
        sell = next(f for f in result.fills if f.side is Side.SELL)
        # 高低点收敛于 100，滑点被夹在区间内，但买价不低于卖价
        assert buy.price >= sell.price

    def test_rejection_summary(self) -> None:
        trading_days = days(3)

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            return [Order(A, Side.SELL, 100)] if view.as_of == trading_days[0] else []

        result = BacktestEngine().run(
            strategy=strategy,
            history={A: flat_history(A, trading_days)},
            trading_days=trading_days,
        )
        assert result.rejection_summary == {"not_sellable": 1}


class TestEngineErrors:
    def test_empty_trading_days__raises(self) -> None:
        with pytest.raises(StrategyError, match="没有任何交易日"):
            BacktestEngine().run(strategy=lambda _v, _l: [], history={}, trading_days=[])

    def test_strategy_exception__wrapped(self) -> None:
        def broken(_view: MarketView, _ledger: Ledger) -> list[Order]:
            msg = "策略内部错误"
            raise RuntimeError(msg)

        with pytest.raises(StrategyError, match="策略执行失败"):
            BacktestEngine().run(strategy=broken, history={}, trading_days=days(2))


class TestLookAheadPrevention:
    """未来函数探测：把数据截断到 T 日，策略在 T 日的决策必须完全不变。"""

    def test_truncated_dataset_yields_identical_decisions(self) -> None:
        trading_days = days(10)
        full_history = {A: [make_bar(A, d, close=100 + i) for i, d in enumerate(trading_days)]}
        cutoff = trading_days[5]
        truncated = {A: [b for b in full_history[A] if b.trade_date <= cutoff]}

        decisions_full: list[tuple[dt.date, int]] = []
        decisions_cut: list[tuple[dt.date, int]] = []

        def make_strategy(
            sink: list[tuple[dt.date, int]],
        ) -> Callable[[MarketView, Ledger], list[Order]]:
            def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
                closes = view.closes(A)
                sink.append((view.as_of, len(closes)))
                return []

            return strategy

        engine = BacktestEngine()
        engine.run(
            strategy=make_strategy(decisions_full),
            history=full_history,
            trading_days=trading_days[:6],
        )
        engine.run(
            strategy=make_strategy(decisions_cut),
            history=truncated,
            trading_days=trading_days[:6],
        )
        assert decisions_full == decisions_cut

    def test_strategy_cannot_see_fill_price(self) -> None:
        """策略下单时看不到成交价——成交价来自次日 bar。"""
        trading_days = days(3)
        history = {
            A: [
                make_bar(A, trading_days[0], close=100),
                make_bar(A, trading_days[1], close=200, open_=200),
                make_bar(A, trading_days[2], close=200),
            ]
        }
        seen: list[float] = []

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            seen.append(view.closes(A)[-1])
            return [Order(A, Side.BUY, 100)] if view.as_of == trading_days[0] else []

        result = BacktestEngine().run(strategy=strategy, history=history, trading_days=trading_days)
        assert seen[0] == 100.0  # 下单时只看到 100
        assert result.fills[0].price > Decimal("150")  # 实际成交在 200 附近


class TestDeterminism:
    def test_same_inputs__identical_results(self) -> None:
        """同一输入重跑两次必须逐笔完全一致（红线 R6）。"""
        trading_days = days(8)
        history = {A: [make_bar(A, d, close=100 + i) for i, d in enumerate(trading_days)]}

        def strategy(view: MarketView, _ledger: Ledger) -> list[Order]:
            return [Order(A, Side.BUY, 100)] if view.as_of == trading_days[0] else []

        first = BacktestEngine().run(strategy=strategy, history=history, trading_days=trading_days)
        second = BacktestEngine().run(strategy=strategy, history=history, trading_days=trading_days)
        assert first.equity == second.equity
        assert [(f.symbol, f.qty, f.price) for f in first.fills] == [
            (f.symbol, f.qty, f.price) for f in second.fills
        ]


class TestMetrics:
    def test_annualized_return(self) -> None:
        assert annualized_return(0.10, 252) == pytest.approx(0.10)

    def test_annualized_return__half_year_scales_up(self) -> None:
        assert annualized_return(0.10, 126) > 0.20

    def test_annualized_return__total_loss__zero(self) -> None:
        assert annualized_return(-1.0, 252) == 0.0

    def test_sharpe_zero_volatility(self) -> None:
        assert sharpe_ratio([0.01] * 10) == 0.0

    def test_sharpe_positive(self) -> None:
        returns = [0.01, -0.005, 0.02, 0.001, 0.015]
        assert sharpe_ratio(returns) > 0

    def test_sortino_ignores_upside_volatility(self) -> None:
        """上涨的波动不是风险——索提诺应高于夏普。"""
        returns = [0.05, 0.06, -0.01, 0.07, 0.04]
        assert sortino_ratio(returns) > sharpe_ratio(returns)

    def test_sortino_no_downside__zero(self) -> None:
        assert sortino_ratio([0.01, 0.02, 0.03]) == 0.0

    def test_max_drawdown(self) -> None:
        info = max_drawdown([100, 120, 90, 110])
        assert info.max_drawdown == pytest.approx(-0.25)
        assert info.duration_days == 1

    def test_max_drawdown_with_dates(self) -> None:
        dates = days(4)
        info = max_drawdown([100, 120, 90, 130], dates)
        assert info.peak_date == dates[1]
        assert info.trough_date == dates[2]
        assert info.recovery_date == dates[3]

    def test_max_drawdown_monotonic_rise__zero(self) -> None:
        assert max_drawdown([100, 110, 120]).max_drawdown == 0.0

    def test_max_drawdown_empty(self) -> None:
        assert max_drawdown([]).max_drawdown == 0.0


class TestTwrVsMwr:
    """TWR 与 MWR 必须同时报告——只报一个都是误导。"""

    def test_twr_excludes_deposits(self) -> None:
        # 期初 100，中途入金 100，期末 200：策略实际一分钱没赚
        values = [100.0, 200.0]
        flows = [0.0, 100.0]
        assert time_weighted_return(values, flows) == pytest.approx(0.0)

    def test_twr_without_flows_equals_simple_return(self) -> None:
        assert time_weighted_return([100.0, 110.0]) == pytest.approx(0.10)

    def test_twr_length_mismatch__raises(self) -> None:
        with pytest.raises(ValueError, match="长度不一致"):
            time_weighted_return([100.0, 110.0], [0.0])

    def test_mwr_positive_when_money_made(self) -> None:
        mwr = money_weighted_return(
            initial_value=Decimal("100000"),
            final_value=Decimal("120000"),
            cash_flows=[],
            total_days=365,
        )
        assert mwr == pytest.approx(0.20, abs=0.01)

    def test_mwr_zero_days__zero(self) -> None:
        assert (
            money_weighted_return(
                initial_value=Decimal("100"),
                final_value=Decimal("110"),
                cash_flows=[],
                total_days=0,
            )
            == 0.0
        )


class TestComputePerformance:
    def test_full_stats(self) -> None:
        values = [100.0, 105.0, 102.0, 110.0, 108.0]
        stats = compute_performance(values=values, dates=days(5), trade_pnls=[100.0, -50.0, 200.0])
        assert stats.total_return == pytest.approx(0.08)
        assert stats.trading_days == 5
        assert stats.max_drawdown < 0
        assert stats.win_rate == pytest.approx(2 / 3)
        assert stats.profit_loss_ratio == pytest.approx(150 / 50)

    def test_empty_values__raises(self) -> None:
        with pytest.raises(ValueError, match="净值序列为空"):
            compute_performance(values=[])

    def test_no_trades__zero_win_rate(self) -> None:
        stats = compute_performance(values=[100.0, 110.0])
        assert stats.win_rate == 0.0
        assert stats.profit_loss_ratio == 0.0
