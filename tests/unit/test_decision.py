"""决策层测试：策略、组合构建、风控引擎。

重点覆盖三条"错了会亏钱"的规则：
- 择时系数只降不升（SHORT 层的不对称设计）
- 缓冲带防止为几百块的偏离反复付手续费
- HALTED 状态禁止一切买入且不会自动恢复
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from quantstock.account.types import Lot, Position
from quantstock.backtest.engine import MarketView
from quantstock.config.models import CircuitBreakerConfig
from quantstock.data.types import Bar
from quantstock.infra.clock import CST
from quantstock.infra.errors import RiskRejectedError
from quantstock.infra.types import (
    AccountId,
    Adjust,
    Direction,
    Freq,
    Horizon,
    Side,
    Symbol,
)
from quantstock.portfolio.builder import (
    PortfolioConstraints,
    RebalanceOrder,
    TargetPosition,
    build_targets,
    diff_to_orders,
)
from quantstock.risk.engine import (
    CircuitState,
    MarketSnapshot,
    RiskEngine,
    Severity,
)
from quantstock.strategy.builtin import (
    EtfRotationStrategy,
    MacroExposureStrategy,
    MomentumTrendStrategy,
    TimingOverlayStrategy,
    blend_scores,
)
from quantstock.strategy.types import Evidence, Signal, StrategyContext, TimingSignal

A = Symbol("600000.SH")
B = Symbol("600001.SH")
C = Symbol("510300.SH")
ACC = AccountId("test")
TODAY = dt.date(2026, 7, 24)


def bar(
    symbol: Symbol,
    *,
    close: str,
    limit_up: str | None = None,
    limit_down: str | None = None,
    suspended: bool = False,
) -> Bar:
    """构造一根 bar。"""
    price = Decimal(close)
    return Bar(
        symbol=symbol,
        dt=dt.datetime.combine(TODAY, dt.time(15, 0), tzinfo=CST),
        trade_date=TODAY,
        freq=Freq.D,
        adjust=Adjust.NONE,
        open=price,
        high=price,
        low=price,
        close=price,
        pre_close=price,
        volume=1_000_000,
        is_suspended=suspended,
        limit_up=Decimal(limit_up) if limit_up else None,
        limit_down=Decimal(limit_down) if limit_down else None,
    )


def position(symbol: Symbol, qty: int, *, available: int | None = None) -> Position:
    """构造一个持仓。"""
    return Position(
        account_id=ACC,
        symbol=symbol,
        qty=qty,
        available_qty=qty if available is None else available,
        frozen_qty=0,
        cost_basis_avg=Decimal("100"),
        cost_basis_tax=Decimal("100"),
        first_open_date=dt.date(2026, 1, 5),
        last_trade_date=dt.date(2026, 1, 5),
        lots=(
            Lot(
                lot_id="l1",
                symbol=symbol,
                open_date=dt.date(2026, 1, 5),
                open_txn_id="t1",
                original_qty=qty,
                remaining_qty=qty,
                cost_price=Decimal("100"),
            ),
        ),
    )


def history(symbol: Symbol, prices: list[float]) -> list[Bar]:
    """构造价格序列的历史。"""
    start = dt.date(2026, 1, 5)
    result = []
    day = start
    for price in prices:
        while day.weekday() >= 5:
            day += dt.timedelta(days=1)
        p = Decimal(str(price))
        result.append(
            Bar(
                symbol=symbol,
                dt=dt.datetime.combine(day, dt.time(15, 0), tzinfo=CST),
                trade_date=day,
                freq=Freq.D,
                adjust=Adjust.HFQ,
                open=p,
                high=p,
                low=p,
                close=p,
                pre_close=p,
                volume=1_000_000,
            )
        )
        day += dt.timedelta(days=1)
    return result


def context(hist: dict[Symbol, list[Bar]]) -> StrategyContext:
    """构造策略上下文，as_of 取最后一天。"""
    last = max(b.trade_date for bars in hist.values() for b in bars)
    return StrategyContext(as_of=last, market=MarketView(hist, last), universe=tuple(sorted(hist)))


class TestSignalValidation:
    def test_confidence_out_of_range__raises(self) -> None:
        with pytest.raises(ValueError, match="置信度必须在"):
            Signal(
                symbol=A,
                trade_date=TODAY,
                direction=Direction.LONG,
                score=0.5,
                confidence=1.5,
                horizon=Horizon.MEDIUM,
                strategy_id="x",
                strategy_version="v1",
            )


class TestTimingSignalAsymmetry:
    """SHORT 层只能降权不能加权——做错时只是少赚，做对时能少亏。"""

    def test_coefficient_above_one__rejected(self) -> None:
        with pytest.raises(ValueError, match="只降不升"):
            TimingSignal(symbol=A, trade_date=TODAY, coefficient=Decimal("1.2"))

    def test_coefficient_within_range__ok(self) -> None:
        assert TimingSignal(
            symbol=A, trade_date=TODAY, coefficient=Decimal("0.7")
        ).coefficient == Decimal("0.7")


class TestMomentumTrendStrategy:
    def test_uptrend_ranks_above_downtrend(self) -> None:
        """打分是**横截面**分位，必须有对照组才有意义。"""
        hist = {
            A: history(A, [100.0 + i for i in range(80)]),
            B: history(B, [200.0 - i for i in range(80)]),
        }
        signals = {s.symbol: s for s in MomentumTrendStrategy().generate(context(hist))}
        assert signals[A].score > signals[B].score
        assert signals[A].direction is Direction.LONG
        assert signals[A].evidence

    def test_downtrend_gets_counter_evidence(self) -> None:
        """趋势未确认时必须给出反面证据，而非静默降分。"""
        falling = [200.0 - i for i in range(80)]
        signals = MomentumTrendStrategy().generate(context({A: history(A, falling)}))
        assert signals[0].counter_evidence
        assert "趋势未确认" in signals[0].counter_evidence[0].statement

    def test_insufficient_data__skipped(self) -> None:
        assert MomentumTrendStrategy().generate(context({A: history(A, [100.0] * 5)})) == []

    def test_bad_ma_params__raises(self) -> None:
        with pytest.raises(ValueError, match="短期均线必须短于长期均线"):
            MomentumTrendStrategy(fast_ma=60, slow_ma=20)


class TestEtfRotation:
    def test_picks_strongest(self) -> None:
        hist = {
            A: history(A, [100.0 + i * 2 for i in range(30)]),
            B: history(B, [100.0] * 30),
            C: history(C, [100.0 - i * 0.5 for i in range(30)]),
        }
        signals = {s.symbol: s for s in EtfRotationStrategy(top_n=1).generate(context(hist))}
        assert signals[A].direction is Direction.LONG
        assert signals[C].direction is Direction.FLAT

    def test_negative_momentum__not_selected_even_if_top(self) -> None:
        """全市场下跌时"最强的"仍然是亏钱的，不建仓。"""
        hist = {A: history(A, [100.0 - i for i in range(30)])}
        signals = EtfRotationStrategy(top_n=1).generate(context(hist))
        assert signals[0].direction is Direction.FLAT
        assert signals[0].counter_evidence

    def test_invalid_params__raises(self) -> None:
        with pytest.raises(ValueError, match="必须为正"):
            EtfRotationStrategy(top_n=0)


class TestMacroExposure:
    def test_broad_uptrend__high_exposure(self) -> None:
        hist = {sym: history(sym, [100.0 + i for i in range(140)]) for sym in (A, B, C)}
        signal = MacroExposureStrategy().generate_exposure(context(hist))
        assert signal.target_exposure > Decimal("0.8")

    def test_broad_downtrend__low_exposure(self) -> None:
        hist = {sym: history(sym, [300.0 - i for i in range(140)]) for sym in (A, B, C)}
        signal = MacroExposureStrategy().generate_exposure(context(hist))
        assert signal.target_exposure <= Decimal("0.3")

    def test_insufficient_data__falls_back_to_minimum(self) -> None:
        """不确定时取保守一侧。"""
        signal = MacroExposureStrategy().generate_exposure(context({A: history(A, [100.0] * 10)}))
        assert signal.target_exposure == Decimal("0.2")
        assert "保守" in signal.rationale[0]

    def test_inverted_bounds__raises(self) -> None:
        with pytest.raises(ValueError, match="仓位下限不能高于上限"):
            MacroExposureStrategy(min_exposure=Decimal("0.9"), max_exposure=Decimal("0.5"))


class TestTimingOverlay:
    def test_below_ma__reduces_weight(self) -> None:
        prices = [100.0] * 25 + [80.0]
        result = TimingOverlayStrategy().generate_timing(context({A: history(A, prices)}))
        assert result[A].coefficient < Decimal("1.0")

    def test_healthy_trend__no_intervention(self) -> None:
        prices = [100.0 + i * 0.1 for i in range(30)]
        result = TimingOverlayStrategy().generate_timing(context({A: history(A, prices)}))
        assert result[A].coefficient == Decimal("1.00")


class TestBlendScores:
    def test_weighted_average(self) -> None:
        def sig(symbol: Symbol, score: float, sid: str) -> Signal:
            return Signal(
                symbol=symbol,
                trade_date=TODAY,
                direction=Direction.LONG,
                score=score,
                confidence=0.5,
                horizon=Horizon.MEDIUM,
                strategy_id=sid,
                strategy_version="v1",
            )

        blended = blend_scores(
            {"a": [sig(A, 1.0, "a")], "b": [sig(A, 0.0, "b")]},
            {"a": 0.75, "b": 0.25},
        )
        assert blended[A] == pytest.approx(0.75)

    def test_missing_coverage_not_treated_as_bearish(self) -> None:
        """某策略没覆盖该标的时不参与加权——用 0 填充会把"没覆盖"误判成"看空"。"""

        def sig(symbol: Symbol, score: float, sid: str) -> Signal:
            return Signal(
                symbol=symbol,
                trade_date=TODAY,
                direction=Direction.LONG,
                score=score,
                confidence=0.5,
                horizon=Horizon.MEDIUM,
                strategy_id=sid,
                strategy_version="v1",
            )

        blended = blend_scores(
            {"a": [sig(A, 1.0, "a")], "b": [sig(B, 1.0, "b")]}, {"a": 0.5, "b": 0.5}
        )
        assert blended[A] == pytest.approx(1.0)
        assert blended[B] == pytest.approx(1.0)

    def test_zero_weight_strategy_ignored(self) -> None:
        def sig(score: float, sid: str) -> Signal:
            return Signal(
                symbol=A,
                trade_date=TODAY,
                direction=Direction.LONG,
                score=score,
                confidence=0.5,
                horizon=Horizon.MEDIUM,
                strategy_id=sid,
                strategy_version="v1",
            )

        blended = blend_scores({"a": [sig(1.0, "a")], "b": [sig(0.0, "b")]}, {"a": 1.0})
        assert blended[A] == pytest.approx(1.0)


class TestBuildTargets:
    def test_allocates_by_score(self) -> None:
        # 单票上限设得足够高，避免两只都被截到同一个上限而看不出差异
        targets = build_targets(
            scores={A: 2.0, B: 1.0},
            prices={A: Decimal("100"), B: Decimal("100")},
            total_value=Decimal("1000000"),
            exposure=Decimal("0.9"),
            constraints=PortfolioConstraints(max_single_position=Decimal("0.80")),
        )
        by_symbol = {t.symbol: t for t in targets}
        assert by_symbol[A].target_weight > by_symbol[B].target_weight

    def test_respects_single_position_cap(self) -> None:
        targets = build_targets(
            scores={A: 1.0},
            prices={A: Decimal("100")},
            total_value=Decimal("1000000"),
            exposure=Decimal("0.9"),
            constraints=PortfolioConstraints(max_single_position=Decimal("0.10")),
        )
        assert targets[0].target_weight <= Decimal("0.10")

    def test_respects_industry_cap(self) -> None:
        targets = build_targets(
            scores={A: 1.0, B: 1.0, C: 1.0},
            prices=dict.fromkeys((A, B, C), Decimal("100")),
            total_value=Decimal("1000000"),
            exposure=Decimal("0.9"),
            constraints=PortfolioConstraints(max_industry_exposure=Decimal("0.20")),
            industries=dict.fromkeys((A, B, C), "白酒"),
        )
        assert sum(t.target_weight for t in targets) <= Decimal("0.21")

    def test_timing_only_reduces(self) -> None:
        base = build_targets(
            scores={A: 1.0},
            prices={A: Decimal("100")},
            total_value=Decimal("1000000"),
            exposure=Decimal("0.9"),
            constraints=PortfolioConstraints(),
        )
        reduced = build_targets(
            scores={A: 1.0},
            prices={A: Decimal("100")},
            total_value=Decimal("1000000"),
            exposure=Decimal("0.9"),
            constraints=PortfolioConstraints(),
            timing={A: Decimal("0.5")},
        )
        boosted = build_targets(
            scores={A: 1.0},
            prices={A: Decimal("100")},
            total_value=Decimal("1000000"),
            exposure=Decimal("0.9"),
            constraints=PortfolioConstraints(),
            timing={A: Decimal("2.0")},
        )
        assert reduced[0].target_weight < base[0].target_weight
        assert boosted[0].target_weight == base[0].target_weight

    def test_min_position_value_filter(self) -> None:
        targets = build_targets(
            scores={A: 1.0},
            prices={A: Decimal("100")},
            total_value=Decimal("10000"),
            exposure=Decimal("0.1"),
            constraints=PortfolioConstraints(min_position_value=Decimal("5000")),
        )
        assert targets == []

    def test_max_holdings(self) -> None:
        scores = {Symbol(f"60000{i}.SH"): float(20 - i) for i in range(10)}
        targets = build_targets(
            scores=scores,
            prices=dict.fromkeys(scores, Decimal("10")),
            total_value=Decimal("10000000"),
            exposure=Decimal("0.9"),
            constraints=PortfolioConstraints(max_holdings=3),
        )
        assert len(targets) <= 3

    def test_negative_total_value__raises(self) -> None:
        with pytest.raises(ValueError, match="总资产必须为正"):
            build_targets(
                scores={A: 1.0},
                prices={A: Decimal("100")},
                total_value=Decimal("0"),
                exposure=Decimal("0.9"),
                constraints=PortfolioConstraints(),
            )

    def test_exposure_out_of_range__raises(self) -> None:
        with pytest.raises(ValueError, match="权益仓位必须在"):
            build_targets(
                scores={A: 1.0},
                prices={A: Decimal("100")},
                total_value=Decimal("100000"),
                exposure=Decimal("1.5"),
                constraints=PortfolioConstraints(),
            )


class TestRebalanceBand:
    """缓冲带：否则每天都在为几百块钱的偏离付手续费。"""

    def test_small_drift__skipped_with_reason(self) -> None:
        targets = [
            TargetPosition(
                symbol=A,
                target_weight=Decimal("0.101"),
                target_qty=1010,
                reference_price=Decimal("100"),
            )
        ]
        orders, skipped = diff_to_orders(
            targets=targets,
            positions={A: position(A, 1000)},
            prices={A: Decimal("100")},
            total_value=Decimal("1000000"),
            constraints=PortfolioConstraints(rebalance_band=Decimal("0.02")),
        )
        assert orders == []
        assert "缓冲带" in skipped[0].reason

    def test_large_drift__generates_order(self) -> None:
        targets = [
            TargetPosition(
                symbol=A,
                target_weight=Decimal("0.15"),
                target_qty=1500,
                reference_price=Decimal("100"),
            )
        ]
        orders, _ = diff_to_orders(
            targets=targets,
            positions={A: position(A, 1000)},
            prices={A: Decimal("100")},
            total_value=Decimal("1000000"),
            constraints=PortfolioConstraints(),
        )
        assert orders[0].side is Side.BUY
        assert orders[0].qty == 500


class TestDiffToOrders:
    def test_clears_positions_not_in_target(self) -> None:
        orders, _ = diff_to_orders(
            targets=[],
            positions={A: position(A, 1000)},
            prices={A: Decimal("100")},
            total_value=Decimal("1000000"),
            constraints=PortfolioConstraints(),
        )
        assert orders[0].side is Side.SELL
        assert orders[0].target_qty == 0

    def test_unsellable_position__skipped_with_reason(self) -> None:
        orders, skipped = diff_to_orders(
            targets=[],
            positions={A: position(A, 1000, available=0)},
            prices={A: Decimal("100")},
            total_value=Decimal("1000000"),
            constraints=PortfolioConstraints(),
        )
        assert orders == []
        assert "T+1" in skipped[0].reason

    def test_sells_before_buys(self) -> None:
        """卖出排前面：先释放资金再用，避免"钱还没到账就想买"。"""
        targets = [
            TargetPosition(
                symbol=B,
                target_weight=Decimal("0.15"),
                target_qty=1500,
                reference_price=Decimal("100"),
            )
        ]
        orders, _ = diff_to_orders(
            targets=targets,
            positions={A: position(A, 1000)},
            prices={A: Decimal("100"), B: Decimal("100")},
            total_value=Decimal("1000000"),
            constraints=PortfolioConstraints(),
        )
        assert orders[0].side is Side.SELL
        assert orders[-1].side is Side.BUY


class TestCircuitBreaker:
    @pytest.fixture
    def engine(self) -> RiskEngine:
        return RiskEngine(circuit_config=CircuitBreakerConfig())

    def test_normal_stays_normal(self, engine: RiskEngine) -> None:
        state = engine.evaluate_circuit(daily_return=Decimal("0.01"), drawdown_20d=Decimal("-0.02"))
        assert state is CircuitState.NORMAL

    def test_daily_loss_triggers_watch(self, engine: RiskEngine) -> None:
        state = engine.evaluate_circuit(daily_return=Decimal("-0.04"), drawdown_20d=Decimal("0"))
        assert state is CircuitState.WATCH

    def test_big_loss_triggers_halted(self, engine: RiskEngine) -> None:
        state = engine.evaluate_circuit(daily_return=Decimal("-0.06"), drawdown_20d=Decimal("0"))
        assert state is CircuitState.HALTED

    def test_drawdown_triggers_halted(self, engine: RiskEngine) -> None:
        state = engine.evaluate_circuit(daily_return=Decimal("0"), drawdown_20d=Decimal("-0.20"))
        assert state is CircuitState.HALTED

    def test_halted_never_auto_recovers(self, engine: RiskEngine) -> None:
        """自动恢复会让系统在剧烈波动中反复进出，而那正是最该停手的时候。"""
        state = engine.evaluate_circuit(
            daily_return=Decimal("0.05"),
            drawdown_20d=Decimal("0"),
            current=CircuitState.HALTED,
        )
        assert state is CircuitState.HALTED

    def test_watch_recovers_when_drawdown_converges(self, engine: RiskEngine) -> None:
        state = engine.evaluate_circuit(
            daily_return=Decimal("0.01"),
            drawdown_20d=Decimal("-0.05"),
            current=CircuitState.WATCH,
        )
        assert state is CircuitState.NORMAL


def order(symbol: Symbol, side: Side, qty: int, price: str = "100") -> RebalanceOrder:
    """构造调仓指令。"""
    return RebalanceOrder(
        symbol=symbol,
        side=side,
        qty=qty,
        reference_price=Decimal(price),
        current_qty=0,
        target_qty=qty,
    )


class TestRiskEngine:
    @pytest.fixture
    def engine(self) -> RiskEngine:
        return RiskEngine(constraints=PortfolioConstraints())

    def test_normal_buy_approved(self, engine: RiskEngine) -> None:
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 500)],
            positions={},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100")}),
            trade_date=TODAY,
        )
        assert decision.passed
        assert len(decision.approved) == 1

    def test_limit_up_blocks_buy(self, engine: RiskEngine) -> None:
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 500)],
            positions={},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100", limit_up="100")}),
            trade_date=TODAY,
        )
        assert not decision.passed
        assert "涨停" in decision.rejected[0][1]

    def test_suspended_blocks(self, engine: RiskEngine) -> None:
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 500)],
            positions={},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100", suspended=True)}),
            trade_date=TODAY,
        )
        assert "停牌" in decision.rejected[0][1]

    def test_missing_bar_blocks(self, engine: RiskEngine) -> None:
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 500)],
            positions={},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={}),
            trade_date=TODAY,
        )
        assert "无当日行情" in decision.rejected[0][1]

    def test_single_position_cap_blocks(self, engine: RiskEngine) -> None:
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 2000)],
            positions={},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100")}),
            trade_date=TODAY,
        )
        assert "占比将达" in decision.rejected[0][1]

    def test_rejection_message_contains_numbers(self, engine: RiskEngine) -> None:
        """ "仓位超限"没法让用户判断该怎么办，带数值才可以。"""
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 2000)],
            positions={},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100")}),
            trade_date=TODAY,
        )
        message = decision.rejected[0][1]
        assert "%" in message
        assert "上限" in message

    def test_industry_cap_blocks(self, engine: RiskEngine) -> None:
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 1000), order(B, Side.BUY, 1000)],
            positions={C: position(C, 2500)},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(
                bars={s: bar(s, close="100") for s in (A, B, C)},
                industries=dict.fromkeys((A, B, C), "白酒"),
            ),
            trade_date=TODAY,
        )
        assert any("行业" in reason for _, reason in decision.rejected)

    def test_liquidity_floor_blocks(self, engine: RiskEngine) -> None:
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 500)],
            positions={},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(
                bars={A: bar(A, close="100")},
                avg_amount_20d={A: Decimal("1000000")},
            ),
            trade_date=TODAY,
        )
        assert "均额" in decision.rejected[0][1]

    def test_intel_blacklist_blocks_buy(self, engine: RiskEngine) -> None:
        """情报通路 1：单向否决，只禁买不禁卖。"""
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 500)],
            positions={},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100")}, blacklist=frozenset({A})),
            trade_date=TODAY,
        )
        assert "情报黑名单" in decision.rejected[0][1]

    def test_intel_blacklist_allows_sell(self, engine: RiskEngine) -> None:
        decision = engine.pre_trade_check(
            orders=[order(A, Side.SELL, 500)],
            positions={A: position(A, 1000)},
            cash=Decimal("0"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100")}, blacklist=frozenset({A})),
            trade_date=TODAY,
        )
        assert decision.passed

    def test_insufficient_cash__adjusts_instead_of_blocking(self, engine: RiskEngine) -> None:
        """资金不足时缩量比直接丢弃更有用。"""
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 1000)],
            positions={},
            cash=Decimal("30000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100")}),
            trade_date=TODAY,
        )
        assert decision.passed
        assert decision.approved[0].qty == 300

    def test_sell_more_than_available__adjusted(self, engine: RiskEngine) -> None:
        decision = engine.pre_trade_check(
            orders=[order(A, Side.SELL, 1000)],
            positions={A: position(A, 1000, available=400)},
            cash=Decimal("0"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100")}),
            trade_date=TODAY,
        )
        assert decision.approved[0].qty == 400

    def test_halted_blocks_all_buys(self, engine: RiskEngine) -> None:
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 500)],
            positions={},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100")}),
            trade_date=TODAY,
            circuit_state=CircuitState.HALTED,
        )
        assert "HALTED" in decision.rejected[0][1]

    def test_halted_still_allows_sells(self, engine: RiskEngine) -> None:
        """HALTED 只禁买不禁卖——该出的时候必须能出。"""
        decision = engine.pre_trade_check(
            orders=[order(A, Side.SELL, 500)],
            positions={A: position(A, 1000)},
            cash=Decimal("0"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100")}),
            trade_date=TODAY,
            circuit_state=CircuitState.HALTED,
        )
        assert decision.passed

    def test_watch_blocks_new_positions_only(self, engine: RiskEngine) -> None:
        new_open = order(A, Side.BUY, 500)
        add_on = RebalanceOrder(
            symbol=B,
            side=Side.BUY,
            qty=100,
            reference_price=Decimal("100"),
            current_qty=500,
            target_qty=600,
        )
        decision = engine.pre_trade_check(
            orders=[new_open, add_on],
            positions={B: position(B, 500)},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={s: bar(s, close="100") for s in (A, B)}),
            trade_date=TODAY,
            circuit_state=CircuitState.WATCH,
        )
        assert [o.symbol for o in decision.approved] == [B]
        assert decision.rejected[0][0].symbol == A

    def test_raise_if_rejected(self, engine: RiskEngine) -> None:
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 500)],
            positions={},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100", limit_up="100")}),
            trade_date=TODAY,
        )
        with pytest.raises(RiskRejectedError, match="风控拒绝"):
            decision.raise_if_rejected()

    def test_portfolio_turnover_warning(self, engine: RiskEngine) -> None:
        decision = engine.pre_trade_check(
            orders=[order(A, Side.BUY, 500)],
            positions={},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=MarketSnapshot(bars={A: bar(A, close="100")}),
            trade_date=TODAY,
        )
        assert decision.portfolio_level[0].rule_id == "B09"
        assert decision.portfolio_level[0].severity is Severity.WARN


class TestEvidence:
    def test_carries_numbers_for_explanation(self) -> None:
        """ "动量不错"不是依据，带值和分位的才是。"""
        evidence = Evidence(
            factor="momentum_60d",
            value=0.18,
            rank_pct=0.87,
            contribution=0.42,
            statement="60 日动量 +18.00%，处于全市场 87% 分位",
        )
        assert "87%" in evidence.statement
