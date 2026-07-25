"""建议层测试。

最重要的一组是 `TestRationaleCompleteness`——缺支柱的建议必须被剔除。
最后一组是端到端串联：数据 → 策略 → 组合 → 风控 → 建议。
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from quantstock.account.ledger import Ledger, replay
from quantstock.account.types import Position, Transaction, TxnSource, TxnType
from quantstock.advisor.analytics import build_analytics
from quantstock.advisor.planner import PlanBuilder, compute_param_hash
from quantstock.advisor.types import (
    IntelEvidence,
    IntelImpact,
    PositionAnalytics,
    RationaleBundle,
    TradeIntent,
    Urgency,
)
from quantstock.backtest.engine import MarketView
from quantstock.data.types import Bar
from quantstock.infra.clock import CST
from quantstock.infra.types import (
    AccountId,
    Adjust,
    Freq,
    IntentId,
    Side,
    Symbol,
)
from quantstock.portfolio.builder import (
    PortfolioConstraints,
    RebalanceOrder,
    build_targets,
    diff_to_orders,
)
from quantstock.risk.engine import CircuitState, MarketSnapshot, RiskEngine
from quantstock.strategy.builtin import MomentumTrendStrategy
from quantstock.strategy.types import Evidence, StrategyContext

A = Symbol("600519.SH")
B = Symbol("300750.SZ")
ACC = AccountId("main")
TODAY = dt.date(2026, 7, 24)
BUY_DAY = dt.date(2026, 1, 5)


def evidence(name: str = "momentum_60d") -> Evidence:
    """构造一条量化依据。"""
    return Evidence(
        factor=name,
        value=0.18,
        rank_pct=0.87,
        contribution=0.42,
        statement=f"{name} = 0.18，处于全市场 87% 分位",
    )


def intel_item(*, url: str = "https://example.com/a") -> IntelEvidence:
    """构造一条情报证据。"""
    return IntelEvidence(
        title="控股股东计划减持不超过 1% 股份",
        source="上交所公告",
        published_at=dt.datetime(2026, 7, 23, 18, 2, tzinfo=CST),
        url=url,
        domain="COMPANY",
        sentiment=-0.45,
        importance=78,
        impact=IntelImpact.WEAKEN,
    )


def analytics_with_technicals() -> PositionAnalytics:
    """构造带技术形态的分析结果。"""
    return PositionAnalytics(
        symbol=A,
        as_of=TODAY,
        market_price=Decimal("100"),
        ma20=98.0,
        ma60=102.0,
        ma_alignment="空头排列",
    )


def order(side: Side = Side.SELL, qty: int = 400) -> RebalanceOrder:
    """构造调仓指令。"""
    return RebalanceOrder(
        symbol=A,
        side=side,
        qty=qty,
        reference_price=Decimal("100"),
        current_qty=1000,
        target_qty=1000 - qty if side is Side.SELL else 1000 + qty,
        reason="权重偏离超出缓冲带",
    )


class TestIntelEvidence:
    def test_requires_url(self) -> None:
        """红线 I-R4：不可复述无出处的内容。"""
        with pytest.raises(ValueError, match="必须带原文链接"):
            intel_item(url="  ")

    def test_valid_item(self) -> None:
        assert intel_item().url.startswith("https://")


class TestTradeIntentValidation:
    def test_non_positive_qty__raises(self) -> None:
        with pytest.raises(ValueError, match="数量必须为正"):
            TradeIntent(
                intent_id=IntentId("x"),
                symbol=A,
                side=Side.BUY,
                qty=0,
                price_low=Decimal("100"),
                price_high=Decimal("101"),
                urgency=Urgency.NORMAL,
                rationale=_full_rationale(),
            )

    def test_inverted_price_band__raises(self) -> None:
        with pytest.raises(ValueError, match="价格区间倒置"):
            TradeIntent(
                intent_id=IntentId("x"),
                symbol=A,
                side=Side.BUY,
                qty=100,
                price_low=Decimal("110"),
                price_high=Decimal("100"),
                urgency=Urgency.NORMAL,
                rationale=_full_rationale(),
            )


def _full_rationale() -> RationaleBundle:
    """构造四支柱齐全的解释。"""
    return RationaleBundle(
        verdict="卖出",
        quant_evidence=(evidence(),),
        technical=analytics_with_technicals(),
        intel_evidence=(intel_item(),),
        counter_evidence=(evidence("dividend_yield"),),
        falsification=("若站回 MA20 则判断被证伪",),
    )


class TestRationaleCompleteness:
    """缺支柱的建议不进入计划——宁可不建议，也不给无法解释的建议。"""

    def test_full_bundle_is_complete(self) -> None:
        assert _full_rationale().is_complete

    def test_missing_quant__incomplete(self) -> None:
        bundle = RationaleBundle(
            verdict="x",
            quant_evidence=(),
            technical=analytics_with_technicals(),
            intel_evidence=(intel_item(),),
            counter_evidence=(evidence(),),
            falsification=("x",),
        )
        assert "①量化依据" in bundle.missing_pillars()

    def test_missing_technical__incomplete(self) -> None:
        empty = PositionAnalytics(symbol=A, as_of=TODAY, market_price=Decimal("100"))
        bundle = RationaleBundle(
            verdict="x",
            quant_evidence=(evidence(),),
            technical=empty,
            intel_evidence=(intel_item(),),
            counter_evidence=(evidence(),),
            falsification=("x",),
        )
        assert "②持仓与技术分析" in bundle.missing_pillars()

    def test_missing_intel_without_note__incomplete(self) -> None:
        """无情报时也必须注明——留空让人分不清"没查"和"查了没有"。"""
        bundle = RationaleBundle(
            verdict="x",
            quant_evidence=(evidence(),),
            technical=analytics_with_technicals(),
            intel_evidence=(),
            counter_evidence=(evidence(),),
            falsification=("x",),
        )
        assert "③情报证据（无情报时也须注明）" in bundle.missing_pillars()

    def test_intel_absent_note_satisfies_pillar(self) -> None:
        bundle = RationaleBundle(
            verdict="x",
            quant_evidence=(evidence(),),
            technical=analytics_with_technicals(),
            intel_evidence=(),
            counter_evidence=(evidence(),),
            falsification=("x",),
            intel_absent_note="近 7 日无该标的相关消息",
        )
        assert bundle.is_complete

    def test_missing_counter_and_falsification__incomplete(self) -> None:
        bundle = RationaleBundle(
            verdict="x",
            quant_evidence=(evidence(),),
            technical=analytics_with_technicals(),
            intel_evidence=(intel_item(),),
            counter_evidence=(),
            falsification=(),
        )
        assert "④反面证据与证伪条件" in bundle.missing_pillars()


class TestPositionAnalytics:
    def test_statements_omit_missing_dimensions(self) -> None:
        """数据缺失的维度自动省略——输出"MA60=0.00"这种假数据比不输出更糟。"""
        sparse = PositionAnalytics(symbol=A, as_of=TODAY, market_price=Decimal("100"))
        assert sparse.statements() == []

    def test_statements_include_holding_details(self) -> None:
        held = PositionAnalytics(
            symbol=A,
            as_of=TODAY,
            market_price=Decimal("100"),
            holding_days=47,
            cost_basis=Decimal("110"),
            unrealized_pnl_pct=Decimal("-0.0909"),
            ma20=98.0,
            ma60=102.0,
        )
        text = " ".join(held.statements())
        assert "47" in text
        assert "110" in text

    def test_tax_free_countdown_surfaces(self) -> None:
        """高股息标的的免税倒计时可能值数千元，必须出现在解释里。"""
        held = PositionAnalytics(
            symbol=A,
            as_of=TODAY,
            market_price=Decimal("100"),
            cost_basis=Decimal("90"),
            days_to_tax_free=15,
            tax_saving_if_wait=Decimal("3240.00"),
        )
        text = " ".join(held.statements())
        assert "15 天" in text
        assert "3240.00" in text


def _bars(prices: list[float]) -> list[Bar]:
    """构造历史 bar。"""
    result = []
    day = BUY_DAY
    for price in prices:
        while day.weekday() >= 5:
            day += dt.timedelta(days=1)
        p = Decimal(str(price))
        result.append(
            Bar(
                symbol=A,
                dt=dt.datetime.combine(day, dt.time(15, 0), tzinfo=CST),
                trade_date=day,
                freq=Freq.D,
                adjust=Adjust.HFQ,
                open=p,
                high=p * Decimal("1.02"),
                low=p * Decimal("0.98"),
                close=p,
                pre_close=p,
                volume=1_000_000,
            )
        )
        day += dt.timedelta(days=1)
    return result


class TestBuildAnalytics:
    def test_empty_series__raises(self) -> None:
        with pytest.raises(ValueError, match="价格序列为空"):
            build_analytics(symbol=A, as_of=TODAY, closes=[])

    def test_computes_technicals(self) -> None:
        closes = [100.0 + i for i in range(80)]
        result = build_analytics(
            symbol=A,
            as_of=TODAY,
            closes=closes,
            highs=[c * 1.02 for c in closes],
            lows=[c * 0.98 for c in closes],
            volumes=[1_000_000.0] * 80,
        )
        assert result.ma20 is not None
        assert result.ma_alignment == "多头排列"
        assert result.stop_loss_price is not None
        assert result.distance_to_stop_pct is not None
        assert result.distance_to_stop_pct < 0

    def test_short_series__leaves_gaps_rather_than_zeros(self) -> None:
        result = build_analytics(symbol=A, as_of=TODAY, closes=[100.0, 101.0])
        assert result.ma60 is None
        assert result.ma_alignment == ""

    def test_with_ledger__fills_holding_fields(self) -> None:
        ledger = replay(
            ACC,
            [
                Transaction(
                    txn_id="t1",
                    account_id=ACC,
                    txn_type=TxnType.BUY,
                    trade_date=BUY_DAY,
                    occurred_at=dt.datetime.combine(BUY_DAY, dt.time(14), tzinfo=CST),
                    symbol=A,
                    qty=1000,
                    price=Decimal("90"),
                    amount=Decimal("90000"),
                    commission=Decimal("5"),
                    net_cash=Decimal("-90005"),
                    source=TxnSource.MANUAL,
                )
            ],
        )
        result = build_analytics(
            symbol=A,
            as_of=TODAY,
            closes=[100.0] * 80,
            ledger=ledger,
            total_value=Decimal("1000000"),
        )
        assert result.is_held
        assert result.cost_basis == Decimal("90.0050")
        assert result.unrealized_pnl_pct is not None
        assert result.unrealized_pnl_pct > 0
        assert result.weight_in_portfolio == pytest.approx(Decimal("0.1"), abs=Decimal("0.01"))

    def test_ledger_without_position__no_holding_fields(self) -> None:
        result = build_analytics(symbol=A, as_of=TODAY, closes=[100.0] * 30, ledger=Ledger(ACC))
        assert not result.is_held


class TestParamHash:
    def test_deterministic(self) -> None:
        params = {"window": 60, "top_n": 10}
        assert compute_param_hash(params) == compute_param_hash(params)

    def test_key_order_independent(self) -> None:
        assert compute_param_hash({"a": 1, "b": 2}) == compute_param_hash({"b": 2, "a": 1})

    def test_different_params__different_hash(self) -> None:
        assert compute_param_hash({"w": 60}) != compute_param_hash({"w": 20})


def _decision(orders: list[RebalanceOrder]) -> object:
    """构造一个全部通过的风控判定。"""
    engine = RiskEngine(constraints=PortfolioConstraints())
    bar = Bar(
        symbol=A,
        dt=dt.datetime.combine(TODAY, dt.time(15), tzinfo=CST),
        trade_date=TODAY,
        freq=Freq.D,
        adjust=Adjust.NONE,
        open=Decimal("100"),
        high=Decimal("100"),
        low=Decimal("100"),
        close=Decimal("100"),
        pre_close=Decimal("100"),
        volume=1_000_000,
    )
    position = Position(
        account_id=ACC,
        symbol=A,
        qty=1000,
        available_qty=1000,
        frozen_qty=0,
        cost_basis_avg=Decimal("100"),
        cost_basis_tax=Decimal("100"),
        first_open_date=BUY_DAY,
        last_trade_date=BUY_DAY,
    )
    return engine.pre_trade_check(
        orders=orders,
        positions={A: position},
        cash=Decimal("1000000"),
        total_value=Decimal("1000000"),
        market=MarketSnapshot(bars={A: bar}),
        trade_date=TODAY,
    )


class TestPlanBuilder:
    def test_builds_intent_with_price_band(self) -> None:
        builder = PlanBuilder(account_id="main")
        plan = builder.build(
            trade_date=TODAY,
            decision=_decision([order()]),  # type: ignore[arg-type]
            analytics={A: analytics_with_technicals()},
            quant_evidence={A: [evidence()]},
            counter_evidence={A: [evidence("dividend_yield")]},
            intel={A: [intel_item()]},
        )
        assert len(plan.intents) == 1
        intent = plan.intents[0]
        assert intent.price_low < Decimal("100") < intent.price_high
        assert intent.qty == 400

    def test_drops_incomplete_rationale(self) -> None:
        """没有量化依据的建议必须被剔除，并记录原因。"""
        builder = PlanBuilder(account_id="main")
        plan = builder.build(
            trade_date=TODAY,
            decision=_decision([order()]),  # type: ignore[arg-type]
            analytics={A: analytics_with_technicals()},
            quant_evidence={},
        )
        assert plan.intents == ()
        assert plan.incomplete
        assert "①量化依据" in plan.incomplete[0][1]

    def test_can_disable_completeness_check(self) -> None:
        builder = PlanBuilder(account_id="main", require_full_rationale=False)
        plan = builder.build(
            trade_date=TODAY,
            decision=_decision([order()]),  # type: ignore[arg-type]
            analytics={A: analytics_with_technicals()},
            quant_evidence={},
        )
        assert len(plan.intents) == 1

    def test_missing_analytics__dropped(self) -> None:
        builder = PlanBuilder(account_id="main")
        plan = builder.build(
            trade_date=TODAY,
            decision=_decision([order()]),  # type: ignore[arg-type]
            analytics={},
            quant_evidence={A: [evidence()]},
        )
        assert plan.intents == ()
        assert "②持仓与技术分析" in plan.incomplete[0][1]

    def test_generates_falsification_conditions(self) -> None:
        """没有证伪条件的判断不可证伪，也就无法从错误中学习。"""
        builder = PlanBuilder(account_id="main")
        plan = builder.build(
            trade_date=TODAY,
            decision=_decision([order()]),  # type: ignore[arg-type]
            analytics={A: analytics_with_technicals()},
            quant_evidence={A: [evidence()]},
            intel={A: [intel_item()]},
        )
        assert plan.intents[0].rationale.falsification
        assert "MA20" in plan.intents[0].rationale.falsification[0]

    def test_intel_absent_note_generated(self) -> None:
        builder = PlanBuilder(account_id="main")
        plan = builder.build(
            trade_date=TODAY,
            decision=_decision([order()]),  # type: ignore[arg-type]
            analytics={A: analytics_with_technicals()},
            quant_evidence={A: [evidence()]},
            intel_lookback_days=7,
        )
        assert "近 7 日无该标的相关消息" in plan.intents[0].rationale.intel_absent_note

    def test_clearance_is_high_urgency(self) -> None:
        clearance = RebalanceOrder(
            symbol=A,
            side=Side.SELL,
            qty=1000,
            reference_price=Decimal("100"),
            current_qty=1000,
            target_qty=0,
            reason="不在目标组合中",
        )
        builder = PlanBuilder(account_id="main")
        plan = builder.build(
            trade_date=TODAY,
            decision=_decision([clearance]),  # type: ignore[arg-type]
            analytics={A: analytics_with_technicals()},
            quant_evidence={A: [evidence()]},
        )
        assert plan.intents[0].urgency is Urgency.HIGH

    def test_plan_carries_traceability(self) -> None:
        """红线 R6：数据指纹 + 策略版本 + 参数哈希。"""
        builder = PlanBuilder(account_id="main")
        plan = builder.build(
            trade_date=TODAY,
            decision=_decision([order()]),  # type: ignore[arg-type]
            analytics={A: analytics_with_technicals()},
            quant_evidence={A: [evidence()]},
            data_fingerprint="abc123",
            strategy_versions={"momentum_trend": "v1"},
            param_hash="def456",
        )
        assert plan.data_fingerprint == "abc123"
        assert plan.strategy_versions == {"momentum_trend": "v1"}
        assert plan.param_hash == "def456"

    def test_plan_not_confirmed_by_default(self) -> None:
        """红线 R5：未确认的计划不得提交真实通道。"""
        builder = PlanBuilder(account_id="main")
        plan = builder.build(
            trade_date=TODAY,
            decision=_decision([order()]),  # type: ignore[arg-type]
            analytics={A: analytics_with_technicals()},
            quant_evidence={A: [evidence()]},
        )
        assert not plan.is_confirmed

    def test_summary_counts_actions(self) -> None:
        builder = PlanBuilder(account_id="main")
        plan = builder.build(
            trade_date=TODAY,
            decision=_decision([order()]),  # type: ignore[arg-type]
            analytics={A: analytics_with_technicals()},
            quant_evidence={A: [evidence()]},
        )
        assert "0 买 / 1 卖" in plan.summary


class TestEndToEnd:
    """数据 → 策略 → 组合 → 风控 → 建议 的完整串联。"""

    def test_full_pipeline_produces_explainable_plan(self) -> None:
        # 1. 数据：两只标的，一涨一跌
        rising = _bars([100.0 + i for i in range(80)])
        history = {A: rising}

        # 2. 策略：动量打分
        as_of = rising[-1].trade_date
        ctx = StrategyContext(as_of=as_of, market=MarketView(history, as_of), universe=(A,))
        signals = MomentumTrendStrategy().generate(ctx)
        assert signals

        # 3. 组合：目标权重与调仓指令
        scores = {s.symbol: max(s.score, 0.01) for s in signals}
        prices = {A: rising[-1].close}
        targets = build_targets(
            scores=scores,
            prices=prices,
            total_value=Decimal("1000000"),
            exposure=Decimal("0.9"),
            constraints=PortfolioConstraints(),
        )
        orders, _ = diff_to_orders(
            targets=targets,
            positions={},
            prices=prices,
            total_value=Decimal("1000000"),
            constraints=PortfolioConstraints(),
        )
        assert orders

        # 4. 风控
        snapshot = MarketSnapshot(bars={A: rising[-1]})
        decision = RiskEngine().pre_trade_check(
            orders=orders,
            positions={},
            cash=Decimal("1000000"),
            total_value=Decimal("1000000"),
            market=snapshot,
            trade_date=as_of,
            circuit_state=CircuitState.NORMAL,
        )

        # 5. 建议：四支柱齐全才能进计划
        analytics = build_analytics(
            symbol=A,
            as_of=as_of,
            closes=[float(b.close) for b in rising],
            highs=[float(b.high) for b in rising],
            lows=[float(b.low) for b in rising],
        )
        plan = PlanBuilder(account_id="main").build(
            trade_date=as_of,
            decision=decision,
            analytics={A: analytics},
            quant_evidence={A: signals[0].evidence},
            counter_evidence={A: signals[0].counter_evidence},
        )

        assert len(plan.intents) == 1
        intent = plan.intents[0]
        assert intent.side is Side.BUY
        assert intent.rationale.is_complete
        assert intent.rationale.quant_evidence
        assert intent.rationale.technical.statements()
        assert intent.rationale.falsification
        assert intent.stop_loss is not None
