"""执行层测试。

覆盖 docs/05-风控规范.md 第七节要求的执行侧验收点：
- `var/HALT` 存在时所有通道下单均被拒绝
- 硬闸超限中止**整个**计划而非跳过单笔
- 同一 intent 重复提交被拒
- 真实通道未显式 --live 时拒绝执行
"""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

import pytest

from quantstock.advisor.types import (
    PositionAnalytics,
    RationaleBundle,
    TradeIntent,
    TradePlan,
    Urgency,
)
from quantstock.config.models import HardLimitConfig
from quantstock.execution.brokers import (
    Broker,
    FileBridgeBroker,
    ManualBroker,
    PaperBroker,
)
from quantstock.execution.executor import (
    ConfirmationDecision,
    ExecutionRequest,
    Executor,
)
from quantstock.execution.types import (
    BrokerOrder,
    DriftCheck,
    OrderBook,
    OrderStatus,
    SkipReason,
    can_transition,
)
from quantstock.infra.clock import CST, FrozenClock, set_clock
from quantstock.infra.errors import (
    ExecutionError,
    OrderRejectedError,
    TradingHaltedError,
)
from quantstock.infra.types import IntentId, PlanId, Side, Symbol
from quantstock.risk.halt import HaltSwitch, HardLimitGuard
from quantstock.strategy.types import Evidence

A = Symbol("600519.SH")
B = Symbol("300750.SZ")
TODAY = dt.date(2026, 7, 24)


@pytest.fixture(autouse=True)
def _frozen() -> None:
    """把时钟固定在 TODAY 收盘后，避免计划被判过期。"""
    set_clock(FrozenClock(dt.datetime(2026, 7, 24, 15, 30, tzinfo=CST)))


def _rationale() -> RationaleBundle:
    """构造四支柱齐全的解释。"""
    return RationaleBundle(
        verdict="买入",
        quant_evidence=(
            Evidence(
                factor="momentum_60d",
                value=0.18,
                rank_pct=0.87,
                contribution=0.42,
                statement="动量处于 87% 分位",
            ),
        ),
        technical=PositionAnalytics(
            symbol=A, as_of=TODAY, market_price=Decimal("100"), ma20=98.0, ma60=95.0
        ),
        intel_evidence=(),
        counter_evidence=(),
        falsification=("跌破 MA20 则证伪",),
        intel_absent_note="近 7 日无相关消息",
    )


def intent(
    symbol: Symbol = A,
    *,
    side: Side = Side.BUY,
    qty: int = 500,
    price: str = "100",
    intent_id: str = "i1",
) -> TradeIntent:
    """构造一条交易意图。"""
    base = Decimal(price)
    return TradeIntent(
        intent_id=IntentId(intent_id),
        symbol=symbol,
        side=side,
        qty=qty,
        price_low=base * Decimal("0.994"),
        price_high=base * Decimal("1.006"),
        urgency=Urgency.NORMAL,
        rationale=_rationale(),
    )


def plan(*intents: TradeIntent, trade_date: dt.date = TODAY) -> TradePlan:
    """构造交易计划。"""
    return TradePlan(
        plan_id=PlanId("p1"),
        account_id="main",
        trade_date=trade_date,
        generated_at=dt.datetime.combine(trade_date, dt.time(16), tzinfo=CST),
        intents=intents or (intent(),),
    )


def accept_all(_intent: TradeIntent, _drift: DriftCheck | None) -> ConfirmationDecision:
    """全部接受的确认回调。"""
    return ConfirmationDecision(intent_id=str(_intent.intent_id), accepted=True)


def reject_all(_intent: TradeIntent, _drift: DriftCheck | None) -> ConfirmationDecision:
    """全部跳过的确认回调。"""
    return ConfirmationDecision(
        intent_id=str(_intent.intent_id),
        accepted=False,
        skip_reason=SkipReason.DISAGREE_LOGIC,
        skip_note="不认同该逻辑",
    )


def make_executor(
    tmp_path: Path,
    *,
    broker: Broker | None = None,
    limits: HardLimitConfig | None = None,
    order_book: OrderBook | None = None,
) -> tuple[Executor, HaltSwitch]:
    """构造执行器与急停开关。"""
    switch = HaltSwitch(tmp_path)
    executor = Executor(
        broker=broker or PaperBroker(),
        halt_switch=switch,
        hard_limits=HardLimitGuard(limits or HardLimitConfig()),
        order_book=order_book,
    )
    return executor, switch


class TestOrderStateMachine:
    @pytest.mark.parametrize(
        ("source", "target"),
        [
            (OrderStatus.DRAFT, OrderStatus.CONFIRMED),
            (OrderStatus.CONFIRMED, OrderStatus.SUBMITTED),
            (OrderStatus.SUBMITTED, OrderStatus.FILLED),
            (OrderStatus.SUBMITTED, OrderStatus.PARTIAL),
            (OrderStatus.PARTIAL, OrderStatus.FILLED),
        ],
    )
    def test_valid_transitions(self, source: OrderStatus, target: OrderStatus) -> None:
        assert can_transition(source, target)

    @pytest.mark.parametrize(
        ("source", "target"),
        [
            (OrderStatus.DRAFT, OrderStatus.SUBMITTED),
            (OrderStatus.FILLED, OrderStatus.CANCELLED),
            (OrderStatus.CANCELLED, OrderStatus.FILLED),
        ],
    )
    def test_invalid_transitions(self, source: OrderStatus, target: OrderStatus) -> None:
        """已成交的订单又收到"已撤单"通常意味着回报乱序，必须拦下。"""
        assert not can_transition(source, target)

    def test_with_status_rejects_illegal(self) -> None:
        order = BrokerOrder(
            order_id="o1",
            intent_id=IntentId("i1"),
            plan_id=PlanId("p1"),
            symbol=A,
            side=Side.BUY,
            qty=100,
            price=Decimal("100"),
        )
        with pytest.raises(ValueError, match="非法的订单状态迁移"):
            order.with_status(OrderStatus.FILLED)

    def test_terminal_and_live_flags(self) -> None:
        assert OrderStatus.FILLED.is_terminal
        assert not OrderStatus.SUBMITTED.is_terminal
        assert OrderStatus.SUBMITTED.is_live
        assert not OrderStatus.DRAFT.is_live


class TestPaperBroker:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(PaperBroker(), Broker)

    def test_fills_confirmed_orders(self, tmp_path: Path) -> None:
        executor, _ = make_executor(tmp_path)
        report = executor.execute(
            ExecutionRequest(plan=plan(), current_prices={A: Decimal("100")}, confirmed_by="me"),
            confirm=accept_all,
        )
        assert len(report.submitted) == 1
        assert report.submitted[0].status is OrderStatus.FILLED
        assert report.fills

    def test_rejects_unconfirmed_order(self) -> None:
        """红线 R5：只有已确认的订单才能提交。"""
        order = BrokerOrder(
            order_id="o1",
            intent_id=IntentId("i1"),
            plan_id=PlanId("p1"),
            symbol=A,
            side=Side.BUY,
            qty=100,
            price=Decimal("100"),
        )
        with pytest.raises(OrderRejectedError, match="只有已确认的订单"):
            PaperBroker().submit([order])

    def test_does_not_require_live_flag(self) -> None:
        assert PaperBroker().requires_live_flag is False


class TestManualBroker:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(ManualBroker(), Broker)

    def test_produces_copyable_checklist(self, tmp_path: Path) -> None:
        """清单要能直接照抄——不掺解释文字，避免在券商 App 里看错。"""
        executor, _ = make_executor(tmp_path, broker=ManualBroker())
        report = executor.execute(
            ExecutionRequest(plan=plan(), current_prices={A: Decimal("100")}, confirmed_by="me"),
            confirm=accept_all,
        )
        assert report.manual_checklist
        line = report.manual_checklist[0]
        assert "买入" in line
        assert str(A) in line
        assert "500 股" in line

    def test_record_fill(self) -> None:
        broker = ManualBroker()
        order = BrokerOrder(
            order_id="o1",
            intent_id=IntentId("i1"),
            plan_id=PlanId("p1"),
            symbol=A,
            side=Side.BUY,
            qty=500,
            price=Decimal("100"),
            status=OrderStatus.CONFIRMED,
        )
        broker.submit([order])
        fill = broker.record_fill("o1", qty=500, price=Decimal("99.5"))
        assert fill.qty == 500
        assert broker.fetch_fills(["o1"])[0].price == Decimal("99.5")

    def test_record_fill_unknown_order__raises(self) -> None:
        with pytest.raises(OrderRejectedError, match="订单不存在"):
            ManualBroker().record_fill("nope", qty=1, price=Decimal("1"))

    def test_record_fill_over_qty__raises(self) -> None:
        broker = ManualBroker()
        order = BrokerOrder(
            order_id="o1",
            intent_id=IntentId("i1"),
            plan_id=PlanId("p1"),
            symbol=A,
            side=Side.BUY,
            qty=100,
            price=Decimal("100"),
            status=OrderStatus.CONFIRMED,
        )
        broker.submit([order])
        with pytest.raises(OrderRejectedError, match="回填数量"):
            broker.record_fill("o1", qty=200, price=Decimal("100"))


class TestFileBridgeBroker:
    def test_satisfies_protocol(self, tmp_path: Path) -> None:
        assert isinstance(FileBridgeBroker(tmp_path), Broker)

    def test_requires_live_flag(self, tmp_path: Path) -> None:
        """文件桥接会驱动真实下单，必须显式 --live。"""
        executor, _ = make_executor(tmp_path, broker=FileBridgeBroker(tmp_path / "bridge"))
        with pytest.raises(ExecutionError, match="必须显式传入 live=True"):
            executor.execute(
                ExecutionRequest(
                    plan=plan(), current_prices={A: Decimal("100")}, confirmed_by="me"
                ),
                confirm=accept_all,
            )

    def test_writes_plan_file(self, tmp_path: Path) -> None:
        bridge = tmp_path / "bridge"
        executor, _ = make_executor(tmp_path, broker=FileBridgeBroker(bridge))
        executor.execute(
            ExecutionRequest(plan=plan(), current_prices={A: Decimal("100")}, confirmed_by="me"),
            confirm=accept_all,
            live=True,
        )
        written = list(bridge.glob("plan-*.json"))
        assert written
        payload = json.loads(written[0].read_text(encoding="utf-8"))
        assert payload["schema_version"] == FileBridgeBroker.SCHEMA_VERSION
        assert payload["orders"][0]["symbol"] == str(A)

    def test_reads_fills_written_by_executor_side(self, tmp_path: Path) -> None:
        bridge = tmp_path / "bridge"
        broker = FileBridgeBroker(bridge)
        bridge.mkdir(parents=True)
        broker.fills_path.write_text(
            json.dumps(
                {
                    "fills": [
                        {
                            "order_id": "o1",
                            "symbol": str(A),
                            "side": "buy",
                            "qty": 500,
                            "price": "99.80",
                            "fee": "5",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        fills = broker.fetch_fills(["o1"])
        assert len(fills) == 1
        assert fills[0].price == Decimal("99.80")

    def test_missing_fills_file__empty(self, tmp_path: Path) -> None:
        assert FileBridgeBroker(tmp_path / "bridge").fetch_fills(["o1"]) == []


class TestHaltBlocksExecution:
    def test_halt_blocks_all_channels(self, tmp_path: Path) -> None:
        """风控 A12：急停时所有通道下单均被拒绝。"""
        for broker in (PaperBroker(), ManualBroker(), FileBridgeBroker(tmp_path / "b")):
            executor, switch = make_executor(tmp_path, broker=broker)
            switch.halt(reason="测试急停")
            with pytest.raises(TradingHaltedError):
                executor.execute(
                    ExecutionRequest(
                        plan=plan(), current_prices={A: Decimal("100")}, confirmed_by="me"
                    ),
                    confirm=accept_all,
                    live=True,
                )
            switch.resume()

    def test_halt_checked_before_anything_else(self, tmp_path: Path) -> None:
        """急停检查在第一行——即使计划已过期也先报急停。"""
        executor, switch = make_executor(tmp_path)
        switch.halt(reason="测试")
        stale = plan(trade_date=dt.date(2020, 1, 1))
        with pytest.raises(TradingHaltedError):
            executor.execute(
                ExecutionRequest(plan=stale, current_prices={A: Decimal("100")}, confirmed_by="me"),
                confirm=accept_all,
            )


class TestHardLimitAbortsWholePlan:
    def test_single_over_limit_aborts_entire_plan(self, tmp_path: Path) -> None:
        """单笔超限通常意味着计算基数出错，其余单笔同样不可信。"""
        executor, _ = make_executor(
            tmp_path,
            limits=HardLimitConfig(
                max_single_order_amount=Decimal("10000"),
                max_daily_total_amount=Decimal("50000"),
            ),
        )
        report = executor.execute(
            ExecutionRequest(
                plan=plan(
                    intent(A, qty=100, intent_id="i1"),
                    intent(B, qty=500, price="1000", intent_id="i2"),
                ),
                current_prices={A: Decimal("100"), B: Decimal("1000")},
                confirmed_by="me",
            ),
            confirm=accept_all,
        )
        assert report.aborted
        assert "硬闸" in report.abort_reason
        assert report.orders == ()

    def test_within_limits__proceeds(self, tmp_path: Path) -> None:
        executor, _ = make_executor(
            tmp_path, limits=HardLimitConfig(max_single_order_amount=Decimal("100000"))
        )
        report = executor.execute(
            ExecutionRequest(plan=plan(), current_prices={A: Decimal("100")}, confirmed_by="me"),
            confirm=accept_all,
        )
        assert not report.aborted


class TestIdempotency:
    def test_duplicate_intent_blocked(self, tmp_path: Path) -> None:
        """重复下单在真实资金上的代价不可逆。"""
        book = OrderBook()
        executor, _ = make_executor(tmp_path, order_book=book)
        request = ExecutionRequest(
            plan=plan(), current_prices={A: Decimal("100")}, confirmed_by="me"
        )

        first = executor.execute(request, confirm=accept_all)
        assert len(first.submitted) == 1

        second = executor.execute(request, confirm=accept_all)
        assert second.submitted == ()
        assert second.skipped[0].skip_note == "该意图已提交过，拒绝重复下单"


class TestConfirmation:
    def test_requires_confirmed_by(self, tmp_path: Path) -> None:
        """红线 R5：必须记录确认人。"""
        executor, _ = make_executor(tmp_path)
        with pytest.raises(ExecutionError, match="必须记录确认人"):
            executor.execute(
                ExecutionRequest(
                    plan=plan(), current_prices={A: Decimal("100")}, confirmed_by="  "
                ),
                confirm=accept_all,
            )

    def test_skip_requires_reason(self) -> None:
        """复盘要按原因分组统计人工干预的价值，所以原因必填。"""
        with pytest.raises(ValueError, match="必须选择原因"):
            ConfirmationDecision(intent_id="i1", accepted=False)

    def test_skipped_orders_recorded_with_reason(self, tmp_path: Path) -> None:
        executor, _ = make_executor(tmp_path)
        report = executor.execute(
            ExecutionRequest(plan=plan(), current_prices={A: Decimal("100")}, confirmed_by="me"),
            confirm=reject_all,
        )
        assert report.submitted == ()
        assert report.skipped[0].skip_reason is SkipReason.DISAGREE_LOGIC
        assert report.skip_reasons() == {"disagree_logic": 1}

    def test_adjusted_qty_applied(self, tmp_path: Path) -> None:
        def half(intent_obj: TradeIntent, _drift: DriftCheck | None) -> ConfirmationDecision:
            return ConfirmationDecision(
                intent_id=str(intent_obj.intent_id),
                accepted=True,
                adjusted_qty=intent_obj.qty // 2,
            )

        executor, _ = make_executor(tmp_path)
        report = executor.execute(
            ExecutionRequest(plan=plan(), current_prices={A: Decimal("100")}, confirmed_by="me"),
            confirm=half,
        )
        assert report.submitted[0].qty == 250


class TestPriceDrift:
    def test_drift_reported_to_confirmer(self, tmp_path: Path) -> None:
        seen: list[DriftCheck | None] = []

        def capture(i: TradeIntent, drift: DriftCheck | None) -> ConfirmationDecision:
            seen.append(drift)
            return ConfirmationDecision(intent_id=str(i.intent_id), accepted=True)

        executor, _ = make_executor(tmp_path)
        executor.execute(
            ExecutionRequest(plan=plan(), current_prices={A: Decimal("110")}, confirmed_by="me"),
            confirm=capture,
        )
        drift = seen[0]
        assert drift is not None
        assert drift.is_stale
        assert "超出阈值" in drift.message

    def test_small_drift_not_stale(self, tmp_path: Path) -> None:
        seen: list[DriftCheck | None] = []

        def capture(i: TradeIntent, drift: DriftCheck | None) -> ConfirmationDecision:
            seen.append(drift)
            return ConfirmationDecision(intent_id=str(i.intent_id), accepted=True)

        executor, _ = make_executor(tmp_path)
        executor.execute(
            ExecutionRequest(plan=plan(), current_prices={A: Decimal("101")}, confirmed_by="me"),
            confirm=capture,
        )
        assert seen[0] is not None
        assert not seen[0].is_stale

    def test_missing_price__no_drift_check(self, tmp_path: Path) -> None:
        seen: list[DriftCheck | None] = []

        def capture(i: TradeIntent, drift: DriftCheck | None) -> ConfirmationDecision:
            seen.append(drift)
            return ConfirmationDecision(intent_id=str(i.intent_id), accepted=True)

        executor, _ = make_executor(tmp_path)
        executor.execute(
            ExecutionRequest(plan=plan(), current_prices={}, confirmed_by="me"),
            confirm=capture,
        )
        assert seen[0] is None


class TestPlanExpiry:
    def test_stale_plan_aborted(self, tmp_path: Path) -> None:
        """隔夜的计划不能照单执行——市场已经变了。"""
        executor, _ = make_executor(tmp_path)
        report = executor.execute(
            ExecutionRequest(
                plan=plan(trade_date=dt.date(2026, 7, 1)),
                current_prices={A: Decimal("100")},
                confirmed_by="me",
            ),
            confirm=accept_all,
        )
        assert report.aborted
        assert "已过期" in report.abort_reason

    def test_todays_plan_ok(self, tmp_path: Path) -> None:
        executor, _ = make_executor(tmp_path)
        report = executor.execute(
            ExecutionRequest(plan=plan(), current_prices={A: Decimal("100")}, confirmed_by="me"),
            confirm=accept_all,
        )
        assert not report.aborted


class TestOnlyFilter:
    def test_executes_only_selected_symbols(self, tmp_path: Path) -> None:
        executor, _ = make_executor(tmp_path)
        report = executor.execute(
            ExecutionRequest(
                plan=plan(intent(A, intent_id="i1"), intent(B, intent_id="i2")),
                current_prices={A: Decimal("100"), B: Decimal("100")},
                confirmed_by="me",
                only_symbols=frozenset({A}),
            ),
            confirm=accept_all,
        )
        assert [o.symbol for o in report.submitted] == [A]


class TestLimitPriceSide:
    def test_buy_uses_upper_bound(self, tmp_path: Path) -> None:
        """限价取区间的不利一侧：宁可成交在略差的价位，也不要整天不成交。"""
        executor, _ = make_executor(tmp_path)
        report = executor.execute(
            ExecutionRequest(plan=plan(), current_prices={A: Decimal("100")}, confirmed_by="me"),
            confirm=accept_all,
        )
        assert report.submitted[0].price == plan().intents[0].price_high

    def test_sell_uses_lower_bound(self, tmp_path: Path) -> None:
        sell = intent(side=Side.SELL)
        executor, _ = make_executor(tmp_path)
        report = executor.execute(
            ExecutionRequest(
                plan=plan(sell), current_prices={A: Decimal("100")}, confirmed_by="me"
            ),
            confirm=accept_all,
        )
        assert report.submitted[0].price == sell.price_low
