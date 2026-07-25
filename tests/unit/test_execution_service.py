"""执行服务测试。

服务层是界面与 CLI 的**唯一**入口，所以这里重点验证：
- 界面不是风控后门：急停、硬闸、真实通道标志在服务层同样生效；
- 漏掉的确认按跳过处理，而不是按执行处理。
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from pathlib import Path

import pytest

from quantstock.config.models import RootConfig
from quantstock.config.settings import Secrets, Settings
from quantstock.execution.types import OrderStatus, SkipReason
from quantstock.infra.clock import CST, FrozenClock, set_clock
from quantstock.infra.errors import ConfigError, ExecutionError, TradingHaltedError
from quantstock.infra.types import Side, Symbol
from quantstock.services.execution_service import ConfirmationDecision, ExecutionService
from tests.unit.test_execution import A, B, intent, plan

pytestmark = pytest.mark.usefixtures("_frozen_service_clock")


@pytest.fixture(autouse=True)
def _frozen_service_clock() -> None:
    """固定在计划当日收盘后，避免计划被判过期。"""
    set_clock(FrozenClock(dt.datetime(2026, 7, 24, 15, 30, tzinfo=CST)))


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    """构造指向临时目录的配置。

    Args:
        tmp_path: 临时目录。
        **overrides: ``execution`` 节的覆盖项。

    Returns:
        Settings 实例。
    """
    config = RootConfig.model_validate(
        {
            "app": {"var_dir": str(tmp_path / "var")},
            "execution": {"broker": "paper", **overrides},
        }
    )
    return Settings(config=config, secrets=Secrets(), config_dir=tmp_path)


def accept(intent_id: str) -> ConfirmationDecision:
    """接受某条意图。"""
    return ConfirmationDecision(intent_id=intent_id, accepted=True)


class TestBrokerSelection:
    """通道选择。"""

    @pytest.mark.parametrize(
        ("choice", "expected"),
        [("paper", "paper"), ("manual", "manual"), ("file_bridge", "file_bridge")],
    )
    def test_builds_configured_broker(self, tmp_path: Path, choice: str, expected: str) -> None:
        service = ExecutionService(make_settings(tmp_path, broker=choice))
        assert service.broker_name == expected

    def test_unavailable_real_channel_fails_loudly(self, tmp_path: Path) -> None:
        # miniQMT 自 2026-07-06 停止新开通，配置了也不该假装能用
        with pytest.raises(ConfigError, match="miniQMT"):
            ExecutionService(make_settings(tmp_path, broker="qmt"))

    def test_bridge_dir_only_for_file_bridge(self, tmp_path: Path) -> None:
        assert ExecutionService(make_settings(tmp_path)).bridge_dir() is None
        bridged = ExecutionService(make_settings(tmp_path, broker="file_bridge"))
        assert bridged.bridge_dir() is not None


class TestPreview:
    """执行前视图。"""

    def test_lists_every_intent_with_limit_price_on_unfavourable_side(self, tmp_path: Path) -> None:
        service = ExecutionService(make_settings(tmp_path))
        target = plan(intent(A, intent_id="i1"), intent(B, side=Side.SELL, intent_id="i2"))
        preview = service.preview(target, current_prices={})

        assert [i.symbol for i in preview.items] == [A, B]
        buy, sell = preview.items
        assert buy.limit_price == buy.price_high, "买入挂区间上沿，宁可贵一点也要成交"
        assert sell.limit_price == sell.price_low

    def test_flags_stale_price_for_review(self, tmp_path: Path) -> None:
        service = ExecutionService(make_settings(tmp_path))
        target = plan(intent(A, price="100", intent_id="i1"))
        preview = service.preview(target, current_prices={A: Decimal("112")})

        assert preview.items[0].needs_review
        assert preview.review_count == 1

    def test_no_realtime_price_means_no_drift_check(self, tmp_path: Path) -> None:
        service = ExecutionService(make_settings(tmp_path))
        preview = service.preview(plan(), current_prices={})
        assert preview.items[0].drift is None
        assert not preview.items[0].needs_review

    def test_surfaces_halt_state(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        service = ExecutionService(settings)
        from quantstock.risk.halt import HaltSwitch  # noqa: PLC0415 - 仅测试内使用

        HaltSwitch(settings.var_dir).halt(reason="数据源异常", by="test")

        preview = service.preview(plan(), current_prices={})
        assert preview.halted
        assert preview.halt_reason == "数据源异常"

    def test_net_cash_delta(self, tmp_path: Path) -> None:
        service = ExecutionService(make_settings(tmp_path))
        target = plan(
            intent(A, qty=100, price="100", intent_id="i1"),
            intent(B, side=Side.SELL, qty=100, price="50", intent_id="i2"),
        )
        preview = service.preview(target, current_prices={})
        assert ExecutionService.net_cash_delta(preview) == pytest.approx(Decimal("-5000"))


class TestExecute:
    """执行。"""

    def test_missing_decision_is_skipped_not_executed(self, tmp_path: Path) -> None:
        # "没来得及看"绝不能等于"下单了"——这个方向的错误不可逆
        service = ExecutionService(make_settings(tmp_path))
        target = plan(intent(A, intent_id="i1"), intent(B, intent_id="i2"))
        service.store.save(target)

        report = service.execute(
            target,
            decisions=[accept("i1")],
            current_prices={},
            confirmed_by="张三",
        )
        assert len(report.submitted) == 1
        assert len(report.skipped) == 1
        assert report.skipped[0].symbol == B
        assert report.skipped[0].skip_reason is SkipReason.OTHER

    def test_accepted_orders_fill_on_paper_channel(self, tmp_path: Path) -> None:
        service = ExecutionService(make_settings(tmp_path))
        target = plan(intent(A, intent_id="i1"))
        service.store.save(target)

        report = service.execute(
            target, decisions=[accept("i1")], current_prices={}, confirmed_by="张三"
        )
        assert report.submitted[0].status is OrderStatus.FILLED
        assert len(report.fills) == 1

    def test_halt_blocks_execution(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        service = ExecutionService(settings)
        target = plan()
        service.store.save(target)
        from quantstock.risk.halt import HaltSwitch  # noqa: PLC0415 - 仅测试内使用

        HaltSwitch(settings.var_dir).halt(reason="人工急停", by="test")

        with pytest.raises(TradingHaltedError):
            service.execute(
                target, decisions=[accept("i1")], current_prices={}, confirmed_by="张三"
            )

    def test_real_channel_requires_live_flag(self, tmp_path: Path) -> None:
        service = ExecutionService(make_settings(tmp_path, broker="file_bridge"))
        target = plan()
        service.store.save(target)
        with pytest.raises(ExecutionError, match="live"):
            service.execute(
                target, decisions=[accept("i1")], current_prices={}, confirmed_by="张三"
            )

    def test_confirmation_recorded_only_after_execution_runs(self, tmp_path: Path) -> None:
        # 前置检查抛错的执行根本没发生过，不该留下确认痕迹
        service = ExecutionService(make_settings(tmp_path, broker="file_bridge"))
        target = plan()
        path = service.store.save(target)
        with pytest.raises(ExecutionError):
            service.execute(
                target, decisions=[accept("i1")], current_prices={}, confirmed_by="张三"
            )
        assert not path.with_suffix(".confirm.json").exists()

    def test_confirmation_is_persisted_on_success(self, tmp_path: Path) -> None:
        service = ExecutionService(make_settings(tmp_path))
        target = plan()
        service.store.save(target)
        service.execute(target, decisions=[accept("i1")], current_prices={}, confirmed_by="张三")
        reloaded = service.store.load(target.trade_date, target.plan_id)
        assert reloaded.confirmed_by == "张三"

    def test_only_symbols_restricts_execution(self, tmp_path: Path) -> None:
        service = ExecutionService(make_settings(tmp_path))
        target = plan(intent(A, intent_id="i1"), intent(B, intent_id="i2"))
        service.store.save(target)

        report = service.execute(
            target,
            decisions=[accept("i1"), accept("i2")],
            current_prices={},
            confirmed_by="张三",
            only_symbols=frozenset({A}),
        )
        assert [o.symbol for o in report.orders] == [A]

    def test_duplicate_execution_is_blocked(self, tmp_path: Path) -> None:
        service = ExecutionService(make_settings(tmp_path))
        target = plan(intent(A, intent_id="i1"))
        service.store.save(target)
        first = service.execute(
            target, decisions=[accept("i1")], current_prices={}, confirmed_by="张三"
        )
        second = service.execute(
            target, decisions=[accept("i1")], current_prices={}, confirmed_by="张三"
        )
        assert len(first.submitted) == 1
        assert len(second.submitted) == 0
        assert "重复下单" in second.skipped[0].skip_note

    def test_hard_limit_aborts_whole_plan(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        settings.config.risk.hard_limits.max_single_order_amount = Decimal("1000")
        service = ExecutionService(settings)
        target = plan(intent(A, qty=500, price="100", intent_id="i1"))
        service.store.save(target)

        report = service.execute(
            target, decisions=[accept("i1")], current_prices={}, confirmed_by="张三"
        )
        assert report.aborted
        assert report.orders == ()

    def test_cancel_all_delegates_to_broker(self, tmp_path: Path) -> None:
        assert ExecutionService(make_settings(tmp_path)).cancel_all() == 0


class TestManualChannel:
    """手工通道：miniQMT 停开后最现实的落地方案。"""

    def test_produces_copyable_checklist(self, tmp_path: Path) -> None:
        service = ExecutionService(make_settings(tmp_path, broker="manual"))
        target = plan(intent(Symbol("600519.SH"), qty=100, price="100", intent_id="i1"))
        service.store.save(target)

        report = service.execute(
            target, decisions=[accept("i1")], current_prices={}, confirmed_by="张三"
        )
        assert len(report.manual_checklist) == 1
        line = report.manual_checklist[0]
        assert "买入" in line
        assert "600519.SH" in line
        assert "100 股" in line


class TestOrderPriceIsTradeable:
    """委托价必须对齐到 0.01（端到端发现的缺陷）。

    ``price_high = 1587 × 1.006 = 1596.522`` 这样的价格在券商 App 里输不进去。
    手工通道是 miniQMT 停开后的主力通道，清单上印一个下不了的价格，
    用户只能自己瞎凑一个数。
    """

    def test_preview_limit_price_is_on_the_tick(self, tmp_path: Path) -> None:
        service = ExecutionService(make_settings(tmp_path))
        target = plan(intent(A, price="1587", intent_id="i1"))
        limit = service.preview(target, current_prices={}).items[0].limit_price
        assert limit == limit.quantize(Decimal("0.01"))

    def test_buy_rounds_up_sell_rounds_down(self, tmp_path: Path) -> None:
        # 取整方向按保住成交概率来定
        service = ExecutionService(make_settings(tmp_path))
        target = plan(
            intent(A, price="1587", intent_id="i1"),
            intent(B, side=Side.SELL, price="1587", intent_id="i2"),
        )
        buy, sell = service.preview(target, current_prices={}).items
        assert buy.limit_price >= buy.price_high
        assert sell.limit_price <= sell.price_low

    def test_submitted_order_matches_the_preview(self, tmp_path: Path) -> None:
        # 用户确认的价格必须就是系统提交的价格
        service = ExecutionService(make_settings(tmp_path))
        target = plan(intent(A, qty=50, price="1587", intent_id="i1"))
        service.store.save(target)
        previewed = service.preview(target, current_prices={}).items[0].limit_price

        report = service.execute(
            target, decisions=[accept("i1")], current_prices={}, confirmed_by="张三"
        )
        assert report.submitted[0].price == previewed

    def test_manual_checklist_price_is_tradeable(self, tmp_path: Path) -> None:
        service = ExecutionService(make_settings(tmp_path, broker="manual"))
        target = plan(intent(A, qty=50, price="1587", intent_id="i1"))
        service.store.save(target)

        report = service.execute(
            target, decisions=[accept("i1")], current_prices={}, confirmed_by="张三"
        )
        line = report.manual_checklist[0]
        assert "1596.53" in line, f"清单上的价格必须能直接照抄：{line}"
