"""急停开关与绝对金额硬闸测试。

覆盖 docs/05-风控规范.md 第七节要求：
- A12：标志文件存在时所有下单路径均被拒绝
- A10：**比例风控全部通过但绝对值超限**的场景必须被拦截
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from quantstock.config.models import HardLimitConfig
from quantstock.infra.errors import HardLimitExceededError, TradingHaltedError
from quantstock.risk.halt import HaltSwitch, HardLimitGuard


class TestHaltSwitch:
    def test_initially_not_halted(self, tmp_path: Path) -> None:
        assert HaltSwitch(tmp_path).is_halted() is False

    def test_halt_creates_flag_and_blocks(self, tmp_path: Path) -> None:
        switch = HaltSwitch(tmp_path)
        switch.halt(reason="发现数据异常", by="cli")
        assert switch.is_halted()
        with pytest.raises(TradingHaltedError, match="急停状态"):
            switch.ensure_not_halted()

    def test_halt_requires_reason(self, tmp_path: Path) -> None:
        """急停原因必填——事后复盘要靠它。"""
        with pytest.raises(ValueError, match="必须填写原因"):
            HaltSwitch(tmp_path).halt(reason="   ")

    def test_state_records_reason_and_time(self, tmp_path: Path) -> None:
        switch = HaltSwitch(tmp_path)
        switch.halt(reason="账户对账不平", by="ui")
        state = switch.state()
        assert state.halted
        assert state.reason == "账户对账不平"
        assert state.halted_by == "ui"
        assert state.halted_at

    def test_resume_clears_flag(self, tmp_path: Path) -> None:
        switch = HaltSwitch(tmp_path)
        switch.halt(reason="测试")
        switch.resume(by="cli")
        assert not switch.is_halted()
        switch.ensure_not_halted()  # 不应抛出

    def test_resume_when_not_halted__no_error(self, tmp_path: Path) -> None:
        HaltSwitch(tmp_path).resume()

    def test_survives_process_restart(self, tmp_path: Path) -> None:
        """用文件而非内存状态：进程重启后急停依然生效。"""
        HaltSwitch(tmp_path).halt(reason="重启前触发")
        assert HaltSwitch(tmp_path).is_halted()

    def test_corrupted_flag__still_treated_as_halted(self, tmp_path: Path) -> None:
        """无法确认状态时必须取保守一侧。"""
        (tmp_path / "HALT").write_text("这不是 JSON", encoding="utf-8")
        switch = HaltSwitch(tmp_path)
        assert switch.state().halted
        with pytest.raises(TradingHaltedError):
            switch.ensure_not_halted()

    def test_manual_file_removal_works(self, tmp_path: Path) -> None:
        """用户可以直接 rm var/HALT 手工解除——出事时不该依赖程序本身还正常。"""
        switch = HaltSwitch(tmp_path)
        switch.halt(reason="测试")
        switch.path.unlink()
        assert not switch.is_halted()


class TestHardLimitGuard:
    @pytest.fixture
    def guard(self) -> HardLimitGuard:
        return HardLimitGuard(
            HardLimitConfig(
                max_single_order_amount=Decimal("100000"),
                max_daily_total_amount=Decimal("300000"),
                max_daily_order_count=5,
                max_single_order_qty=10000,
            )
        )

    def test_within_limits__passes(self, guard: HardLimitGuard) -> None:
        result = guard.check_orders(
            order_amounts=[Decimal("50000"), Decimal("30000")],
            order_quantities=[500, 300],
        )
        assert result.passed
        guard.enforce(result)  # 不应抛出

    def test_single_order_over_limit__fails(self, guard: HardLimitGuard) -> None:
        result = guard.check_orders(order_amounts=[Decimal("150000")])
        assert not result.passed
        assert result.failures[0].name == "max_single_order_amount"

    def test_daily_total_over_limit__fails(self, guard: HardLimitGuard) -> None:
        result = guard.check_orders(order_amounts=[Decimal("90000")] * 4)
        assert not result.passed
        assert any(f.name == "max_daily_total_amount" for f in result.failures)

    def test_too_many_orders__fails(self, guard: HardLimitGuard) -> None:
        result = guard.check_orders(order_amounts=[Decimal("1000")] * 10)
        assert any(f.name == "max_daily_order_count" for f in result.failures)

    def test_quantity_over_limit__fails(self, guard: HardLimitGuard) -> None:
        result = guard.check_orders(order_amounts=[Decimal("1000")], order_quantities=[999999])
        assert any(f.name == "max_single_order_qty" for f in result.failures)

    def test_enforce__aborts_entire_plan(self, guard: HardLimitGuard) -> None:
        """任一项超限即中止整个计划——单笔超限通常意味着计算基数出错，
        此时其余单笔同样不可信。"""
        result = guard.check_orders(
            order_amounts=[Decimal("10000"), Decimal("150000"), Decimal("20000")]
        )
        with pytest.raises(HardLimitExceededError, match="中止整个交易计划"):
            guard.enforce(result)

    def test_disabled__skips_checks(self) -> None:
        guard = HardLimitGuard(HardLimitConfig(enabled=False))
        assert guard.check_orders(order_amounts=[Decimal("99999999")]).passed


class TestAccountSanity:
    """规则 A11：账户数据不可信时停止一切下单。"""

    @pytest.fixture
    def guard(self) -> HardLimitGuard:
        return HardLimitGuard(
            HardLimitConfig(
                min_account_value_sanity=Decimal("1000"),
                max_account_value_sanity=Decimal("10000000"),
                max_account_value_daily_change=0.30,
            )
        )

    def test_normal_value__passes(self, guard: HardLimitGuard) -> None:
        assert guard.check_account_sanity(total_value=Decimal("500000")).passed

    def test_below_floor__fails(self, guard: HardLimitGuard) -> None:
        result = guard.check_account_sanity(total_value=Decimal("100"))
        assert any(f.name == "account_value_min" for f in result.failures)

    def test_above_ceiling__fails(self, guard: HardLimitGuard) -> None:
        result = guard.check_account_sanity(total_value=Decimal("99999999"))
        assert any(f.name == "account_value_max" for f in result.failures)

    def test_sudden_jump__flagged_as_sync_error(self, guard: HardLimitGuard) -> None:
        result = guard.check_account_sanity(
            total_value=Decimal("1000000"), previous_value=Decimal("500000")
        )
        assert any(f.name == "account_value_daily_change" for f in result.failures)

    def test_normal_daily_move__passes(self, guard: HardLimitGuard) -> None:
        result = guard.check_account_sanity(
            total_value=Decimal("510000"), previous_value=Decimal("500000")
        )
        assert result.passed


class TestProportionalBlindSpot:
    """核心场景：比例风控全部通过，但绝对值超限。

    这正是硬闸存在的理由——若账户总资产被算成真实值的 10 倍，
    "单票不超过 15%" 之类的约束仍会通过，但实际下单金额是灾难性的。
    """

    def test_inflated_account_value__proportional_ok_but_hard_limit_blocks(self) -> None:
        real_account_value = Decimal("100000")
        inflated_value = real_account_value * 10  # 数据错误导致的十倍放大

        # 比例风控视角：单笔占"总资产"的 10%，完全在 15% 上限之内
        order_amount = inflated_value * Decimal("0.10")
        assert order_amount / inflated_value <= Decimal("0.15")

        # 但绝对金额已达真实账户的 100%，硬闸必须拦下
        guard = HardLimitGuard(
            HardLimitConfig(
                max_single_order_amount=Decimal("30000"),
                max_daily_total_amount=Decimal("60000"),
            )
        )
        result = guard.check_orders(order_amounts=[order_amount])
        assert not result.passed
        with pytest.raises(HardLimitExceededError):
            guard.enforce(result)
