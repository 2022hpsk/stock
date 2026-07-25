"""执行相关 CLI 命令测试。

CLI 是与界面平级的**薄**客户端，这里只验证它把参数正确传给 services、
把结果正确渲染出来，业务规则本身由 services / execution 层的测试覆盖。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from quantstock.advisor.store import PlanStore
from quantstock.cli.main import app
from quantstock.infra.clock import CST, FrozenClock, set_clock
from quantstock.infra.types import Side
from tests.unit.test_execution import TODAY, A, B, intent, plan

runner = CliRunner()

pytestmark = pytest.mark.usefixtures("_frozen_cli_clock")


@pytest.fixture(autouse=True)
def _frozen_cli_clock() -> None:
    """固定在计划当日收盘后。"""
    set_clock(FrozenClock(dt.datetime(2026, 7, 24, 15, 30, tzinfo=CST)))


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """构造一个含配置与已保存计划的工作目录。

    Args:
        tmp_path: 临时目录。

    Returns:
        配置目录路径。
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "app": {"var_dir": str(tmp_path / "var")},
                "execution": {"broker": "paper"},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    store = PlanStore(tmp_path / "var" / "plans")
    store.save(plan(intent(A, intent_id="i1"), intent(B, side=Side.SELL, intent_id="i2")))
    return config_dir


class TestPlanShow:
    """plan show。"""

    def test_renders_intents_and_traceability(self, workspace: Path) -> None:
        result = runner.invoke(app, ["plan", "show", "-c", str(workspace)])
        assert result.exit_code == 0, result.output
        assert "600519.SH" in result.output
        assert "300750.SZ" in result.output
        assert "买入" in result.output
        assert "卖出" in result.output

    def test_missing_plan_exits_nonzero(self, workspace: Path) -> None:
        result = runner.invoke(app, ["plan", "show", "-c", str(workspace), "-d", "2026-07-20"])
        assert result.exit_code == 1
        assert "没有已保存的计划" in result.output


class TestExecuteCommand:
    """execute。"""

    def test_accepting_everything_submits_all(self, workspace: Path) -> None:
        # 输入：确认人 → 逐单 y
        result = runner.invoke(app, ["execute", "-c", str(workspace)], input="张三\ny\ny\n")
        assert result.exit_code == 0, result.output
        assert "提交 2 笔" in result.output

    def test_skipping_requires_choosing_a_reason(self, workspace: Path) -> None:
        # 第一单跳过 → 选原因 1（disagree_logic）→ 备注留空；第二单接受
        result = runner.invoke(app, ["execute", "-c", str(workspace)], input="张三\nn\n1\n\ny\n")
        assert result.exit_code == 0, result.output
        assert "提交 1 笔" in result.output
        assert "disagree_logic×1" in result.output

    def test_only_filter_limits_scope(self, workspace: Path) -> None:
        result = runner.invoke(
            app,
            ["execute", "-c", str(workspace), "--only", str(A)],
            input="张三\ny\n",
        )
        assert result.exit_code == 0, result.output
        assert "提交 1 笔" in result.output

    def test_halt_blocks_before_asking_anything(self, workspace: Path) -> None:
        halted = runner.invoke(app, ["halt", "-c", str(workspace), "-r", "数据源异常"])
        assert halted.exit_code == 0, halted.output

        result = runner.invoke(app, ["execute", "-c", str(workspace)], input="张三\ny\ny\n")
        assert result.exit_code == 1
        assert "急停" in result.output

    def test_missing_plan_exits_nonzero(self, workspace: Path) -> None:
        result = runner.invoke(app, ["execute", "-c", str(workspace), "-d", "2026-07-20"])
        assert result.exit_code == 1
        assert "没有已保存的计划" in result.output


class TestCancelAll:
    """cancel-all。"""

    def test_requires_confirmation(self, workspace: Path) -> None:
        result = runner.invoke(app, ["cancel-all", "-c", str(workspace)], input="n\n")
        assert result.exit_code == 0
        assert "已取消" in result.output

    def test_runs_with_yes(self, workspace: Path) -> None:
        result = runner.invoke(app, ["cancel-all", "-c", str(workspace), "-y"])
        assert result.exit_code == 0, result.output
        assert "已发出撤单指令" in result.output


def test_plan_dates_are_listed(tmp_path: Path) -> None:
    """审计页要能翻阅历史计划。"""
    store = PlanStore(tmp_path)
    store.save(plan())
    assert store.list_dates() == [TODAY]
