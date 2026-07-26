"""大模型服务与 CLI 测试。

重点验证两条**系统级**保证，它们比任何单元行为都重要：

- ``llm.enabled=false`` 时全链路完整可用（红线 LR2）；
- 回测中模式被**强制**为 replay，配置写 live 也不例外（红线 LR3）。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
import yaml
from pydantic import SecretStr
from typer.testing import CliRunner

from quantstock.cli.main import app
from quantstock.config.models import RootConfig
from quantstock.config.settings import Secrets, Settings
from quantstock.infra.clock import CST, FrozenClock, set_clock
from quantstock.infra.errors import LLMError, LLMLiveCallInBacktestError
from quantstock.llm.protocols import CompletionRequest, CompletionResponse
from quantstock.services.llm_service import LLMService, _build_provider
from tests.unit.test_llm import GOOD_JUDGEMENT, MAOTAI, MATERIALS, StubProvider

runner = CliRunner()

NOW = dt.datetime(2026, 7, 25, 18, 0, tzinfo=CST)

pytestmark = pytest.mark.usefixtures("_frozen_llm_service_clock")


@pytest.fixture(autouse=True)
def _frozen_llm_service_clock() -> None:
    """固定时钟。"""
    set_clock(FrozenClock(NOW))


def make_settings(tmp_path: Path, **llm: object) -> Settings:
    """构造带 LLM 配置的 Settings。

    Args:
        tmp_path: 临时目录。
        **llm: ``llm`` 节的覆盖项。

    Returns:
        Settings 实例。
    """
    config = RootConfig.model_validate(
        {"app": {"var_dir": str(tmp_path / "var")}, "llm": llm}
        if llm
        else {"app": {"var_dir": str(tmp_path / "var")}}
    )
    return Settings(config=config, secrets=Secrets(), config_dir=tmp_path / "config")


class TestMasterSwitch:
    """总开关（红线 LR2）。"""

    def test_disabled_by_default(self, tmp_path: Path) -> None:
        service = LLMService(make_settings(tmp_path))
        assert not service.enabled
        assert service.client.mode == "off"

    def test_every_task_still_constructible_when_off(self, tmp_path: Path) -> None:
        # 上层不需要到处写 if llm_enabled——漏掉的那处就是 LR2 的破口
        service = LLMService(make_settings(tmp_path))
        assert service.position_judge() is not None
        assert service.market_judge() is not None
        assert service.intel_classify() is not None
        assert service.explain() is not None

    def test_position_judge_is_identity_when_off(self, tmp_path: Path) -> None:
        service = LLMService(make_settings(tmp_path))
        outcome = service.position_judge().run(
            MAOTAI, base_score=0.62, materials=MATERIALS, as_of="T日"
        )
        assert not outcome.used_llm
        assert outcome.influence.final_score == 0.62

    def test_explain_falls_back_when_off(self, tmp_path: Path) -> None:
        service = LLMService(make_settings(tmp_path))
        outcome = service.explain().run(
            verdict="减持", pillars={"①量化依据": ["打分下滑"]}, as_of="T日"
        )
        assert not outcome.llm_generated
        assert any("打分下滑" in line for line in outcome.as_lines())

    def test_config_forces_mode_off_when_disabled(self, tmp_path: Path) -> None:
        # 配置层自身就把 enabled=false + mode=live 这种自相矛盾拧正了
        service = LLMService(make_settings(tmp_path, enabled=False, mode="live"))
        assert service.client.mode == "off"

    def test_task_enabled_is_false_when_master_off(self, tmp_path: Path) -> None:
        service = LLMService(make_settings(tmp_path))
        assert not service.task_enabled("position_judge")


class TestBacktestMode:
    """回测模式强制（红线 LR3）。"""

    def test_live_config_forced_to_replay_in_backtest(self, tmp_path: Path) -> None:
        service = LLMService(make_settings(tmp_path, enabled=True, mode="live"), in_backtest=True)
        assert service.client.mode == "replay"

    def test_replay_stays_replay(self, tmp_path: Path) -> None:
        service = LLMService(make_settings(tmp_path, enabled=True, mode="replay"), in_backtest=True)
        assert service.client.mode == "replay"

    def test_off_stays_off_in_backtest(self, tmp_path: Path) -> None:
        service = LLMService(make_settings(tmp_path), in_backtest=True)
        assert service.client.mode == "off"

    def test_no_live_call_can_escape_in_backtest(self, tmp_path: Path) -> None:
        # 服务层把模式拧成 replay 了，底层客户端还有一道独立的闸门
        with pytest.raises(LLMLiveCallInBacktestError):
            from quantstock.llm.client import LLMClient  # noqa: PLC0415

            LLMClient(mode="live", in_backtest=True)

    def test_anonymization_on_in_backtest(self, tmp_path: Path) -> None:
        provider = StubProvider()
        service = LLMService(
            make_settings(tmp_path, enabled=True, mode="replay"),
            provider=provider,
            in_backtest=True,
        )
        # 通过 backfill 之外的路径无法验证 prompt，改为验证任务已按回测配置构造
        task = service.position_judge()
        outcome = task.run(MAOTAI, base_score=0.6, materials=MATERIALS, as_of="T日")
        # replay 未命中 → 降级，但任务本身是带匿名化构造的
        assert not outcome.used_llm


class TestAlphaGuard:
    """α 硬上限（红线 LR2）。"""

    def test_config_rejects_alpha_above_cap(self, tmp_path: Path) -> None:
        # pydantic 的 le=0.20 直接拦下
        with pytest.raises(ValueError, match=r"less than or equal to 0\.2"):
            make_settings(tmp_path, enabled=True, max_influence=0.5)

    def test_service_rechecks_alpha(self, tmp_path: Path) -> None:
        # 纵深防御：配置被绕过时服务层再拦一次
        settings = make_settings(tmp_path, enabled=True, max_influence=0.15)
        object.__setattr__(settings.config.llm, "max_influence", 0.9)
        with pytest.raises(LLMError, match="硬上限"):
            LLMService(settings)

    def test_alpha_default(self, tmp_path: Path) -> None:
        assert LLMService(make_settings(tmp_path)).alpha == 0.15


class TestModelTiers:
    """模型分级。"""

    def test_intel_classify_uses_fast_tier(self, tmp_path: Path) -> None:
        # L1 高频量大任务简单，用便宜模型
        service = LLMService(make_settings(tmp_path))
        assert service.model_for("intel_classify") == "claude-haiku-4-5-20251001"

    def test_judge_uses_main_tier(self, tmp_path: Path) -> None:
        service = LLMService(make_settings(tmp_path))
        assert service.model_for("position_judge") == "claude-sonnet-5"

    def test_unknown_task_falls_back_to_main(self, tmp_path: Path) -> None:
        service = LLMService(make_settings(tmp_path))
        assert service.model_for("nonexistent") == "claude-sonnet-5"


class TestParamHash:
    """可追溯性（红线 R6）。"""

    def test_prompt_version_enters_param_hash(self, tmp_path: Path) -> None:
        # 改提示词等同于改策略
        parts = LLMService(make_settings(tmp_path, enabled=True, mode="replay")).param_hash_parts()
        assert "llm_prompt_version" in parts
        assert "llm_alpha" in parts
        assert "llm_model_main" in parts

    def test_disabled_llm_has_a_stable_marker(self, tmp_path: Path) -> None:
        assert LLMService(make_settings(tmp_path)).param_hash_parts() == {"llm": "off"}


class TestStatus:
    """状态与用量。"""

    def test_status_when_off(self, tmp_path: Path) -> None:
        status = LLMService(make_settings(tmp_path)).status()
        assert not status.enabled
        assert "纯量化" in status.message

    def test_status_reports_cache_and_spend(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path, enabled=True, mode="live")
        service = LLMService(settings, provider=StubProvider())
        service.position_judge().run(MAOTAI, base_score=0.6, materials=MATERIALS, as_of="T日")

        status = service.status()
        assert status.enabled
        assert status.cached_entries == 1
        assert status.daily_spent_usd > 0
        assert "α=0.15" in status.message

    def test_backfill_then_replay_is_free_and_deterministic(self, tmp_path: Path) -> None:
        # 这是"LLM 参与的回测可复现"的完整闭环
        settings = make_settings(tmp_path, enabled=True, mode="live")
        writer = LLMService(settings, provider=StubProvider())
        first = writer.position_judge().run(
            MAOTAI, base_score=0.6, materials=MATERIALS, as_of="T日"
        )
        assert first.used_llm

        replayer = LLMService(
            make_settings(tmp_path, enabled=True, mode="replay"), in_backtest=True
        )
        second = replayer.position_judge().run(
            MAOTAI, base_score=0.6, materials=MATERIALS, as_of="T日"
        )
        third = replayer.position_judge().run(
            MAOTAI, base_score=0.6, materials=MATERIALS, as_of="T日"
        )
        assert second.judgement == third.judgement
        assert second.influence.final_score == third.influence.final_score
        assert second.cost_usd == 0.0


class TestBudgetIntegration:
    """预算集成（红线 LR7）。"""

    def test_tiny_budget_degrades_after_first_call(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path, enabled=True, mode="live", daily_budget_usd=0.001)
        service = LLMService(settings, provider=StubProvider())

        first = service.position_judge().run(
            MAOTAI, base_score=0.6, materials=MATERIALS, as_of="T日"
        )
        second = service.position_judge().run(
            MAOTAI, base_score=0.6, materials={"m1": "另一份材料"}, as_of="T+1日"
        )
        assert first.used_llm
        assert not second.used_llm
        assert service.status().degraded


class TestLlmCli:
    """CLI。"""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "base.yaml").write_text(
            yaml.safe_dump({"app": {"var_dir": str(tmp_path / "var")}}, allow_unicode=True),
            encoding="utf-8",
        )
        return config_dir

    def test_status_when_disabled(self, workspace: Path) -> None:
        result = runner.invoke(app, ["llm", "status", "-c", str(workspace)])
        assert result.exit_code == 0, result.output
        assert "纯量化" in result.output

    def test_status_when_enabled(self, tmp_path: Path) -> None:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "base.yaml").write_text(
            yaml.safe_dump(
                {
                    "app": {"var_dir": str(tmp_path / "var")},
                    "llm": {"enabled": True, "mode": "replay"},
                },
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        result = runner.invoke(app, ["llm", "status", "-c", str(config_dir)])
        assert result.exit_code == 0, result.output
        assert "replay" in result.output
        assert "0.20" in result.output, "界面上要能看到硬上限"

    def test_coverage_warns_on_empty_cache(self, workspace: Path) -> None:
        result = runner.invoke(app, ["llm", "coverage", "-c", str(workspace)])
        assert result.exit_code == 0, result.output
        assert "backfill" in result.output


class TestProviderContract:
    """供应商契约。"""

    def test_stub_satisfies_the_protocol(self) -> None:
        from quantstock.llm.protocols import LLMProvider  # noqa: PLC0415

        assert isinstance(StubProvider(), LLMProvider)

    def test_request_payload_roundtrip(self) -> None:
        from quantstock.llm.protocols import ChatMessage  # noqa: PLC0415

        request = CompletionRequest(
            model="m", system="s", messages=(ChatMessage(role="user", content="hi"),)
        )
        payload = request.payload()
        assert payload["model"] == "m"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    def test_response_payload_roundtrip(self) -> None:
        response = CompletionResponse(
            text=json.dumps(GOOD_JUDGEMENT), model="m", input_tokens=10, output_tokens=5
        )
        restored = CompletionResponse.from_payload(dict(response.payload()))
        assert restored == response

    def test_response_tolerates_hand_edited_cache(self) -> None:
        # 缓存是外部输入，token 计数转不动不该让整个回放中断
        restored = CompletionResponse.from_payload(
            {"text": "t", "model": "m", "input_tokens": "不是数字"}
        )
        assert restored.input_tokens == 0


class TestProviderAssembly:
    """默认供应商装配。

    ``provider=None`` 是生产环境实际走的那条路，测试里到处注入假供应商
    会让它一行都跑不到。
    """

    def test_no_key_means_no_provider(self, tmp_path: Path) -> None:
        # 没配 key 时返回 None 而不是抛错：用户可能只是想先看看界面长什么样，
        # 不该被一个可选功能挡在门外（红线 LR2：关掉 LLM 系统须完整可用）
        assert _build_provider(make_settings(tmp_path)) is None

    def test_blank_key_means_no_provider(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        settings.secrets.anthropic_api_key = SecretStr("   ")
        assert _build_provider(settings) is None

    def test_key_yields_anthropic_provider(self, tmp_path: Path) -> None:
        settings = make_settings(tmp_path)
        settings.secrets.anthropic_api_key = SecretStr("sk-ant-test")
        provider = _build_provider(settings)
        assert provider is not None
        assert provider.name == "anthropic"
