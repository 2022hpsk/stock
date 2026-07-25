"""配置模型与加载测试。

重点覆盖那些"配置写错就会出事"的校验：硬闸一致性、熔断阈值顺序、
真实通道强制人工确认、LLM 影响上限。
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from quantstock.config.models import (
    CircuitBreakerConfig,
    ExecutionConfig,
    HardLimitConfig,
    LLMConfig,
    PortfolioConfig,
    RootConfig,
)
from quantstock.config.settings import Secrets, load_config, load_settings
from quantstock.infra.errors import ConfigError


class TestDefaults:
    def test_root_config__defaults_are_valid(self) -> None:
        config = RootConfig()
        assert config.app.timezone == "Asia/Shanghai"
        assert config.execution.broker == "paper"
        assert config.execution.require_manual_confirm is True

    def test_defaults_are_conservative(self) -> None:
        """涉及真实资金的默认值必须取最保守一侧。"""
        config = RootConfig()
        assert config.execution.broker == "paper"
        assert config.llm.enabled is False
        assert config.risk.hard_limits.enabled is True
        assert config.advisor.require_full_rationale is True
        assert config.factors.label.exclude_unbuyable_at_entry is True

    def test_unknown_key__rejected(self) -> None:
        """拼错的配置项必须报错，而不是被静默忽略。"""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            RootConfig.model_validate({"app": {"log_levl": "INFO"}})


class TestHardLimits:
    def test_single_order_over_daily_total__rejected(self) -> None:
        with pytest.raises(ValidationError, match="不应大于单日上限"):
            HardLimitConfig(
                max_single_order_amount=Decimal("500000"),
                max_daily_total_amount=Decimal("300000"),
            )

    def test_sanity_bounds_inverted__rejected(self) -> None:
        with pytest.raises(ValidationError, match="下限必须小于上限"):
            HardLimitConfig(
                min_account_value_sanity=Decimal("100000"),
                max_account_value_sanity=Decimal("1000"),
            )

    @pytest.mark.parametrize("amount", [Decimal("0"), Decimal("-1")])
    def test_non_positive_amount__rejected(self, amount: Decimal) -> None:
        with pytest.raises(ValidationError):
            HardLimitConfig(max_single_order_amount=amount)


class TestCircuitBreaker:
    def test_watch_threshold_must_be_tighter_than_halted(self) -> None:
        with pytest.raises(ValidationError, match="WATCH 的当日亏损阈值必须小于"):
            CircuitBreakerConfig(watch_daily_loss=0.06, halted_daily_loss=0.05)

    def test_watch_drawdown_must_be_tighter(self) -> None:
        with pytest.raises(ValidationError, match="WATCH 的回撤阈值必须小于"):
            CircuitBreakerConfig(watch_drawdown_20d=0.20, halted_drawdown_20d=0.18)


class TestExecution:
    @pytest.mark.parametrize("broker", ["qmt", "ptrade"])
    def test_live_broker__cannot_disable_manual_confirm(self, broker: str) -> None:
        """红线 R5：真实通道下人工确认不可关闭。"""
        with pytest.raises(ValidationError, match="不可关闭"):
            ExecutionConfig.model_validate({"broker": broker, "require_manual_confirm": False})

    def test_paper_broker__may_disable_manual_confirm(self) -> None:
        config = ExecutionConfig(broker="paper", require_manual_confirm=False)
        assert config.require_manual_confirm is False


class TestLLM:
    def test_influence_above_hard_cap__rejected(self) -> None:
        """红线 LR2：α 硬上限 0.20。"""
        with pytest.raises(ValidationError):
            LLMConfig(max_influence=0.5)

    def test_influence_at_cap__accepted(self) -> None:
        assert LLMConfig(max_influence=0.20).max_influence == 0.20

    def test_disabled__forces_mode_off(self) -> None:
        """总开关关闭时不允许 mode 仍为 live，避免配置自相矛盾。"""
        config = LLMConfig(enabled=False, mode="live")
        assert config.mode == "off"

    def test_default_disabled(self) -> None:
        """默认关闭：需先通过 A/B 回测证明有净增量再启用（LR6）。"""
        assert LLMConfig().enabled is False


class TestPortfolio:
    def test_core_satellite_must_sum_to_one(self) -> None:
        with pytest.raises(ValidationError, match="必须为 1"):
            PortfolioConfig(core_ratio=0.7, satellite_ratio=0.4)

    def test_valid_split(self) -> None:
        assert PortfolioConfig(core_ratio=0.5, satellite_ratio=0.5).core_ratio == 0.5


class TestLoading:
    def test_load_config__repo_base_yaml__valid(self) -> None:
        """仓库自带的 config/base.yaml 必须与模型完全对齐。"""
        config = load_config(Path("config"))
        assert config.app.timezone == "Asia/Shanghai"

    def test_load_config__missing_dir__uses_defaults(self, tmp_path: Path) -> None:
        config = load_config(tmp_path)
        assert config == RootConfig()

    def test_local_overrides_base(self, tmp_path: Path) -> None:
        (tmp_path / "base.yaml").write_text(
            yaml.safe_dump({"app": {"log_level": "INFO", "random_seed": 1}}),
            encoding="utf-8",
        )
        (tmp_path / "local.yaml").write_text(
            yaml.safe_dump({"app": {"log_level": "DEBUG"}}), encoding="utf-8"
        )
        config = load_config(tmp_path)
        assert config.app.log_level == "DEBUG"
        assert config.app.random_seed == 1  # 未覆盖的键保留

    def test_invalid_yaml__raises_config_error(self, tmp_path: Path) -> None:
        (tmp_path / "base.yaml").write_text("app: [not, a, mapping", encoding="utf-8")
        with pytest.raises(ConfigError, match="解析失败"):
            load_config(tmp_path)

    def test_non_mapping_top_level__raises(self, tmp_path: Path) -> None:
        (tmp_path / "base.yaml").write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="顶层必须是映射"):
            load_config(tmp_path)

    def test_invalid_value__raises_config_error(self, tmp_path: Path) -> None:
        (tmp_path / "base.yaml").write_text(
            yaml.safe_dump({"llm": {"max_influence": 0.9}}), encoding="utf-8"
        )
        with pytest.raises(ConfigError, match="配置校验失败"):
            load_config(tmp_path)


class TestSecrets:
    def test_unset_secret__reports_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
        secrets = Secrets(_env_file=None)
        assert secrets.has("tushare_token") is False
        assert secrets.get("tushare_token") is None

    def test_set_secret__readable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TUSHARE_TOKEN", "abcd1234")
        secrets = Secrets(_env_file=None)
        assert secrets.has("tushare_token")
        assert secrets.get("tushare_token") == "abcd1234"

    def test_unknown_field__raises(self) -> None:
        secrets = Secrets(_env_file=None)
        with pytest.raises(ConfigError, match="未知的密钥字段"):
            secrets.has("nonexistent")

    def test_repr_does_not_leak_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """红线 R7：密钥不得出现在日志或异常里。"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret-value")
        secrets = Secrets(_env_file=None)
        assert "sk-super-secret-value" not in repr(secrets)


class TestSettings:
    def test_redacted_dump__never_contains_secret_values(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret-value")
        settings = load_settings(tmp_path)
        dumped = repr(settings.redacted_dump())
        assert "sk-super-secret-value" not in dumped
        assert settings.redacted_dump()["secrets_configured"]["anthropic_api_key"] is True


class TestJsonSchema:
    """界面配置表单由 JSON Schema 自动生成，Schema 的完备性是界面可用性的前提。"""

    def test_schema_generates(self) -> None:
        schema = RootConfig.model_json_schema()
        assert "properties" in schema
        assert set(schema["properties"]) >= {"app", "data", "risk", "llm", "execution"}

    def test_every_field_has_description(self) -> None:
        """没有 description 的字段在界面上就是一个没有说明的裸输入框。"""
        schema = RootConfig.model_json_schema()
        missing: list[str] = []
        for def_name, definition in schema.get("$defs", {}).items():
            for field_name, field in definition.get("properties", {}).items():
                if not field.get("description") and "$ref" not in field:
                    missing.append(f"{def_name}.{field_name}")
        assert not missing, f"以下配置字段缺少 description：{missing}"
