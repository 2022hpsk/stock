"""配置加载与密钥管理。

规范见 docs/01-开发规范.md 第八条：

- 密钥**只从环境变量读取**，业务代码禁止直接访问 ``os.environ``。
- 配置分层合并：``base.yaml`` → ``local.yaml`` → 环境变量。
- 启动时打印**脱敏后**的生效配置。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from quantstock.config.models import RootConfig
from quantstock.infra.errors import ConfigError

__all__ = ["Secrets", "Settings", "load_config", "load_settings"]


class Secrets(BaseSettings):
    """密钥。只从环境变量与 ``.env`` 读取，永不落入版本库（红线 R7）。

    所有字段用 ``SecretStr``，避免误打印明文。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    tushare_token: SecretStr | None = Field(default=None, description="Tushare Pro token。")
    jin10_api_key: SecretStr | None = Field(default=None, description="金十数据开放平台 Key。")
    jin10_api_secret: SecretStr | None = Field(default=None, description="金十数据 Secret。")
    anthropic_api_key: SecretStr | None = Field(default=None, description="Claude API Key。")
    intel_ingest_token: SecretStr | None = Field(
        default=None, description="情报 HTTP 接收端的鉴权 token。"
    )
    notify_email_password: SecretStr | None = Field(default=None, description="邮件密码。")
    wecom_webhook_url: SecretStr | None = Field(default=None, description="企业微信机器人。")
    telegram_bot_token: SecretStr | None = Field(default=None, description="Telegram Bot。")
    xtquant_account_id: SecretStr | None = Field(
        default=None, description="券商资金账号。日志中必须脱敏。"
    )

    def has(self, name: str) -> bool:
        """是否配置了某个密钥。

        Args:
            name: 字段名。

        Returns:
            已配置且非空则 True。

        Raises:
            ConfigError: 字段名不存在。
        """
        if name not in type(self).model_fields:
            msg = f"未知的密钥字段：{name}"
            raise ConfigError(msg, known=sorted(type(self).model_fields))
        value: SecretStr | None = getattr(self, name)
        return value is not None and bool(value.get_secret_value())

    def get(self, name: str) -> str | None:
        """取密钥明文。

        调用点应尽量靠近实际使用处，避免明文在内存中长期传递。

        Args:
            name: 字段名。

        Returns:
            明文密钥；未配置则 None。

        Raises:
            ConfigError: 字段名不存在。
        """
        if not self.has(name):
            return None
        value: SecretStr = getattr(self, name)
        return value.get_secret_value()


class Settings:
    """运行期配置聚合。

    Attributes:
        config: 业务配置（可在界面上编辑）。
        secrets: 密钥（只读自环境变量）。
        config_dir: 配置文件目录。
    """

    def __init__(
        self,
        config: RootConfig,
        secrets: Secrets,
        *,
        config_dir: Path,
    ) -> None:
        """初始化。

        Args:
            config: 已合并校验的业务配置。
            secrets: 密钥。
            config_dir: 配置文件目录。
        """
        self.config = config
        self.secrets = secrets
        self.config_dir = config_dir

    @property
    def var_dir(self) -> Path:
        """运行时数据目录（绝对路径）。"""
        return Path(self.config.app.var_dir).resolve()

    def redacted_dump(self) -> dict[str, Any]:
        """导出脱敏后的生效配置，用于启动日志与界面展示。

        Returns:
            配置字典，密钥字段以是否已配置的布尔值代替明文。
        """
        dumped: dict[str, Any] = self.config.model_dump(mode="json")
        dumped["secrets_configured"] = {
            name: self.secrets.has(name) for name in sorted(type(self.secrets).model_fields)
        }
        return dumped


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两层配置字典。

    ``override`` 中的标量与列表整体覆盖 ``base``，字典逐键递归合并。
    列表不做合并——半个列表的语义几乎总是错的。

    Args:
        base: 底层配置。
        override: 覆盖层配置。

    Returns:
        合并后的新字典，输入不被修改。
    """
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 文件。

    Args:
        path: 文件路径。

    Returns:
        解析出的字典；文件不存在返回空字典。

    Raises:
        ConfigError: 文件存在但内容不是字典或解析失败。
    """
    if not path.exists():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        msg = f"配置文件解析失败：{path}"
        raise ConfigError(msg, error=str(exc)) from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        msg = f"配置文件顶层必须是映射：{path}"
        raise ConfigError(msg, actual_type=type(raw).__name__)
    return raw


def load_config(config_dir: Path | str = "config") -> RootConfig:
    """加载并校验业务配置。

    合并顺序：``base.yaml`` → ``local.yaml``（后者覆盖前者）。

    Args:
        config_dir: 配置目录。

    Returns:
        校验通过的配置对象。

    Raises:
        ConfigError: 配置非法。错误信息会指出具体是哪个字段。
    """
    directory = Path(config_dir)
    merged = _deep_merge(
        _read_yaml(directory / "base.yaml"),
        _read_yaml(directory / "local.yaml"),
    )
    try:
        return RootConfig.model_validate(merged)
    except Exception as exc:  # pydantic.ValidationError 及其它校验异常
        msg = f"配置校验失败（{directory}）"
        raise ConfigError(msg, error=str(exc)) from exc


def load_settings(config_dir: Path | str = "config") -> Settings:
    """加载完整运行期配置（业务配置 + 密钥）。

    Args:
        config_dir: 配置目录。

    Returns:
        Settings 实例。
    """
    directory = Path(config_dir)
    return Settings(
        config=load_config(directory),
        secrets=Secrets(),
        config_dir=directory,
    )
