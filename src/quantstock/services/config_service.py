"""配置服务。

规范见 docs/09-可视化界面规格.md 第四节。

**"所有配置都能在界面上改"的实现基础**：配置模型导出 JSON Schema，
前端据此自动生成表单；保存前做校验、Diff 预览、自动备份，保存后可回滚。
新增配置项只需改 pydantic 模型，界面自动出现对应控件。
"""

from __future__ import annotations

import difflib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from quantstock.config.models import RootConfig
from quantstock.config.settings import Secrets, load_config
from quantstock.infra.clock import now
from quantstock.infra.errors import ConfigError
from quantstock.infra.logging import get_logger

__all__ = ["ConfigService", "SaveResult", "ValidationIssue"]

_log = get_logger(__name__)

LOCAL_CONFIG_NAME = "local.yaml"
"""界面上的修改一律写入本地覆盖层，不动仓库里的 base.yaml。"""


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """一条配置校验错误。

    ``location`` 用点号路径（如 ``risk.hard_limits.max_single_order_amount``），
    便于界面定位到具体控件并高亮。
    """

    location: str
    message: str
    input_value: str


@dataclass(frozen=True, slots=True)
class SaveResult:
    """配置保存结果。"""

    saved: bool
    diff: str
    backup_path: str | None
    issues: tuple[ValidationIssue, ...] = ()


class ConfigService:
    """配置的读取、校验、保存与回滚。

    UI 与 CLI 都调用本服务，保证"界面上改的"和"命令行改的"行为完全一致。
    """

    def __init__(self, config_dir: Path, var_dir: Path) -> None:
        """初始化。

        Args:
            config_dir: 配置目录。
            var_dir: 运行时数据目录，备份写在其下。
        """
        self._config_dir = Path(config_dir)
        self._backup_dir = Path(var_dir) / "config_backups"

    @property
    def local_path(self) -> Path:
        """本地覆盖配置文件路径。"""
        return self._config_dir / LOCAL_CONFIG_NAME

    # ------------------------------------------------------------------ 读取
    @staticmethod
    def json_schema() -> dict[str, Any]:
        """导出配置的 JSON Schema。

        界面表单由它自动生成——字段的 ``description`` 就是界面上的说明文字，
        ``minimum``/``maximum``/``enum`` 直接变成控件的校验规则。

        Returns:
            JSON Schema 字典。
        """
        return RootConfig.model_json_schema()

    def current(self) -> RootConfig:
        """取当前生效配置（base + local 合并后）。

        Returns:
            配置对象。

        Raises:
            ConfigError: 配置非法。
        """
        return load_config(self._config_dir)

    def current_dict(self) -> dict[str, Any]:
        """取当前生效配置的字典形式，供界面渲染。

        Returns:
            配置字典（JSON 可序列化）。
        """
        return self.current().model_dump(mode="json")

    def local_overrides(self) -> dict[str, Any]:
        """取本地覆盖层的原始内容。

        Returns:
            覆盖项字典；文件不存在时返回空字典。
        """
        if not self.local_path.exists():
            return {}
        raw = yaml.safe_load(self.local_path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}

    # ------------------------------------------------------------------ 校验
    @staticmethod
    def validate(candidate: dict[str, Any]) -> tuple[ValidationIssue, ...]:
        """校验一份完整配置。

        Args:
            candidate: 待校验的完整配置字典。

        Returns:
            校验错误列表；全部通过时为空元组。
        """
        try:
            RootConfig.model_validate(candidate)
        except ValidationError as exc:
            return tuple(
                ValidationIssue(
                    location=".".join(str(p) for p in err["loc"]),
                    message=err["msg"],
                    input_value=str(err.get("input", "")),
                )
                for err in exc.errors()
            )
        return ()

    # ------------------------------------------------------------------ 保存
    def preview(self, candidate: dict[str, Any]) -> str:
        """生成保存前的 YAML Diff 预览。

        让用户在点确认之前，先看清楚这次到底改了什么——
        风控阈值这类配置改错的代价很高。

        Args:
            candidate: 待保存的完整配置字典。

        Returns:
            统一格式的 diff 文本；无变化时为空串。
        """
        before = _dump_yaml(self.current_dict())
        after = _dump_yaml(candidate)
        if before == after:
            return ""
        return "".join(
            difflib.unified_diff(
                before.splitlines(keepends=True),
                after.splitlines(keepends=True),
                fromfile="当前配置",
                tofile="修改后",
                n=2,
            )
        )

    def save(
        self, candidate: dict[str, Any], *, changed_by: str = "ui", dry_run: bool = False
    ) -> SaveResult:
        """校验并保存配置。

        流程：校验 → 生成 Diff → 备份旧配置 → 写入 local.yaml → 记日志。

        Args:
            candidate: 待保存的完整配置字典。
            changed_by: 操作人标识，写入日志用于审计。
            dry_run: 只校验与预览，不实际写入。

        Returns:
            保存结果。校验不通过时 ``saved=False`` 且带 ``issues``。
        """
        issues = self.validate(candidate)
        if issues:
            return SaveResult(saved=False, diff="", backup_path=None, issues=issues)

        diff = self.preview(candidate)
        if dry_run or not diff:
            return SaveResult(saved=False, diff=diff, backup_path=None)

        backup = self._backup()
        # 保存完整配置而非增量——界面上看到的就是存下来的，避免"改了 base.yaml 后
        # local.yaml 的语义悄悄变化"这类难查的问题。
        normalized = RootConfig.model_validate(candidate).model_dump(mode="json")
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.local_path.write_text(_dump_yaml(normalized), encoding="utf-8")

        _log.info(
            "config_saved",
            changed_by=changed_by,
            backup=str(backup) if backup else None,
            changed_lines=diff.count("\n"),
        )
        return SaveResult(saved=True, diff=diff, backup_path=str(backup) if backup else None)

    def _backup(self) -> Path | None:
        """备份当前的 local.yaml。

        Returns:
            备份文件路径；无本地配置可备份时返回 None。
        """
        if not self.local_path.exists():
            return None
        stamp = now().strftime("%Y%m%d-%H%M%S")
        target_dir = self._backup_dir / stamp
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / LOCAL_CONFIG_NAME
        shutil.copy2(self.local_path, target)
        return target

    # ------------------------------------------------------------------ 回滚
    def backups(self) -> list[str]:
        """列出可回滚的备份版本。

        Returns:
            备份时间戳列表，最新的在前。
        """
        if not self._backup_dir.exists():
            return []
        return sorted((p.name for p in self._backup_dir.iterdir() if p.is_dir()), reverse=True)

    def rollback(self, version: str, *, changed_by: str = "ui") -> SaveResult:
        """回滚到指定备份版本。

        Args:
            version: 备份时间戳，来自 :meth:`backups`。
            changed_by: 操作人标识。

        Returns:
            保存结果。

        Raises:
            ConfigError: 备份不存在或内容非法。
        """
        source = self._backup_dir / version / LOCAL_CONFIG_NAME
        if not source.exists():
            msg = "备份版本不存在"
            raise ConfigError(msg, version=version, available=self.backups()[:5])

        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = "备份内容非法"
            raise ConfigError(msg, version=version)

        merged = load_config(self._config_dir).model_dump(mode="json")
        merged.update(raw)
        return self.save(merged, changed_by=f"{changed_by}:rollback:{version}")

    # ------------------------------------------------------------------ 密钥
    @staticmethod
    def secrets_status() -> dict[str, bool]:
        """各密钥是否已配置。

        **只返回布尔值，永不返回明文**——后端不提供读取密钥明文的接口
        （见 docs/09-可视化界面规格.md §4.4）。

        Returns:
            密钥字段名到"是否已配置"的映射。
        """
        secrets = Secrets()
        return {name: secrets.has(name) for name in sorted(type(secrets).model_fields)}


def _dump_yaml(data: dict[str, Any]) -> str:
    """按稳定顺序导出 YAML。

    Args:
        data: 待导出的字典。

    Returns:
        YAML 文本。
    """
    return yaml.safe_dump(
        data, allow_unicode=True, sort_keys=True, default_flow_style=False, indent=2
    )
