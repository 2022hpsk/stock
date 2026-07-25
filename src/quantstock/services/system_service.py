"""系统服务：健康状态、急停开关。

UI 与 CLI 共用同一实现（见 docs/09-可视化界面规格.md §1.1）。
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

from quantstock.config.settings import Settings
from quantstock.infra.clock import now
from quantstock.risk.halt import HaltState, HaltSwitch

__all__ = ["ComponentHealth", "SystemService", "SystemStatus"]


@dataclass(frozen=True, slots=True)
class ComponentHealth:
    """单个组件的健康状态。"""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SystemStatus:
    """系统整体状态，供仪表盘展示。"""

    version: str
    checked_at: str
    halt: HaltState
    broker: str
    llm_enabled: bool
    llm_mode: str
    components: tuple[ComponentHealth, ...]

    @property
    def ok(self) -> bool:
        """是否全部组件正常且未急停。"""
        return not self.halt.halted and all(c.ok for c in self.components)


class SystemService:
    """系统状态与急停控制。"""

    def __init__(self, settings: Settings) -> None:
        """初始化。

        Args:
            settings: 运行期配置。
        """
        self._settings = settings
        self._halt = HaltSwitch(settings.var_dir)

    @property
    def halt_switch(self) -> HaltSwitch:
        """急停开关。"""
        return self._halt

    def status(self) -> SystemStatus:
        """采集系统状态。

        Returns:
            系统状态快照。
        """
        try:
            version = pkg_version("quantstock")
        except PackageNotFoundError:  # pragma: no cover - 开发模式
            version = "dev"

        config = self._settings.config
        return SystemStatus(
            version=version,
            checked_at=now().isoformat(),
            halt=self._halt.state(),
            broker=config.execution.broker,
            llm_enabled=config.llm.enabled,
            llm_mode=config.llm.mode,
            components=self._check_components(),
        )

    def _check_components(self) -> tuple[ComponentHealth, ...]:
        """逐组件自检。

        Returns:
            各组件健康状态。
        """
        var_dir = self._settings.var_dir
        checks = [
            ComponentHealth(
                name="var_dir",
                ok=_is_writable(var_dir),
                detail=f"运行时数据目录 {var_dir}",
            ),
            ComponentHealth(
                name="config",
                ok=True,
                detail=f"配置目录 {self._settings.config_dir}",
            ),
            ComponentHealth(
                name="hard_limits",
                ok=self._settings.config.risk.hard_limits.enabled,
                detail=(
                    "绝对金额硬闸已启用"
                    if self._settings.config.risk.hard_limits.enabled
                    else "绝对金额硬闸未启用——真实交易前必须开启"
                ),
            ),
        ]
        return tuple(checks)

    def halt(self, *, reason: str, by: str) -> HaltState:
        """触发急停。

        Args:
            reason: 急停原因，必填。
            by: 操作人标识。

        Returns:
            急停后的状态。
        """
        return self._halt.halt(reason=reason, by=by)

    def resume(self, *, by: str) -> HaltState:
        """解除急停。

        Args:
            by: 操作人标识。

        Returns:
            解除后的状态。
        """
        self._halt.resume(by=by)
        return self._halt.state()


def _is_writable(path: Path) -> bool:
    """目录是否可写。

    Args:
        path: 目录路径，不存在时会尝试创建。

    Returns:
        可写则 True。
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError:
        return False
    return True
