"""风控服务：规则目录、熔断状态、绝对硬闸、拒绝记录（docs/09 P10）。

这一层刻意**只读**（急停除外）。风控阈值的修改走配置服务，那条路径有
校验、Diff 预览、备份与审计；给风控单开一条"快速调阈值"的接口，
等于把最需要留痕的操作做成了最容易悄悄做掉的操作。

规则表来自 ``risk.catalogue``，它声明了每条规则能不能关。界面据此
**根本不渲染** A 类的开关——画出来再拒绝和根本不画是两回事，
前者会让人一直去试（验收 5）。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from quantstock.config.settings import Settings
from quantstock.infra.logging import get_logger
from quantstock.infra.types import Money
from quantstock.risk.catalogue import RULES, RuleClass, RuleSpec
from quantstock.risk.engine import CircuitState
from quantstock.risk.halt import HaltState, HaltSwitch, HardLimitGuard

# 界面与 CLI 只允许依赖 services（F20.1 分层契约），风控契约类型在这里转出
__all__ = ["CircuitState", "HaltState", "RiskService", "RuleClass", "RuleSpec", "RuleView"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RuleView:
    """规则表的一行。"""

    rule_id: str
    name: str
    rule_class: str
    description: str
    closable: bool
    threshold_editable: bool
    threshold_key: str
    current_threshold: str


@dataclass(frozen=True, slots=True)
class HardLimitView:
    """绝对金额硬闸的当前设置。"""

    enabled: bool
    max_single_order_amount: Money
    max_daily_total_amount: Money
    max_daily_order_count: int
    min_account_value_sanity: Money
    max_account_value_sanity: Money

    @property
    def message(self) -> str:
        """一行摘要。

        Returns:
            摘要文本。
        """
        if not self.enabled:
            return "绝对金额硬闸已关闭——比例风控挡不住计算基数出错，真实通道下不应关闭"
        return (
            f"单笔 ≤ {self.max_single_order_amount}，单日 ≤ {self.max_daily_total_amount}，"
            f"笔数 ≤ {self.max_daily_order_count}"
        )


class RiskService:
    """风控查询编排。"""

    def __init__(self, settings: Settings) -> None:
        """初始化。

        Args:
            settings: 运行期配置。
        """
        self._settings = settings
        self._halt = HaltSwitch(settings.var_dir)
        self._guard = HardLimitGuard(settings.config.risk.hard_limits)

    @property
    def halt_switch(self) -> HaltSwitch:
        """急停开关。"""
        return self._halt

    def halt_state(self) -> HaltState:
        """当前急停状态。

        Returns:
            急停状态。标志文件损坏时仍视为已急停——
            无法确认状态时必须取保守一侧。
        """
        return self._halt.state()

    def rules(self) -> list[RuleView]:
        """规则表。

        Returns:
            全部规则，按编号排序。每条都带 ``closable``，
            界面必须据此决定要不要渲染开关。
        """
        return [
            RuleView(
                rule_id=spec.rule_id,
                name=spec.name,
                rule_class=spec.rule_class.value,
                description=spec.description,
                closable=spec.closable,
                threshold_editable=spec.threshold_editable,
                threshold_key=spec.threshold_key,
                current_threshold=self._threshold_of(spec),
            )
            for spec in sorted(RULES, key=lambda r: r.rule_id)
        ]

    def hard_limits(self) -> HardLimitView:
        """绝对金额硬闸的当前设置。

        Returns:
            设置视图。
        """
        cfg = self._settings.config.risk.hard_limits
        return HardLimitView(
            enabled=cfg.enabled,
            max_single_order_amount=cfg.max_single_order_amount,
            max_daily_total_amount=cfg.max_daily_total_amount,
            max_daily_order_count=cfg.max_daily_order_count,
            min_account_value_sanity=cfg.min_account_value_sanity,
            max_account_value_sanity=cfg.max_account_value_sanity,
        )

    def check_account_sanity(
        self, *, total_value: Money, previous_value: Money | None = None
    ) -> dict[str, object]:
        """账户合理性校验（A11）。

        Args:
            total_value: 当前总资产。
            previous_value: 上一交易日总资产。

        Returns:
            逐项结果。
        """
        result = self._guard.check_account_sanity(
            total_value=total_value, previous_value=previous_value
        )
        return {
            "passed": not result.failures,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "limit": str(c.limit),
                    "actual": str(c.actual),
                    "message": c.message,
                }
                for c in result.checks
            ],
        }

    def circuit_distance(
        self, *, daily_return: Decimal, drawdown_20d: Decimal
    ) -> dict[str, object]:
        """距各熔断阈值还有多远。

        界面上画成仪表盘。**显示距离而不只显示状态**：状态只有到了才变，
        而距离能让人提前看到自己正在往哪走。

        Args:
            daily_return: 当日收益率（负值为亏损）。
            drawdown_20d: 20 日回撤（负值）。

        Returns:
            各阈值与当前值的对照。
        """
        cfg = self._settings.config.risk.circuit_breaker
        loss = -daily_return
        dd = -drawdown_20d
        return {
            "daily_loss": {
                "current": float(loss),
                "watch": float(cfg.watch_daily_loss),
                "halted": float(cfg.halted_daily_loss),
            },
            "drawdown_20d": {
                "current": float(dd),
                "watch": float(cfg.watch_drawdown_20d),
                "halted": float(cfg.halted_drawdown_20d),
            },
            "recover_drawdown": float(cfg.recover_drawdown),
            # HALTED 不会自动恢复。界面必须说清楚，否则用户会一直等着它自己好
            "auto_recovers": False,
        }

    def _threshold_of(self, spec: RuleSpec) -> str:
        """取规则当前生效的阈值。

        Args:
            spec: 规则声明。

        Returns:
            阈值的可读表示；取不到时空串。
        """
        if not spec.threshold_key:
            return ""
        cursor: object = self._settings.config
        for part in spec.threshold_key.split("."):
            if part.startswith("<"):
                return "按账户配置"
            cursor = getattr(cursor, part, None)
            if cursor is None:
                return ""
        if hasattr(cursor, "model_dump"):
            return "见配置页"
        return str(cursor)
