"""急停开关（风控规则 A12）与绝对金额硬闸（A10/A11）。

规范见 docs/05-风控规范.md §1.1、§1.2。

两条设计要点：

1. **急停检查必须放在最靠近下单的位置**（``execution.submit()`` 的第一行），
   任何代码路径都绕不过。
2. **硬闸独立于比例风控**。若账户总资产被算成真实值的 10 倍，所有比例约束仍会通过，
   但实际下单金额是灾难性的——比例风控无法防御计算基数本身出错。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from quantstock.config.models import HardLimitConfig
from quantstock.infra.clock import now
from quantstock.infra.errors import HardLimitExceededError, TradingHaltedError
from quantstock.infra.logging import get_logger
from quantstock.infra.money import ZERO
from quantstock.infra.types import Money

__all__ = ["HaltState", "HaltSwitch", "HardLimitCheck", "HardLimitGuard"]

_log = get_logger(__name__)

HALT_FILENAME = "HALT"
"""急停标志文件名。存在即拒绝一切下单。"""


@dataclass(frozen=True, slots=True)
class HaltState:
    """急停状态。"""

    halted: bool
    reason: str = ""
    halted_at: str = ""
    halted_by: str = ""


class HaltSwitch:
    """急停开关。

    用**文件**而非内存状态表示急停，理由：进程重启后仍然生效，
    且用户可以直接 ``rm var/HALT`` 手工解除——出事时最不该依赖的就是程序本身还正常。
    """

    def __init__(self, var_dir: Path) -> None:
        """初始化。

        Args:
            var_dir: 运行时数据目录。
        """
        self._path = var_dir / HALT_FILENAME

    @property
    def path(self) -> Path:
        """急停标志文件路径。"""
        return self._path

    def is_halted(self) -> bool:
        """当前是否处于急停。

        Returns:
            标志文件存在则 True。
        """
        return self._path.exists()

    def state(self) -> HaltState:
        """读取急停状态详情。

        Returns:
            急停状态。标志文件损坏时仍视为已急停——
            无法确认状态时必须取保守一侧。
        """
        if not self._path.exists():
            return HaltState(halted=False)
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return HaltState(halted=True, reason="急停标志文件存在但无法解析")
        return HaltState(
            halted=True,
            reason=str(payload.get("reason", "")),
            halted_at=str(payload.get("halted_at", "")),
            halted_by=str(payload.get("halted_by", "")),
        )

    def halt(self, *, reason: str, by: str = "cli") -> HaltState:
        """触发急停。

        Args:
            reason: 急停原因，必填——事后复盘要靠它。
            by: 操作人标识。

        Returns:
            急停后的状态。

        Raises:
            ValueError: 未填写原因。
        """
        if not reason.strip():
            msg = "急停必须填写原因"
            raise ValueError(msg)
        state = HaltState(halted=True, reason=reason, halted_at=now().isoformat(), halted_by=by)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {
                    "reason": state.reason,
                    "halted_at": state.halted_at,
                    "halted_by": state.halted_by,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _log.critical("trading_halted", reason=reason, halted_by=by)
        return state

    def resume(self, *, by: str = "cli") -> None:
        """解除急停。

        Args:
            by: 操作人标识。
        """
        if self._path.exists():
            previous = self.state()
            self._path.unlink()
            _log.warning("trading_resumed", previous_reason=previous.reason, resumed_by=by)

    def ensure_not_halted(self) -> None:
        """断言当前未急停。应在任何下单路径的**第一行**调用。

        Raises:
            TradingHaltedError: 处于急停状态。
        """
        state = self.state()
        if state.halted:
            msg = "系统处于急停状态，拒绝下单"
            raise TradingHaltedError(
                msg, reason=state.reason, halted_at=state.halted_at, halt_file=str(self._path)
            )


@dataclass(frozen=True, slots=True)
class HardLimitCheck:
    """单项硬闸检查结果。"""

    name: str
    passed: bool
    limit: Decimal
    actual: Decimal
    message: str


@dataclass(frozen=True, slots=True)
class HardLimitResult:
    """硬闸整体检查结果。"""

    passed: bool
    checks: tuple[HardLimitCheck, ...]

    @property
    def failures(self) -> tuple[HardLimitCheck, ...]:
        """未通过的检查项。"""
        return tuple(c for c in self.checks if not c.passed)


class HardLimitGuard:
    """绝对金额硬闸。

    与 ``RiskEngine`` 的比例约束**用不同代码路径实现**，形成双保险——
    共用实现会导致共同失效。

    任一项超限即中止**整个计划**而非跳过单笔：单笔超限通常意味着计算基数出错，
    此时其余单笔同样不可信。
    """

    def __init__(self, config: HardLimitConfig) -> None:
        """初始化。

        Args:
            config: 硬闸阈值。必须由用户按实际资金规模手工设定。
        """
        self._config = config

    def check_account_sanity(
        self, *, total_value: Money, previous_value: Money | None = None
    ) -> HardLimitResult:
        """账户数据合理性校验（规则 A11）。

        总资产落在合理区间之外、或单日变动过大且无对应资金流水，
        通常意味着账户同步出了问题——此时任何下单都是危险的。

        Args:
            total_value: 当前账户总资产。
            previous_value: 上一交易日总资产，用于变动幅度校验。

        Returns:
            检查结果。
        """
        cfg = self._config
        checks = [
            HardLimitCheck(
                name="account_value_min",
                passed=total_value >= cfg.min_account_value_sanity,
                limit=cfg.min_account_value_sanity,
                actual=total_value,
                message=f"账户总资产 {total_value} 低于合理性下限 {cfg.min_account_value_sanity}",
            ),
            HardLimitCheck(
                name="account_value_max",
                passed=total_value <= cfg.max_account_value_sanity,
                limit=cfg.max_account_value_sanity,
                actual=total_value,
                message=f"账户总资产 {total_value} 高于合理性上限 {cfg.max_account_value_sanity}",
            ),
        ]

        if previous_value is not None and previous_value > 0:
            change = abs(total_value - previous_value) / previous_value
            threshold = Decimal(str(cfg.max_account_value_daily_change))
            checks.append(
                HardLimitCheck(
                    name="account_value_daily_change",
                    passed=change <= threshold,
                    limit=threshold,
                    actual=change,
                    message=(
                        f"账户总资产单日变动 {change:.2%} 超过阈值 {threshold:.2%}，"
                        f"且无对应资金流水，判定为账户同步异常"
                    ),
                )
            )

        return _summarize(checks)

    def check_orders(
        self,
        *,
        order_amounts: list[Money],
        order_quantities: list[int] | None = None,
    ) -> HardLimitResult:
        """委托金额与笔数校验（规则 A10）。

        Args:
            order_amounts: 本批全部委托的金额。
            order_quantities: 本批全部委托的股数。

        Returns:
            检查结果。
        """
        cfg = self._config
        if not cfg.enabled:
            return HardLimitResult(passed=True, checks=())

        max_amount = max(order_amounts, default=ZERO)
        total_amount = sum(order_amounts, start=ZERO)
        count = len(order_amounts)
        max_qty = max(order_quantities, default=0) if order_quantities else 0

        checks = [
            HardLimitCheck(
                name="max_single_order_amount",
                passed=max_amount <= cfg.max_single_order_amount,
                limit=cfg.max_single_order_amount,
                actual=max_amount,
                message=(f"单笔委托金额 {max_amount} 超过硬闸上限 {cfg.max_single_order_amount}"),
            ),
            HardLimitCheck(
                name="max_daily_total_amount",
                passed=total_amount <= cfg.max_daily_total_amount,
                limit=cfg.max_daily_total_amount,
                actual=total_amount,
                message=(
                    f"当批委托总金额 {total_amount} 超过单日上限 {cfg.max_daily_total_amount}"
                ),
            ),
            HardLimitCheck(
                name="max_daily_order_count",
                passed=count <= cfg.max_daily_order_count,
                limit=Decimal(cfg.max_daily_order_count),
                actual=Decimal(count),
                message=f"委托笔数 {count} 超过单日上限 {cfg.max_daily_order_count}",
            ),
            HardLimitCheck(
                name="max_single_order_qty",
                passed=max_qty <= cfg.max_single_order_qty,
                limit=Decimal(cfg.max_single_order_qty),
                actual=Decimal(max_qty),
                message=f"单笔委托股数 {max_qty} 超过上限 {cfg.max_single_order_qty}",
            ),
        ]
        return _summarize(checks)

    def enforce(self, result: HardLimitResult) -> None:
        """硬闸不通过时中止整个计划。

        Args:
            result: 检查结果。

        Raises:
            HardLimitExceededError: 存在未通过的检查项。
        """
        if result.passed:
            return
        failures = result.failures
        _log.critical(
            "hard_limit_exceeded",
            failures=[f.name for f in failures],
            details=[f.message for f in failures],
        )
        msg = "触发绝对金额硬闸，已中止整个交易计划"
        raise HardLimitExceededError(
            msg,
            failed_checks=[f.name for f in failures],
            details=[f.message for f in failures],
        )


def _summarize(checks: list[HardLimitCheck]) -> HardLimitResult:
    """汇总检查项。

    Args:
        checks: 各检查项。

    Returns:
        整体结果。
    """
    return HardLimitResult(passed=all(c.passed for c in checks), checks=tuple(checks))
