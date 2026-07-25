"""成本预算与自动降级（红线 LR7）。

规范见 docs/10-大模型集成规格.md 第七节。

超预算时**降级到 ``off`` 而不是抛错**：LLM 是增强项，账单超了不该让当天
出不了建议。降级会告警并记录，用户看得到"今天的建议没有 LLM 参与"。

用量按自然日/自然月记账，落 ``var/llm_usage/``，界面可看趋势。
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from quantstock.infra.clock import now
from quantstock.infra.logging import get_logger

__all__ = ["BudgetGuard", "BudgetState", "UsageRecord", "estimate_cost"]

_log = get_logger(__name__)

MODEL_PRICING: dict[str, tuple[float, float]] = {
    # 模型 ID → (输入价, 输出价)，单位：美元 / 百万 token
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-5": (15.0, 75.0),
}
"""计价表。

这是**估算**，用于预算闸门与 backfill 前的费用预估，不是账单。
真实费用以供应商为准；未知模型按最贵档估，宁可提前降级也不要超支。
"""

_FALLBACK_PRICING = (15.0, 75.0)
_PER_MILLION = 1_000_000


def estimate_cost(model_id: str, *, input_tokens: int, output_tokens: int) -> float:
    """估算一次调用的费用。

    Args:
        model_id: 模型 ID。
        input_tokens: 输入 token 数。
        output_tokens: 输出 token 数。

    Returns:
        美元金额。未知模型按最贵档估。
    """
    price_in, price_out = MODEL_PRICING.get(model_id, _FALLBACK_PRICING)
    return (input_tokens * price_in + output_tokens * price_out) / _PER_MILLION


@dataclass(frozen=True, slots=True)
class UsageRecord:
    """一次调用的用量。"""

    task_id: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    at: dt.datetime


@dataclass
class BudgetState:
    """预算状态。"""

    day: dt.date
    month: str
    daily_usd: float = 0.0
    monthly_usd: float = 0.0
    calls: int = 0
    degraded: bool = False
    degraded_reason: str = ""


class BudgetGuard:
    """预算闸门。"""

    def __init__(
        self,
        *,
        daily_usd: float,
        monthly_usd: float,
        usage_dir: Path | None = None,
    ) -> None:
        """初始化。

        Args:
            daily_usd: 每日上限。
            monthly_usd: 每月上限。
            usage_dir: 用量落盘目录。None 表示只在内存记账。
        """
        self._daily = daily_usd
        self._monthly = monthly_usd
        self._dir = Path(usage_dir) if usage_dir else None
        moment = now()
        self._state = BudgetState(day=moment.date(), month=moment.strftime("%Y-%m"))
        self._load()

    @property
    def state(self) -> BudgetState:
        """当前状态。"""
        self._roll_over()
        return self._state

    def can_spend(self, estimated_usd: float = 0.0) -> bool:
        """在预算内是否还能再花这么多。

        **预估费用也要算进去**：一次调用就把余额打穿的情况必须提前拦下，
        事后发现已经晚了。

        Args:
            estimated_usd: 本次调用的预估费用。

        Returns:
            可以则 True。
        """
        self._roll_over()
        if self._state.degraded:
            return False
        projected_day = self._state.daily_usd + estimated_usd
        if self._daily > 0 and projected_day > self._daily:
            self._degrade(f"当日费用将达 {projected_day:.4f} 美元，超出上限 {self._daily}")
            return False

        projected_month = self._state.monthly_usd + estimated_usd
        if self._monthly > 0 and projected_month > self._monthly:
            self._degrade(f"当月费用将达 {projected_month:.4f} 美元，超出上限 {self._monthly}")
            return False
        return True

    def record(self, usage: UsageRecord) -> None:
        """记一次用量。

        Args:
            usage: 用量记录。
        """
        self._roll_over()
        self._state.daily_usd += usage.cost_usd
        self._state.monthly_usd += usage.cost_usd
        self._state.calls += 1
        self._persist()

        if self._daily > 0 and self._state.daily_usd > self._daily:
            self._degrade(f"当日费用 {self._state.daily_usd:.4f} 美元已超上限 {self._daily}")
        elif self._monthly > 0 and self._state.monthly_usd > self._monthly:
            self._degrade(f"当月费用 {self._state.monthly_usd:.4f} 美元已超上限 {self._monthly}")

    def reset_degradation(self) -> None:
        """手工解除降级。跨日会自动解除，这里供用户提前恢复。"""
        self._state.degraded = False
        self._state.degraded_reason = ""

    def _degrade(self, reason: str) -> None:
        """降级到 off 并告警。

        Args:
            reason: 降级原因。
        """
        if self._state.degraded:
            return
        self._state.degraded = True
        self._state.degraded_reason = reason
        # 用 warning 而非 error：这是预期内的保护动作，不是故障
        _log.warning("llm_budget_exceeded", reason=reason)
        self._persist()

    def _roll_over(self) -> None:
        """跨日/跨月时重置计数并解除降级。"""
        moment = now()
        if moment.date() == self._state.day:
            return
        month = moment.strftime("%Y-%m")
        self._state = BudgetState(
            day=moment.date(),
            month=month,
            monthly_usd=self._state.monthly_usd if month == self._state.month else 0.0,
        )
        self._load()

    def _usage_path(self) -> Path | None:
        """当日用量文件路径。

        Returns:
            路径；未配置目录时 None。
        """
        if self._dir is None:
            return None
        return self._dir / f"{self._state.day.isoformat()}.json"

    def _persist(self) -> None:
        """落盘当日用量。"""
        path = self._usage_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "day": self._state.day.isoformat(),
                    "month": self._state.month,
                    "daily_usd": round(self._state.daily_usd, 6),
                    "monthly_usd": round(self._state.monthly_usd, 6),
                    "calls": self._state.calls,
                    "degraded": self._state.degraded,
                    "degraded_reason": self._state.degraded_reason,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load(self) -> None:
        """从磁盘恢复当日用量。

        进程重启不该让预算清零——否则反复重启就能绕过预算。
        """
        path = self._usage_path()
        if path is None or not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        self._state.daily_usd = float(payload.get("daily_usd", 0.0))
        self._state.monthly_usd = float(payload.get("monthly_usd", 0.0))
        self._state.calls = int(payload.get("calls", 0))
        self._state.degraded = bool(payload.get("degraded", False))
        self._state.degraded_reason = str(payload.get("degraded_reason", ""))
