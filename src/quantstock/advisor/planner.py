"""交易计划编排。

规范见 docs/03-功能规格.md F7.1、F7.3。

把风控通过的调仓指令转成带完整四支柱解释的 ``TradeIntent``，
并做**解释完整性校验**——缺支柱的建议不进入计划。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from decimal import Decimal

from quantstock.advisor.types import (
    IntelEvidence,
    PositionAnalytics,
    RationaleBundle,
    RejectedCandidate,
    TradeIntent,
    TradePlan,
    Urgency,
)
from quantstock.infra.clock import now
from quantstock.infra.logging import get_logger
from quantstock.infra.money import quantize_price
from quantstock.infra.types import IntentId, PlanId, Side, Symbol, TradeDate
from quantstock.portfolio.builder import RebalanceOrder
from quantstock.risk.engine import RiskDecision
from quantstock.strategy.types import Evidence

__all__ = ["PlanBuilder", "compute_param_hash"]

_log = get_logger(__name__)

DEFAULT_PRICE_BAND = Decimal("0.006")
"""建议限价区间的单侧宽度。"""


def compute_param_hash(params: Mapping[str, object]) -> str:
    """计算参数哈希（红线 R6）。

    参数变了就是另一套策略——哈希进入计划，事后可精确判断"当时用的是哪套参数"。

    Args:
        params: 参数字典。

    Returns:
        16 位十六进制摘要。
    """
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class PlanBuilder:
    """交易计划构建器。"""

    def __init__(
        self,
        *,
        account_id: str,
        require_full_rationale: bool = True,
        price_band: Decimal = DEFAULT_PRICE_BAND,
    ) -> None:
        """初始化。

        Args:
            account_id: 账户标识。
            require_full_rationale: 是否要求四支柱齐全。
                **默认开启**——宁可不建议，也不给无法解释的建议。
            price_band: 限价区间单侧宽度。
        """
        self._account_id = account_id
        self._require_full = require_full_rationale
        self._price_band = price_band

    def build(
        self,
        *,
        trade_date: TradeDate,
        decision: RiskDecision,
        analytics: Mapping[Symbol, PositionAnalytics],
        quant_evidence: Mapping[Symbol, Sequence[Evidence]],
        counter_evidence: Mapping[Symbol, Sequence[Evidence]] | None = None,
        intel: Mapping[Symbol, Sequence[IntelEvidence]] | None = None,
        confidences: Mapping[Symbol, float] | None = None,
        data_fingerprint: str = "",
        strategy_versions: Mapping[str, str] | None = None,
        param_hash: str = "",
        intel_lookback_days: int = 7,
    ) -> TradePlan:
        """构建交易计划。

        Args:
            trade_date: 交易日。
            decision: 风控判定，只有 ``approved`` 的指令会进入计划。
            analytics: 各标的的持仓与技术分析（支柱②）。
            quant_evidence: 各标的的量化依据（支柱①）。
            counter_evidence: 各标的的反面证据（支柱④）。
            intel: 各标的的情报证据（支柱③）。
            confidences: 各标的的置信度。
            data_fingerprint: 数据快照指纹（红线 R6）。
            strategy_versions: 各策略版本。
            param_hash: 参数哈希。
            intel_lookback_days: 情报回看天数，用于"近 N 日无相关消息"的措辞。

        Returns:
            交易计划。解释不完整的建议被剔除并记入 ``incomplete``。
        """
        intents: list[TradeIntent] = []
        incomplete: list[tuple[Symbol, str]] = []

        for order in decision.approved:
            analytic = analytics.get(order.symbol)
            if analytic is None:
                incomplete.append((order.symbol, "缺少②持仓与技术分析"))
                continue

            rationale = self._build_rationale(
                order=order,
                analytic=analytic,
                quant=tuple(quant_evidence.get(order.symbol, ())),
                counter=tuple((counter_evidence or {}).get(order.symbol, ())),
                intel_items=tuple((intel or {}).get(order.symbol, ())),
                confidence=(confidences or {}).get(order.symbol, 0.5),
                intel_lookback_days=intel_lookback_days,
            )

            if self._require_full and not rationale.is_complete:
                missing = "、".join(rationale.missing_pillars())
                incomplete.append((order.symbol, f"解释不完整，缺 {missing}"))
                _log.warning(
                    "intent_dropped_incomplete_rationale",
                    symbol=order.symbol,
                    missing=missing,
                )
                continue

            intents.append(self._to_intent(order, rationale, analytic))

        plan = TradePlan(
            plan_id=PlanId(uuid.uuid4().hex[:12]),
            account_id=self._account_id,
            trade_date=trade_date,
            generated_at=now(),
            intents=tuple(intents),
            rejected=tuple(
                RejectedCandidate(symbol=o.symbol, reason=reason) for o, reason in decision.rejected
            ),
            incomplete=tuple(incomplete),
            circuit_state=decision.circuit_state.value,
            data_fingerprint=data_fingerprint,
            strategy_versions=dict(strategy_versions or {}),
            param_hash=param_hash,
            summary=_summarize(intents, decision),
        )
        _log.info(
            "plan_built",
            plan_id=plan.plan_id,
            intents=len(intents),
            rejected=len(plan.rejected),
            incomplete=len(incomplete),
        )
        return plan

    def _build_rationale(
        self,
        *,
        order: RebalanceOrder,
        analytic: PositionAnalytics,
        quant: tuple[Evidence, ...],
        counter: tuple[Evidence, ...],
        intel_items: tuple[IntelEvidence, ...],
        confidence: float,
        intel_lookback_days: int,
    ) -> RationaleBundle:
        """组装四支柱解释。

        Args:
            order: 调仓指令。
            analytic: 持仓与技术分析。
            quant: 量化依据。
            counter: 反面证据。
            intel_items: 情报证据。
            confidence: 置信度。
            intel_lookback_days: 情报回看天数。

        Returns:
            解释包。
        """
        action = "买入" if order.side is Side.BUY else "卖出"
        verdict = f"{action} {order.symbol} {order.qty} 股：{order.reason}"

        falsification = _falsification_conditions(order, analytic)
        risk_notes = _risk_notes(order, analytic)

        # 无相关情报时必须明写——留空让人分不清"没查"和"查了没有"
        absent = "" if intel_items else f"近 {intel_lookback_days} 日无该标的相关消息"

        return RationaleBundle(
            verdict=verdict,
            quant_evidence=quant,
            technical=analytic,
            intel_evidence=intel_items,
            counter_evidence=counter,
            falsification=falsification,
            risk_notes=risk_notes,
            confidence=confidence,
            confidence_basis=_confidence_basis(quant, counter, intel_items),
            intel_absent_note=absent,
        )

    def _to_intent(
        self,
        order: RebalanceOrder,
        rationale: RationaleBundle,
        analytic: PositionAnalytics,
    ) -> TradeIntent:
        """把调仓指令转成交易意图。

        Args:
            order: 调仓指令。
            rationale: 解释包。
            analytic: 持仓与技术分析。

        Returns:
            交易意图。
        """
        reference = order.reference_price
        low = quantize_price(reference * (1 - self._price_band))
        high = quantize_price(reference * (1 + self._price_band))

        return TradeIntent(
            intent_id=IntentId(uuid.uuid4().hex[:12]),
            symbol=order.symbol,
            side=order.side,
            qty=order.qty,
            price_low=low,
            price_high=high,
            urgency=_urgency_for(order, analytic),
            rationale=rationale,
            stop_loss=analytic.stop_loss_price if order.side is Side.BUY else None,
        )


def _urgency_for(order: RebalanceOrder, analytic: PositionAnalytics) -> Urgency:
    """判断执行紧迫度。

    Args:
        order: 调仓指令。
        analytic: 持仓与技术分析。

    Returns:
        紧迫度。清仓与逼近止损的减仓最急。
    """
    if order.side is Side.SELL and order.target_qty == 0:
        return Urgency.HIGH
    if (
        order.side is Side.SELL
        and analytic.distance_to_stop_pct is not None
        and analytic.distance_to_stop_pct > Decimal("-0.03")
    ):
        return Urgency.HIGH
    return Urgency.NORMAL


def _falsification_conditions(
    order: RebalanceOrder, analytic: PositionAnalytics
) -> tuple[str, ...]:
    """生成证伪条件。

    "什么情况下这个判断会被推翻"——没有证伪条件的判断不可证伪，
    也就无法从错误中学习。

    Args:
        order: 调仓指令。
        analytic: 持仓与技术分析。

    Returns:
        证伪条件列表。
    """
    conditions: list[str] = []
    if order.side is Side.SELL and analytic.ma20 is not None:
        conditions.append(
            f"若次日放量站回 MA20（{analytic.ma20:.2f}）之上，本次趋势转弱判断被证伪，"
            "应暂停后续减仓"
        )
    if order.side is Side.BUY and analytic.ma20 is not None:
        conditions.append(
            f"若买入后跌破 MA20（{analytic.ma20:.2f}）且量能放大，买入逻辑被证伪，应止损离场"
        )
    if analytic.stop_loss_price is not None:
        conditions.append(f"跌破 ATR 止损位 {analytic.stop_loss_price} 即视为判断错误")
    return tuple(conditions)


def _risk_notes(order: RebalanceOrder, analytic: PositionAnalytics) -> tuple[str, ...]:
    """生成风险提示（支柱⑤）。

    Args:
        order: 调仓指令。
        analytic: 持仓与技术分析。

    Returns:
        风险提示列表。
    """
    notes: list[str] = []
    if order.side is Side.SELL:
        notes.append("若次日跳空低开超过 2%，建议改为分两日执行，避免恐慌性抛售")
    else:
        notes.append("若次日跳空高开超过 2%，建议降低数量或改为择机买入")

    if analytic.days_to_tax_free is not None and order.side is Side.SELL:
        saving = f"，预计节省 {analytic.tax_saving_if_wait}" if analytic.tax_saving_if_wait else ""
        notes.append(
            f"该持仓再持有 {analytic.days_to_tax_free} 天即可免征红利税{saving}，请权衡是否值得等待"
        )
    return tuple(notes)


def _confidence_basis(
    quant: tuple[Evidence, ...],
    counter: tuple[Evidence, ...],
    intel_items: tuple[IntelEvidence, ...],
) -> str:
    """说明置信度的来源。

    Args:
        quant: 量化依据。
        counter: 反面证据。
        intel_items: 情报证据。

    Returns:
        一句话说明。
    """
    parts = [f"量化依据 {len(quant)} 条"]
    if counter:
        parts.append(f"反面证据 {len(counter)} 条，存在分歧故取偏保守值")
    if intel_items:
        parts.append(f"情报佐证 {len(intel_items)} 条")
    return "；".join(parts)


def _summarize(intents: Sequence[TradeIntent], decision: RiskDecision) -> str:
    """生成计划摘要。

    Args:
        intents: 交易意图。
        decision: 风控判定。

    Returns:
        一句话摘要。
    """
    buys = sum(1 for i in intents if i.side is Side.BUY)
    sells = len(intents) - buys
    text = f"{buys} 买 / {sells} 卖"
    if decision.rejected:
        text += f"，{len(decision.rejected)} 条被风控否决"
    if decision.circuit_state.value != "normal":
        text += f"（熔断状态：{decision.circuit_state.value}）"
    return text
