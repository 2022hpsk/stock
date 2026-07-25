"""建议层的数据契约。

规范见 docs/03-功能规格.md F7.3、docs/07-信息情报模块.md 第七节。

**四支柱解释是硬性要求**：缺任一强制支柱的建议不进入 ``TradePlan``——
宁可不建议，也不给无法解释的建议。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

from quantstock.infra.types import IntentId, Money, PlanId, Side, Symbol, TradeDate
from quantstock.strategy.types import Evidence

__all__ = [
    "IntelEvidence",
    "PositionAnalytics",
    "RationaleBundle",
    "TradeIntent",
    "TradePlan",
    "Urgency",
]


class Urgency(StrEnum):
    """执行紧迫度。"""

    HIGH = "high"
    """风控触发的清仓/减仓，越快越好。"""
    NORMAL = "normal"
    LOW = "low"
    """可择机执行，不急于一时。"""


class IntelImpact(StrEnum):
    """情报对本条建议的作用方向。"""

    SUPPORT = "support"
    WEAKEN = "weaken"
    NEUTRAL = "neutral"
    """仅提示，不改变判断。"""


@dataclass(frozen=True, slots=True)
class IntelEvidence:
    """一条情报证据（支柱③）。

    ``url`` 与 ``published_at`` **必填**——不可复述无出处的内容（红线 I-R4）。
    """

    title: str
    source: str
    published_at: dt.datetime
    url: str
    domain: str
    sentiment: float
    importance: int
    impact: IntelImpact
    summary: str = ""

    def __post_init__(self) -> None:
        """校验出处完整。

        Raises:
            ValueError: 缺少原文链接。
        """
        if not self.url.strip():
            msg = "情报证据必须带原文链接（红线 I-R4）"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PositionAnalytics:
    """持仓与技术分析（支柱②）。

    对已持仓标的必须使用**真实成本与真实持仓历史**，不得用市价近似。
    """

    symbol: Symbol
    as_of: TradeDate
    market_price: Money
    # ---- 持仓（仅已持仓标的）----
    holding_days: int = 0
    cost_basis: Money | None = None
    unrealized_pnl_pct: Decimal | None = None
    weight_in_portfolio: Decimal | None = None
    holding_excess_vs_benchmark: Decimal | None = None
    days_to_tax_free: int | None = None
    """距满 1 年免红利税的天数。高股息标的可能值数千元。"""
    tax_saving_if_wait: Money | None = None
    # ---- 技术形态 ----
    ma5: float | None = None
    ma20: float | None = None
    ma60: float | None = None
    ma_alignment: str = ""
    pct_in_52w_range: float | None = None
    volume_vs_ma20: float | None = None
    atr20: float | None = None
    stop_loss_price: Money | None = None
    distance_to_stop_pct: Decimal | None = None

    @property
    def is_held(self) -> bool:
        """是否为已持仓标的。"""
        return self.cost_basis is not None

    def statements(self) -> list[str]:
        """渲染成人类可读的句子，供日报支柱②使用。

        Returns:
            句子列表；数据缺失的维度自动省略而非输出占位符。
        """
        lines: list[str] = []
        if self.is_held and self.cost_basis is not None:
            pnl = f"{self.unrealized_pnl_pct:+.2%}" if self.unrealized_pnl_pct else "—"
            lines.append(
                f"持有 {self.holding_days} 个交易日，成本 {self.cost_basis}，"
                f"现价 {self.market_price}（{pnl}）"
            )
            if self.holding_excess_vs_benchmark is not None:
                lines.append(f"持有期相对基准超额 {self.holding_excess_vs_benchmark:+.2%}")
            if self.weight_in_portfolio is not None:
                lines.append(f"当前占总资产 {self.weight_in_portfolio:.1%}")
        if self.ma20 is not None and self.ma60 is not None:
            lines.append(f"MA20={self.ma20:.2f}，MA60={self.ma60:.2f}，均线{self.ma_alignment}")
        if self.pct_in_52w_range is not None:
            lines.append(f"处于 52 周区间 {self.pct_in_52w_range:.0%} 分位")
        if self.volume_vs_ma20 is not None:
            change = self.volume_vs_ma20 - 1
            lines.append(f"量能相对 20 日均量{'放大' if change > 0 else '萎缩'} {abs(change):.0%}")
        if self.stop_loss_price is not None and self.distance_to_stop_pct is not None:
            lines.append(
                f"止损位 {self.stop_loss_price}（距现价 {self.distance_to_stop_pct:+.1%}）"
            )
        if self.days_to_tax_free is not None:
            saving = f"，预计节省 {self.tax_saving_if_wait}" if self.tax_saving_if_wait else ""
            lines.append(f"再持有 {self.days_to_tax_free} 天可免征红利税{saving}")
        return lines


@dataclass(frozen=True, slots=True)
class RationaleBundle:
    """建议解释的四支柱。

    支柱①②④ 为**强制**，③ 在无相关情报时必须明写"近 N 日无相关消息"
    而不是留空——留空让人分不清"没查"和"查了没有"。
    """

    verdict: str
    quant_evidence: tuple[Evidence, ...]
    technical: PositionAnalytics
    intel_evidence: tuple[IntelEvidence, ...]
    counter_evidence: tuple[Evidence, ...]
    falsification: tuple[str, ...]
    """证伪条件：什么情况下这个判断会被推翻。"""
    risk_notes: tuple[str, ...] = ()
    confidence: float = 0.5
    confidence_basis: str = ""
    intel_absent_note: str = ""
    """无相关情报时的说明。"""
    llm_involved: bool = False
    llm_adjustment: float = 0.0
    """LLM 对打分的调整量，用于界面 🤖 展开视图。"""

    def missing_pillars(self) -> list[str]:
        """检查缺失的强制支柱。

        Returns:
            缺失支柱的名称列表；完整时为空。
        """
        missing: list[str] = []
        if not self.quant_evidence:
            missing.append("①量化依据")
        if not self.technical.statements():
            missing.append("②持仓与技术分析")
        if not self.intel_evidence and not self.intel_absent_note:
            missing.append("③情报证据（无情报时也须注明）")
        if not self.counter_evidence and not self.falsification:
            missing.append("④反面证据与证伪条件")
        return missing

    @property
    def is_complete(self) -> bool:
        """四支柱是否齐全。"""
        return not self.missing_pillars()


@dataclass(frozen=True, slots=True)
class TradeIntent:
    """一条交易意图。

    **必须给出可执行的数量与价格区间**——"建议减仓"不可执行，
    "卖出 400 股，限价 1578~1596"才可以。
    """

    intent_id: IntentId
    symbol: Symbol
    side: Side
    qty: int
    price_low: Money
    price_high: Money
    urgency: Urgency
    rationale: RationaleBundle
    stop_loss: Money | None = None
    take_profit: Money | None = None

    @property
    def estimated_amount(self) -> Money:
        """按区间中值估算的成交金额。"""
        return (self.price_low + self.price_high) / 2 * self.qty

    def __post_init__(self) -> None:
        """校验数量与价格区间合法。

        Raises:
            ValueError: 数量非正或价格区间倒置。
        """
        if self.qty <= 0:
            msg = f"意图数量必须为正，收到 {self.qty}"
            raise ValueError(msg)
        if self.price_low > self.price_high:
            msg = f"价格区间倒置：{self.price_low} > {self.price_high}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class RejectedCandidate:
    """被否决的候选。

    日报必须展示这些——"为什么没买"和"为什么买了"同样重要。
    """

    symbol: Symbol
    reason: str
    rule_id: str = ""


@dataclass(frozen=True, slots=True)
class TradePlan:
    """交易计划。**唯一允许进入 execution 的输入**（红线 R5）。"""

    plan_id: PlanId
    account_id: str
    trade_date: TradeDate
    generated_at: dt.datetime
    intents: tuple[TradeIntent, ...]
    rejected: tuple[RejectedCandidate, ...] = ()
    incomplete: tuple[tuple[Symbol, str], ...] = ()
    """因解释不完整而被剔除的建议，附缺失的支柱。"""
    circuit_state: str = "normal"
    # ---- 可追溯性（红线 R6）----
    data_fingerprint: str = ""
    strategy_versions: dict[str, str] = field(default_factory=dict)
    param_hash: str = ""
    summary: str = ""
    # ---- 人工确认（红线 R5）----
    confirmed_by: str = ""
    confirmed_at: dt.datetime | None = None

    @property
    def is_confirmed(self) -> bool:
        """是否已人工确认。未确认的计划不得提交真实通道。"""
        return bool(self.confirmed_by) and self.confirmed_at is not None

    @property
    def total_buy_amount(self) -> Money:
        """买入金额合计。"""
        return sum(
            (i.estimated_amount for i in self.intents if i.side is Side.BUY),
            start=Decimal(0),
        )

    @property
    def total_sell_amount(self) -> Money:
        """卖出金额合计。"""
        return sum(
            (i.estimated_amount for i in self.intents if i.side is Side.SELL),
            start=Decimal(0),
        )
