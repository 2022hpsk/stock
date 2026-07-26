"""领域对象 → JSON 的转换。

**刻意用鸭子类型而不是 import 领域类**。分层契约禁止 ``web`` 直接依赖
``advisor`` / ``intel`` / ``execution`` 等业务层（docs/01 F20.1：界面是薄客户端）。
这里只按属性名读取 ``services`` 转出来的对象，不 import 它们的类型——
拦的是"界面绕过 services 直接调业务层"，读取转出对象的字段是允许的。

两条贯穿全文件的规则：

- **金额一律转成字符串**，不是 float。``Decimal("1596.52")`` 变成 float 会
  在 JSON 里变成 ``1596.5200000000001``，界面上就是一串脏数字；更糟的是
  前端再拿它去算总额，误差会累积（红线 R1）。
- **时间一律 ISO 8601 带时区**，前端负责显示成本地时间（红线 R3）。
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

__all__ = [
    "serialize_backtest",
    "serialize_intel_item",
    "serialize_plan",
    "serialize_preview",
    "serialize_rationale",
]


def _money(value: object) -> str | None:
    """金额转字符串。

    Args:
        value: 金额，可能为 None。

    Returns:
        字符串形式；None 时返回 None。
    """
    return None if value is None else str(value)


def _num(value: object) -> float | None:
    """比率类数值转 float。

    只用于**比率与统计量**（Sharpe、IC、涨跌幅），它们本来就是浮点语义，
    转 float 不损失有效信息。金额绝不走这里。

    Args:
        value: 数值。

    Returns:
        float；None 时返回 None。
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)  # type: ignore[arg-type]


def _when(value: object) -> str | None:
    """时间转 ISO 字符串。

    Args:
        value: 时间或日期。

    Returns:
        ISO 8601 字符串；None 时返回 None。
    """
    if value is None:
        return None
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    return str(value)


def _enum(value: object) -> str:
    """枚举转字符串。

    Args:
        value: 枚举或字符串。

    Returns:
        字符串值。
    """
    return str(getattr(value, "value", value))


def serialize_evidence(evidence: Any) -> dict[str, Any]:  # noqa: ANN401 - 鸭子类型
    """序列化一条量化证据。

    Args:
        evidence: 证据对象。

    Returns:
        JSON 字典。
    """
    return {
        "name": getattr(evidence, "name", ""),
        "value": _num(getattr(evidence, "value", None)),
        "detail": getattr(evidence, "detail", ""),
        "direction": _enum(getattr(evidence, "direction", "")),
    }


def serialize_intel_evidence(item: Any) -> dict[str, Any]:  # noqa: ANN401 - 鸭子类型
    """序列化一条情报证据。

    ``url`` 与 ``published_at`` 必须原样透传——界面上每条情报都要能点回原文
    （红线 I-R4）。少了它们这条证据在界面上就是不可核实的传闻。

    Args:
        item: 情报证据对象。

    Returns:
        JSON 字典。
    """
    return {
        "title": item.title,
        "source": item.source,
        "url": item.url,
        "published_at": _when(item.published_at),
        "domain": _enum(getattr(item, "domain", "")),
        "sentiment": _num(getattr(item, "sentiment", 0.0)),
        "importance": getattr(item, "importance", 0),
        "impact": _enum(getattr(item, "impact", "")),
        "summary": getattr(item, "summary", ""),
    }


def serialize_analytics(a: Any) -> dict[str, Any]:  # noqa: ANN401 - 鸭子类型
    """序列化持仓与技术分析（支柱②）。

    Args:
        a: 分析对象。

    Returns:
        JSON 字典，含预渲染好的 ``statements``——句子的措辞属于领域知识，
        放在后端才能与日报、CLI 保持一致。
    """
    return {
        "symbol": str(a.symbol),
        "as_of": _when(a.as_of),
        "market_price": _money(a.market_price),
        "is_held": a.is_held,
        "holding_days": a.holding_days,
        "cost_basis": _money(a.cost_basis),
        "unrealized_pnl_pct": _num(a.unrealized_pnl_pct),
        "weight_in_portfolio": _num(a.weight_in_portfolio),
        "days_to_tax_free": a.days_to_tax_free,
        "tax_saving_if_wait": _money(a.tax_saving_if_wait),
        "ma5": _num(a.ma5),
        "ma20": _num(a.ma20),
        "ma60": _num(a.ma60),
        "ma_alignment": a.ma_alignment,
        "pct_in_52w_range": _num(a.pct_in_52w_range),
        "volume_vs_ma20": _num(a.volume_vs_ma20),
        "atr20": _num(a.atr20),
        "stop_loss_price": _money(a.stop_loss_price),
        "distance_to_stop_pct": _num(a.distance_to_stop_pct),
        "statements": a.statements(),
    }


def serialize_rationale(r: Any) -> dict[str, Any]:  # noqa: ANN401 - 鸭子类型
    """序列化四支柱解释。

    四个支柱全部透传，缺一不可：界面的展开态就是靠它渲染的。
    特别是 ``counter_evidence`` 与 ``falsification``（支柱④）——
    只展示看多理由的界面会系统性地助长确认偏误。

    Args:
        r: 解释对象。

    Returns:
        JSON 字典。
    """
    return {
        "verdict": r.verdict,
        "quant_evidence": [serialize_evidence(e) for e in r.quant_evidence],
        "technical": serialize_analytics(r.technical),
        "intel_evidence": [serialize_intel_evidence(i) for i in r.intel_evidence],
        "intel_absent_note": r.intel_absent_note,
        "counter_evidence": [serialize_evidence(e) for e in r.counter_evidence],
        "falsification": list(r.falsification),
        "risk_notes": list(r.risk_notes),
        "confidence": _num(r.confidence),
        "confidence_basis": r.confidence_basis,
        "llm_involved": r.llm_involved,
        "llm_adjustment": _num(r.llm_adjustment),
        "is_complete": r.is_complete,
        "missing_pillars": r.missing_pillars(),
    }


def serialize_intent(intent: Any) -> dict[str, Any]:  # noqa: ANN401 - 鸭子类型
    """序列化一条交易意图。

    Args:
        intent: 意图对象。

    Returns:
        JSON 字典。
    """
    return {
        "intent_id": str(intent.intent_id),
        "symbol": str(intent.symbol),
        "side": _enum(intent.side),
        "qty": intent.qty,
        "price_low": _money(intent.price_low),
        "price_high": _money(intent.price_high),
        "estimated_amount": _money(intent.estimated_amount),
        "urgency": _enum(intent.urgency),
        "stop_loss": _money(intent.stop_loss),
        "take_profit": _money(intent.take_profit),
        "rationale": serialize_rationale(intent.rationale),
    }


def serialize_plan(plan: Any) -> dict[str, Any]:  # noqa: ANN401 - 鸭子类型
    """序列化交易计划。

    ``rejected`` 与 ``incomplete`` 与 ``intents`` 同等重要，一并透传：
    "为什么没买"和"为什么买了"在复盘时同样值钱，界面上不能只显示后者。

    Args:
        plan: 计划对象。

    Returns:
        JSON 字典。
    """
    return {
        "plan_id": str(plan.plan_id),
        "account_id": plan.account_id,
        "trade_date": _when(plan.trade_date),
        "generated_at": _when(plan.generated_at),
        "circuit_state": plan.circuit_state,
        "summary": plan.summary,
        "intents": [serialize_intent(i) for i in plan.intents],
        "rejected": [
            {"symbol": str(r.symbol), "reason": r.reason, "rule_id": r.rule_id}
            for r in plan.rejected
        ],
        "incomplete": [
            {"symbol": str(sym), "missing": missing} for sym, missing in plan.incomplete
        ],
        "total_buy_amount": _money(plan.total_buy_amount),
        "total_sell_amount": _money(plan.total_sell_amount),
        # 可追溯性三件套（红线 R6）：审计页要靠它复现当天的建议
        "data_fingerprint": plan.data_fingerprint,
        "strategy_versions": dict(plan.strategy_versions),
        "param_hash": plan.param_hash,
        "confirmed_by": plan.confirmed_by,
        "confirmed_at": _when(plan.confirmed_at),
        "is_confirmed": plan.is_confirmed,
    }


def serialize_preview(preview: Any) -> dict[str, Any]:  # noqa: ANN401 - 鸭子类型
    """序列化执行预检结果。

    Args:
        preview: 预检对象。

    Returns:
        JSON 字典。
    """
    return {
        "plan_id": preview.plan_id,
        "trade_date": preview.trade_date,
        "broker": preview.broker,
        # 真实通道必须显式 --live（红线 R5）。界面据此把提交按钮变成"需要确认码"
        "requires_live_flag": preview.requires_live_flag,
        "halted": preview.halted,
        "halt_reason": preview.halt_reason,
        "total_buy": _money(preview.total_buy),
        "total_sell": _money(preview.total_sell),
        "review_count": preview.review_count,
        "items": [serialize_intent_preview(item) for item in preview.items],
    }


def serialize_intent_preview(item: Any) -> dict[str, Any]:  # noqa: ANN401 - 鸭子类型
    """序列化单条意图的执行前视图。

    Args:
        item: 意图预检对象。

    Returns:
        JSON 字典。
    """
    drift = item.drift
    return {
        "intent_id": item.intent_id,
        "symbol": str(item.symbol),
        "side": _enum(item.side),
        "qty": item.qty,
        "price_low": _money(item.price_low),
        "price_high": _money(item.price_high),
        "limit_price": _money(item.limit_price),
        "estimated_amount": _money(item.estimated_amount),
        "urgency": item.urgency,
        "verdict": item.verdict,
        # needs_review 是"价格漂移过大，请人重新判断"，不是"已拒绝"。
        # 界面必须把两者分开显示，混在一起人会习惯性无视
        "needs_review": item.needs_review,
        "drift": None
        if drift is None
        else {
            "reference_price": _money(getattr(drift, "reference_price", None)),
            "current_price": _money(getattr(drift, "current_price", None)),
            "drift_pct": _num(getattr(drift, "drift_pct", None)),
            "exceeded": bool(getattr(drift, "exceeded", False)),
        },
    }


def serialize_intel_item(item: Any) -> dict[str, Any]:  # noqa: ANN401 - 鸭子类型
    """序列化一条情报。

    Args:
        item: 情报对象。

    Returns:
        JSON 字典。
    """
    return {
        "item_id": item.item_id,
        "source": item.source,
        "source_tier": _enum(item.source_tier),
        "domain": _enum(item.domain),
        "publish_at": _when(item.publish_at),
        "fetched_at": _when(item.fetched_at),
        "title": item.title,
        "body": item.body,
        "url": item.url,
        "symbols": [str(s) for s in item.symbols],
        "industries": list(item.industries),
        "themes": list(item.themes),
        "event_type": _enum(item.event_type) if item.event_type is not None else None,
        "importance": item.importance,
        "sentiment": _num(item.sentiment),
        "sentiment_source": item.sentiment_source,
        "llm_sentiment": _num(item.llm_sentiment),
        "classifier": item.classifier,
        "duplicates": list(item.duplicates),
    }


def serialize_backtest(report: Any) -> dict[str, Any]:  # noqa: ANN401 - 鸭子类型
    """序列化回测报告。

    ``warnings`` 一并返回并要求界面显著展示——一次好看的回测**不等于**
    策略好，这些提示就是为了让人别把单次结果当结论。

    Args:
        report: 回测报告。

    Returns:
        JSON 字典。
    """
    stats = report.stats
    return {
        "start": _when(report.start),
        "end": _when(report.end),
        "trading_days": report.trading_days,
        "initial_cash": report.initial_cash,
        "final_equity": report.final_equity,
        "fills": report.fills,
        "rejections": dict(report.rejections),
        "trial_id": report.trial_id,
        "universe": [str(s) for s in report.universe],
        "llm_mode": report.llm_mode,
        "explain": report.explain(),
        "warnings": report.warnings(),
        "stats": {
            "total_return": _num(stats.total_return),
            "annualized_return": _num(stats.annualized_return),
            "annualized_volatility": _num(stats.annualized_volatility),
            "sharpe": _num(stats.sharpe),
            "sortino": _num(stats.sortino),
            "calmar": _num(stats.calmar),
            "max_drawdown": _num(stats.max_drawdown),
            "max_drawdown_duration": stats.max_drawdown_duration,
            "win_rate": _num(stats.win_rate),
            "profit_loss_ratio": _num(stats.profit_loss_ratio),
            "trading_days": stats.trading_days,
            # TWR 衡量策略能力，MWR 衡量实际赚了多少。只看一个会得出相反结论：
            # 策略很好但在高点加了仓，TWR 漂亮而 MWR 是亏的
            "twr": _num(stats.twr),
            "mwr": _num(stats.mwr),
        },
    }
