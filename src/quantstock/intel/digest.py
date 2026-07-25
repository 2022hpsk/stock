"""分域摘要生成。

规范见 docs/07-信息情报模块.md 4.7。

每日两次：08:30 盘前 / 18:30 盘后。

**要点全部由规则生成**（取 importance 最高的几条，渲染成带出处的引用行）。
LLM 摘要是可选增强：``pipeline`` 可注入一个回调来改写 ``highlights``，
改写后 ``DomainDigest.llm_generated`` 置 true，日报打 🤖 标（红线 I-R3）。
关掉 LLM 时摘要依然完整可用——这是 LR2 的要求在情报侧的体现。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Sequence

from quantstock.infra.clock import now
from quantstock.infra.types import Symbol
from quantstock.intel.scoring import HEADLINE_IMPORTANCE_THRESHOLD
from quantstock.intel.types import (
    CalendarEvent,
    DomainDigest,
    IntelDigest,
    IntelDomain,
    IntelItem,
    PortfolioAlert,
    SourceHealth,
)

__all__ = ["DigestBuilder", "build_portfolio_alerts"]

DEFAULT_HIGHLIGHTS = 5
DEFAULT_TOP_ITEMS = 10
DEFAULT_UPCOMING_DAYS = 7

CRITICAL_IMPORTANCE = 85
WARNING_IMPORTANCE = 70


def build_portfolio_alerts(
    items: Iterable[IntelItem], holdings: Iterable[Symbol]
) -> tuple[PortfolioAlert, ...]:
    """挑出命中当前持仓的事件。

    这是日报里优先级最高的一块：用户对自己持有的标的的消息，
    敏感度远高于对整个市场的。

    ``action_hint`` 只是提示文字，**不构成下单指令**（红线 I-R1）——
    真正的减仓建议要经过量化打分与风控，这里给的只是"该看一眼"。

    Args:
        items: 情报条目。
        holdings: 当前持仓。

    Returns:
        按严重度与重要性排序的告警。
    """
    held = frozenset(holdings)
    alerts: list[PortfolioAlert] = []
    for item in items:
        for symbol in item.symbols:
            if symbol not in held:
                continue
            severity, hint = _severity_of(item)
            alerts.append(
                PortfolioAlert(symbol=symbol, item=item, severity=severity, action_hint=hint)
            )
    order = {"critical": 0, "warning": 1, "info": 2}
    return tuple(sorted(alerts, key=lambda a: (order[a.severity], -a.item.importance, a.symbol)))


def _severity_of(item: IntelItem) -> tuple[str, str]:
    """判定告警级别。

    Args:
        item: 情报条目。

    Returns:
        ``(级别, 动作提示)``。
    """
    is_risk = item.event_type is not None and item.event_type.is_risk_event
    if is_risk and item.importance >= CRITICAL_IMPORTANCE:
        return "critical", "重大风险事件，建议复核该持仓并考虑降低敞口"
    if is_risk or item.importance >= CRITICAL_IMPORTANCE:
        return "warning", "需要关注，建议在今日复核时一并查看"
    if item.importance >= WARNING_IMPORTANCE:
        return "warning", ""
    return "info", ""


class DigestBuilder:
    """摘要生成器。"""

    def __init__(
        self,
        *,
        highlights_per_domain: int = DEFAULT_HIGHLIGHTS,
        top_items: int = DEFAULT_TOP_ITEMS,
        upcoming_days: int = DEFAULT_UPCOMING_DAYS,
    ) -> None:
        """初始化。

        Args:
            highlights_per_domain: 每域要点条数。
            top_items: 全局置顶条数。
            upcoming_days: 前瞻日历天数。
        """
        self._per_domain = highlights_per_domain
        self._top = top_items
        self._upcoming_days = upcoming_days

    def build(
        self,
        items: Sequence[IntelItem],
        *,
        trade_date: dt.date,
        session: str = "post",
        holdings: Iterable[Symbol] = (),
        watchlist: Iterable[Symbol] = (),
        calendar: Sequence[CalendarEvent] = (),
        coverage: dict[str, SourceHealth] | None = None,
        generated_at: dt.datetime | None = None,
    ) -> IntelDigest:
        """生成一次摘要。

        Args:
            items: 已完成打分的情报。
            trade_date: 交易日。
            session: ``pre`` 盘前 / ``post`` 盘后。
            holdings: 当前持仓。
            watchlist: 候选池。
            calendar: 日历事件。
            coverage: 各源健康度。
            generated_at: 生成时刻。

        Returns:
            分域摘要。
        """
        watched = frozenset(watchlist)
        by_domain: dict[IntelDomain, DomainDigest] = {}

        for domain in IntelDomain:
            bucket = [i for i in items if i.domain is domain]
            if not bucket:
                continue
            ranked = sorted(bucket, key=lambda i: (-i.importance, i.publish_at))
            by_domain[domain] = DomainDigest(
                domain=domain,
                highlights=tuple(i.cite() for i in ranked[: self._per_domain]),
                items=tuple(ranked),
                net_sentiment=_weighted_sentiment(ranked),
                symbols=tuple(dict.fromkeys(s for i in ranked for s in i.symbols)),
            )

        headline = [i for i in items if i.importance >= HEADLINE_IMPORTANCE_THRESHOLD]
        top = sorted(headline, key=lambda i: (-i.importance, i.publish_at))[: self._top]

        horizon = trade_date + dt.timedelta(days=self._upcoming_days)
        upcoming = tuple(
            sorted(
                (e for e in calendar if trade_date <= e.event_date <= horizon),
                key=lambda e: (e.event_date, -e.importance),
            )
        )

        return IntelDigest(
            trade_date=trade_date,
            generated_at=generated_at or now(),
            session=session,
            by_domain=by_domain,
            top_items=tuple(top),
            portfolio_alerts=build_portfolio_alerts(items, holdings),
            watchlist_hits=tuple(
                sorted(
                    (i for i in items if any(s in watched for s in i.symbols)),
                    key=lambda i: -i.importance,
                )
            ),
            upcoming=upcoming,
            coverage=dict(coverage or {}),
        )

    @staticmethod
    def render(digest: IntelDigest) -> list[str]:
        """把摘要渲染成日报文本行。

        Args:
            digest: 摘要。

        Returns:
            文本行列表。
        """
        label = "盘前" if digest.session == "pre" else "盘后"
        lines = [f"# 情报摘要 {digest.trade_date} {label}（共 {digest.total_items} 条）"]

        if digest.portfolio_alerts:
            lines.append("\n## ⚠ 持仓相关（优先级最高）")
            for alert in digest.portfolio_alerts:
                mark = {"critical": "🔴", "warning": "🟡"}.get(alert.severity, "·")
                lines.append(f"{mark} {alert.symbol}  {alert.item.cite()}")
                if alert.action_hint:
                    lines.append(f"    {alert.action_hint}")

        if digest.top_items:
            lines.append("\n## 重大消息")
            lines.extend(f"· [{i.importance}] {i.cite()}" for i in digest.top_items)

        for domain, block in digest.by_domain.items():
            flag = " 🤖" if block.llm_generated else ""
            lines.append(
                f"\n## {domain.value}（{block.count} 条，净情绪 {block.net_sentiment:+.2f}）{flag}"
            )
            lines.extend(f"· {h}" for h in block.highlights)

        if digest.upcoming:
            lines.append("\n## 未来一周关注")
            lines.extend(f"· {e.event_date} {e.title}" for e in digest.upcoming)

        # 情报健康永远要写，哪怕全绿——用户需要能分辨"没消息"和"没查成"
        lines.append("\n## 情报健康")
        if failed := digest.failed_sources:
            lines.append(f"⚠ 采集失败：{'、'.join(failed)}")
        if missing := digest.missing_domains:
            lines.append(f"⚠ 今日无情报的域：{'、'.join(d.value for d in missing)}")
        if not failed and not missing:
            lines.append("全部源正常，各域均有情报。")
        return lines


def _weighted_sentiment(items: Sequence[IntelItem]) -> float:
    """按 importance 加权的净情绪。

    加权而非简单平均：一条 90 分的立案调查和十条 10 分的日常快讯，
    简单平均会把前者稀释到看不见。

    Args:
        items: 情报条目。

    Returns:
        -1 ~ 1 的净情绪；无权重时为 0。
    """
    weight = sum(max(i.importance, 1) for i in items)
    if not weight:
        return 0.0
    return sum(i.sentiment * max(i.importance, 1) for i in items) / weight
