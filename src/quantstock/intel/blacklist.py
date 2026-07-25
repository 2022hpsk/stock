"""情报黑名单：情报进入决策的第一条通路（硬约束）。

规范见 docs/07-信息情报模块.md 第六节 通路 1。

**这是情报唯一能产生硬性影响的地方，而且方向是单向的**（红线 I-R1/I-R2）：
只能禁止买入并建议减仓，绝不能触发买入。理由是不对称的——
错过一次利好的代价是少赚，踩中一次立案调查的代价是本金永久损失。

每条黑名单必须记录触发的 ``item_id`` 与原文链接，日报里可点击溯源。
说不出理由的禁买规则，用户迟早会手工绕过它。
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from quantstock.infra.logging import get_logger
from quantstock.infra.serde import from_jsonable, to_jsonable
from quantstock.infra.types import Symbol
from quantstock.intel.scoring import BLACKLIST_IMPORTANCE_THRESHOLD
from quantstock.intel.types import IntelItem, SourceTier

__all__ = [
    "BlacklistEntry",
    "BlacklistState",
    "IntelBlacklist",
]

_log = get_logger(__name__)

DEFAULT_TTL_DAYS = 60
DEFAULT_NEGATIVE_STREAK = 3
"""近 N 日累计负面事件数达到该值即拉黑。"""
DEFAULT_STREAK_WINDOW_DAYS = 60
NEGATIVE_SENTIMENT_THRESHOLD = -0.3
"""计入"负面事件"的情绪分门槛。"""


@dataclass(frozen=True, slots=True)
class BlacklistEntry:
    """一条黑名单记录。"""

    symbol: Symbol
    reason: str
    triggered_at: dt.datetime
    expires_at: dt.datetime
    item_ids: tuple[str, ...]
    """触发本条的情报 id，供溯源。"""
    urls: tuple[str, ...] = ()
    """原文链接（红线 I-R4）。"""
    rule: str = ""
    """触发的规则名：``risk_event`` / ``negative_streak``。"""

    def active_at(self, moment: dt.datetime) -> bool:
        """在给定时点是否仍然有效。

        Args:
            moment: 时点，必须 tz-aware。

        Returns:
            有效则 True。
        """
        return self.triggered_at <= moment < self.expires_at

    def explain(self) -> str:
        """人类可读说明。

        Returns:
            带溯源链接的说明。
        """
        link = f"  🔗 {self.urls[0]}" if self.urls else ""
        return (
            f"{self.symbol} 禁止买入至 {self.expires_at.date()}："
            f"{self.reason}（规则 {self.rule}）{link}"
        )


@dataclass(frozen=True, slots=True)
class BlacklistState:
    """黑名单全量状态。落盘用。"""

    entries: tuple[BlacklistEntry, ...] = ()
    updated_at: dt.datetime | None = None


@dataclass
class IntelBlacklist:
    """情报黑名单。

    可从情报流水重建——与账本同一思路：状态由事件推导，而不是就地修改。
    """

    ttl_days: int = DEFAULT_TTL_DAYS
    negative_streak: int = DEFAULT_NEGATIVE_STREAK
    streak_window_days: int = DEFAULT_STREAK_WINDOW_DAYS
    importance_threshold: int = BLACKLIST_IMPORTANCE_THRESHOLD
    _entries: dict[Symbol, BlacklistEntry] = field(default_factory=dict, init=False)

    def evaluate(self, items: Iterable[IntelItem], *, as_of: dt.datetime) -> list[BlacklistEntry]:
        """按情报流重新计算黑名单。

        Args:
            items: 情报条目（应已完成打分）。
            as_of: 评估时点，必须 tz-aware。

        Returns:
            本次新增或刷新的记录。

        Raises:
            ValueError: ``as_of`` 非 tz-aware。
        """
        if as_of.tzinfo is None:
            msg = "as_of 必须 tz-aware（红线 R3）"
            raise ValueError(msg)

        visible = [i for i in items if i.visible_at(as_of)]
        produced: list[BlacklistEntry] = []
        produced.extend(self._from_risk_events(visible, as_of))
        produced.extend(self._from_negative_streak(visible, as_of))

        for entry in produced:
            existing = self._entries.get(entry.symbol)
            # 同一标的再次触发时延长有效期而非新建：风险还在，计时就该重来
            if existing is None or entry.expires_at > existing.expires_at:
                self._entries[entry.symbol] = entry
                _log.warning(
                    "intel_blacklist_added",
                    symbol=str(entry.symbol),
                    rule=entry.rule,
                    reason=entry.reason,
                    expires_at=entry.expires_at.isoformat(),
                )
        return produced

    def _from_risk_events(
        self, items: Sequence[IntelItem], as_of: dt.datetime
    ) -> list[BlacklistEntry]:
        """规则一：单条重大风险事件。

        要求 **官方或媒体源** —— 社交媒体传言不足以触发禁买，
        否则一条论坛帖子就能把持仓标的锁死。

        Args:
            items: 可见的情报。
            as_of: 评估时点。

        Returns:
            新增记录。
        """
        out: list[BlacklistEntry] = []
        for item in items:
            if item.event_type is None or not item.event_type.is_risk_event:
                continue
            if item.source_tier < SourceTier.MEDIA:
                continue
            if item.importance < self.importance_threshold:
                continue
            for symbol in item.symbols:
                out.append(
                    BlacklistEntry(
                        symbol=symbol,
                        reason=f"{item.event_type.value}：{item.title}",
                        triggered_at=as_of,
                        expires_at=as_of + dt.timedelta(days=self.ttl_days),
                        item_ids=(item.item_id,),
                        urls=(item.url,) if item.url else (),
                        rule="risk_event",
                    )
                )
        return out

    def _from_negative_streak(
        self, items: Sequence[IntelItem], as_of: dt.datetime
    ) -> list[BlacklistEntry]:
        """规则二：窗口内累计负面事件。

        单条不足以定性，但连续负面通常意味着基本面在恶化。

        Args:
            items: 可见的情报。
            as_of: 评估时点。

        Returns:
            新增记录。
        """
        window_start = as_of - dt.timedelta(days=self.streak_window_days)
        buckets: dict[Symbol, list[IntelItem]] = {}
        for item in items:
            if item.publish_at < window_start or item.sentiment > NEGATIVE_SENTIMENT_THRESHOLD:
                continue
            for symbol in item.symbols:
                buckets.setdefault(symbol, []).append(item)

        out: list[BlacklistEntry] = []
        for symbol, hits in buckets.items():
            if len(hits) < self.negative_streak:
                continue
            ordered = sorted(hits, key=lambda i: i.publish_at, reverse=True)
            out.append(
                BlacklistEntry(
                    symbol=symbol,
                    reason=f"近 {self.streak_window_days} 日累计 {len(hits)} 条负面事件",
                    triggered_at=as_of,
                    expires_at=as_of + dt.timedelta(days=self.ttl_days),
                    item_ids=tuple(i.item_id for i in ordered),
                    urls=tuple(i.url for i in ordered if i.url),
                    rule="negative_streak",
                )
            )
        return out

    def is_blocked(self, symbol: Symbol, *, as_of: dt.datetime) -> bool:
        """该标的当前是否禁止买入。

        Args:
            symbol: 标的。
            as_of: 时点。

        Returns:
            禁止则 True。
        """
        entry = self._entries.get(symbol)
        return entry is not None and entry.active_at(as_of)

    def entry_for(self, symbol: Symbol) -> BlacklistEntry | None:
        """取某标的的黑名单记录，用于解释。

        Args:
            symbol: 标的。

        Returns:
            记录；无则 None。
        """
        return self._entries.get(symbol)

    def active_entries(self, *, as_of: dt.datetime) -> tuple[BlacklistEntry, ...]:
        """当前生效的全部记录。

        Args:
            as_of: 时点。

        Returns:
            按标的排序的记录。
        """
        return tuple(
            sorted(
                (e for e in self._entries.values() if e.active_at(as_of)),
                key=lambda e: e.symbol,
            )
        )

    def state(self, *, as_of: dt.datetime | None = None) -> BlacklistState:
        """导出状态。

        Args:
            as_of: 更新时点。

        Returns:
            可落盘的状态。
        """
        return BlacklistState(entries=tuple(self._entries.values()), updated_at=as_of)

    def save(self, path: Path, *, as_of: dt.datetime | None = None) -> Path:
        """落盘。

        Args:
            path: 文件路径。
            as_of: 更新时点。

        Returns:
            写入的路径。
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = to_jsonable(self.state(as_of=as_of))
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, path: Path) -> None:
        """从文件恢复。

        Args:
            path: 文件路径。文件不存在时静默跳过——首次运行没有黑名单是正常的。
        """
        if not path.exists():
            return
        state = from_jsonable(BlacklistState, json.loads(path.read_text(encoding="utf-8")))
        self._entries = {e.symbol: e for e in state.entries}
