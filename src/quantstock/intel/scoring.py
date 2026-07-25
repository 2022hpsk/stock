"""重要性打分。

规范见 docs/07-信息情报模块.md 4.5。

```
importance = w1·来源层级 + w2·事件基础分 + w3·多源印证 + w4·命中持仓/候选池
           + w5·时效衰减 + w6·交易所强制披露
```

**全程规则化、不使用 LLM**。原因不是不信任模型，而是 importance 直接决定
"这条消息要不要触发黑名单"（≥85 且为风险类）。一个会随模型版本漂移的分数
不该拥有这种权力——同一条 2019 年的立案调查公告，今天和明年重跑必须得到同一个分。

时效衰减用**发布时间到评估时点**的小时数，而不是到"现在"：
回测里评估时点是历史某天，用 `now()` 会让所有历史情报都衰减到 0。
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable
from dataclasses import dataclass

from quantstock.infra.types import Symbol
from quantstock.intel.types import EventType, IntelItem, SourceTier

__all__ = [
    "BLACKLIST_IMPORTANCE_THRESHOLD",
    "HEADLINE_IMPORTANCE_THRESHOLD",
    "ImportanceScorer",
    "ImportanceWeights",
    "ScoreBreakdown",
]

HEADLINE_IMPORTANCE_THRESHOLD = 70
"""进入日报"重大消息"置顶区的门槛。"""

BLACKLIST_IMPORTANCE_THRESHOLD = 80
"""风险类事件触发黑名单的门槛（docs/07 第六节 通路 1）。"""

USER_IMPORTANCE_CAP = 90
"""人工导入条目的 importance 上限。

防止单条人工输入压过全部量化信号——用户手输一条 100 分的"利好"就能左右
整个组合，是这个系统最容易被自己绕过风控的地方。
"""

_EVENT_BASE: dict[EventType, int] = {
    EventType.DELISTING_RISK: 40,
    EventType.REGULATORY_PROBE: 38,
    EventType.AUDIT_QUALIFIED: 36,
    EventType.GUARANTEE_RISK: 34,
    EventType.CONTROL_CHANGE: 30,
    EventType.MA: 28,
    EventType.EARNINGS_SURPRISE: 28,
    EventType.PLEDGE_RISK: 26,
    EventType.EARNINGS_FORECAST: 25,
    EventType.SHAREHOLDER_REDUCE: 24,
    EventType.MAJOR_CONTRACT: 24,
    EventType.MONETARY: 24,
    EventType.ST_CHANGE: 24,
    EventType.EARNINGS_REPORT: 22,
    EventType.BUYBACK: 20,
    EventType.SHAREHOLDER_INCREASE: 20,
    EventType.POLICY_NEGATIVE: 20,
    EventType.POLICY_POSITIVE: 20,
    EventType.PLACEMENT: 18,
    EventType.LITIGATION: 18,
    EventType.UNLOCK: 18,
    EventType.MACRO_DATA: 18,
    EventType.SUSPENSION: 16,
    EventType.GEOPOLITICS: 16,
    EventType.DIVIDEND: 15,
    EventType.FISCAL: 15,
    EventType.CAPACITY: 14,
    EventType.PRICE_MOVE: 14,
    EventType.INDEX_ADJUST: 12,
    EventType.ASSET_SALE: 12,
    EventType.SPLIT: 10,
}


@dataclass(frozen=True, slots=True)
class ImportanceWeights:
    """打分权重。可在 ``config/intel.yaml`` 覆盖。"""

    source_tier: float = 1.0
    event_base: float = 1.0
    corroboration: float = 4.0
    """每多一个独立源印证加的分。"""
    max_corroboration: int = 5
    portfolio_hit: int = 18
    """命中当前持仓的加分。持仓标的的消息对用户的实际价值远高于其它。"""
    watchlist_hit: int = 9
    official_disclosure: int = 10
    """交易所强制披露事项的加分。"""
    decay_half_life_hours: float = 48.0
    """时效半衰期。48 小时后时效项减半。"""
    decay_weight: float = 12.0


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """打分明细。

    每一项都留下来是为了日报能回答"这条为什么是 87 分"——
    一个说不清来历的分数，用户迟早会不再相信它。
    """

    total: int
    source_tier: float
    event_base: float
    corroboration: float
    portfolio: float
    official: float
    decay: float

    def explain(self) -> str:
        """渲染成一行说明。

        Returns:
            人类可读的分项说明。
        """
        parts = [
            f"来源{self.source_tier:+.0f}",
            f"事件{self.event_base:+.0f}",
            f"印证{self.corroboration:+.0f}",
            f"持仓{self.portfolio:+.0f}",
            f"官方{self.official:+.0f}",
            f"时效{self.decay:+.0f}",
        ]
        return f"importance {self.total} = " + " ".join(p for p in parts if not p.endswith("+0"))


class ImportanceScorer:
    """重要性打分器。"""

    def __init__(
        self,
        *,
        weights: ImportanceWeights | None = None,
        holdings: Iterable[Symbol] = (),
        watchlist: Iterable[Symbol] = (),
    ) -> None:
        """初始化。

        Args:
            weights: 权重配置。
            holdings: 当前持仓。
            watchlist: 候选池。
        """
        self._w = weights or ImportanceWeights()
        self._holdings = frozenset(holdings)
        self._watchlist = frozenset(watchlist)

    def score(self, item: IntelItem, *, as_of: dt.datetime) -> ScoreBreakdown:
        """对一条情报打分。

        Args:
            item: 情报条目。
            as_of: 评估时点。回测里是历史时点，**不要传 ``now()``**。

        Returns:
            打分明细。

        Raises:
            ValueError: ``as_of`` 非 tz-aware。
        """
        if as_of.tzinfo is None:
            msg = "as_of 必须 tz-aware，否则时效衰减会因时区而错（红线 R3）"
            raise ValueError(msg)

        w = self._w
        tier_score = int(item.source_tier) * 6 * w.source_tier
        event_score = (
            _EVENT_BASE.get(item.event_type, 10) * w.event_base if item.event_type else 6.0
        )

        sources = min(len(item.duplicates) + 1, w.max_corroboration)
        corroboration = (sources - 1) * w.corroboration

        portfolio = 0.0
        if any(s in self._holdings for s in item.symbols):
            portfolio = float(w.portfolio_hit)
        elif any(s in self._watchlist for s in item.symbols):
            portfolio = float(w.watchlist_hit)

        official = float(w.official_disclosure) if item.source_tier is SourceTier.OFFICIAL else 0.0

        age_hours = max((as_of - item.publish_at).total_seconds() / 3600, 0.0)
        freshness = math.exp(-age_hours * math.log(2) / w.decay_half_life_hours)
        decay = w.decay_weight * freshness

        raw = tier_score + event_score + corroboration + portfolio + official + decay
        total = round(max(0.0, min(100.0, raw)))
        if item.source_tier is SourceTier.USER:
            total = min(total, USER_IMPORTANCE_CAP)

        return ScoreBreakdown(
            total=total,
            source_tier=tier_score,
            event_base=event_score,
            corroboration=corroboration,
            portfolio=portfolio,
            official=official,
            decay=decay,
        )
