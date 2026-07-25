"""情报处理流水线。

规范见 docs/07-信息情报模块.md 第四节。

```
采集(多源) → 归一化 → 去重 → 实体链接 → 事件分类 → 重要性打分 → 情绪打分 → 落库
                                                                        ↓
                                                       分域摘要 ← LLM(可选)
```

**顺序不能随意调整**，几处是有依赖的：

- 去重必须在打分**之前**：importance 的"多源印证"项要数合并了几个源；
- 实体链接必须在打分之前：命中持仓的加分依赖链接结果；
- 分类必须在情绪之前：情绪先验来自事件类型。

流水线**不 import ``llm``**。LLM 兜底分类与摘要改写通过回调注入，
保证情报层对模型层零编译期依赖，关掉 LLM 时全链路完整可用（红线 LR2）。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from quantstock.infra.clock import now
from quantstock.infra.logging import get_logger
from quantstock.infra.types import Symbol
from quantstock.intel.blacklist import IntelBlacklist
from quantstock.intel.classify import EventClassifier, SentimentScorer
from quantstock.intel.dedup import DedupResult, dedup
from quantstock.intel.digest import DigestBuilder
from quantstock.intel.entity import EntityLinker
from quantstock.intel.scoring import ImportanceScorer
from quantstock.intel.types import (
    CalendarEvent,
    EventType,
    IntelDigest,
    IntelDomain,
    IntelItem,
    SourceHealth,
)

__all__ = ["IntelPipeline", "PipelineResult"]

_log = get_logger(__name__)

LlmClassifyFn = Callable[[IntelItem], tuple[EventType | None, str]]
"""LLM 兜底分类回调：返回 ``(事件类型, 分类器标识)``。"""

LlmSentimentFn = Callable[[IntelItem], float]
"""LLM 情绪打分回调。结果存入 ``llm_sentiment``，规则分保持不变。"""


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """一次流水线处理的结果。"""

    items: tuple[IntelItem, ...]
    dedup_result: DedupResult
    digest: IntelDigest | None = None
    blacklisted: tuple[Symbol, ...] = ()

    @property
    def summary(self) -> str:
        """人类可读摘要。"""
        base = f"处理 {len(self.items)} 条（合并重复 {self.dedup_result.dropped_count} 条）"
        return base + (f"，新增黑名单 {len(self.blacklisted)} 只" if self.blacklisted else "")


class IntelPipeline:
    """情报处理流水线。"""

    def __init__(
        self,
        *,
        linker: EntityLinker | None = None,
        classifier: EventClassifier | None = None,
        sentiment: SentimentScorer | None = None,
        importance: ImportanceScorer | None = None,
        digest_builder: DigestBuilder | None = None,
        blacklist: IntelBlacklist | None = None,
        llm_classify: LlmClassifyFn | None = None,
        llm_sentiment: LlmSentimentFn | None = None,
    ) -> None:
        """初始化。

        Args:
            linker: 实体链接器。
            classifier: 事件分类器。
            sentiment: 规则情绪打分器。
            importance: 重要性打分器。
            digest_builder: 摘要生成器。
            blacklist: 情报黑名单。
            llm_classify: LLM 兜底分类回调，仅在规则未命中时调用。
            llm_sentiment: LLM 情绪打分回调。
        """
        self._linker = linker or EntityLinker()
        self._classifier = classifier or EventClassifier()
        self._sentiment = sentiment or SentimentScorer()
        self._importance = importance or ImportanceScorer()
        self._digest = digest_builder or DigestBuilder()
        self._blacklist = blacklist or IntelBlacklist()
        self._llm_classify = llm_classify
        self._llm_sentiment = llm_sentiment

    @property
    def blacklist(self) -> IntelBlacklist:
        """情报黑名单。"""
        return self._blacklist

    def process(
        self,
        raw_items: Iterable[IntelItem],
        *,
        as_of: dt.datetime | None = None,
        holdings: Iterable[Symbol] = (),
        watchlist: Iterable[Symbol] = (),
        calendar: Sequence[CalendarEvent] = (),
        coverage: dict[str, SourceHealth] | None = None,
        session: str = "post",
        make_digest: bool = True,
    ) -> PipelineResult:
        """跑完整条流水线。

        Args:
            raw_items: 原始条目。
            as_of: 评估时点。**回测里必须显式传历史时点**，否则时效衰减与
                可见性过滤都会按"现在"算（红线 I-R5）。
            holdings: 当前持仓。
            watchlist: 候选池。
            calendar: 日历事件。
            coverage: 各源健康度。
            session: ``pre`` / ``post``。
            make_digest: 是否生成摘要。

        Returns:
            处理结果。

        Raises:
            ValueError: ``as_of`` 非 tz-aware。
        """
        moment = as_of or now()
        if moment.tzinfo is None:
            msg = "as_of 必须 tz-aware（红线 R3、I-R5）"
            raise ValueError(msg)

        # PIT 过滤放在最前：晚于决策时点的消息根本不该进入流水线
        visible = [i for i in raw_items if i.visible_at(moment)]

        # ① 去重必须在打分之前——importance 的"多源印证"项要数合并了几个源
        deduped = dedup(visible)

        enriched: list[IntelItem] = []
        for item in deduped.kept:
            linked = self._link(item)
            classified = self._classify(linked)
            scored = self._score(classified, as_of=moment)
            enriched.append(scored)

        ranked = tuple(sorted(enriched, key=lambda i: (-i.importance, i.publish_at)))

        # 打分完成后才能判黑名单：规则依赖 importance 与 event_type
        new_entries = self._blacklist.evaluate(ranked, as_of=moment)

        digest = None
        if make_digest:
            digest = self._digest.build(
                ranked,
                trade_date=moment.date(),
                session=session,
                holdings=holdings,
                watchlist=watchlist,
                calendar=calendar,
                coverage=coverage,
                generated_at=moment,
            )

        _log.info(
            "intel_pipeline_done",
            items=len(ranked),
            merged=deduped.dropped_count,
            blacklisted=len(new_entries),
        )
        return PipelineResult(
            items=ranked,
            dedup_result=deduped,
            digest=digest,
            blacklisted=tuple(dict.fromkeys(e.symbol for e in new_entries)),
        )

    # ------------------------------------------------------------------ 各段
    def _link(self, item: IntelItem) -> IntelItem:
        """实体链接。已带标的的条目不再覆盖——源方声明优先于文本推断。

        Args:
            item: 情报条目。

        Returns:
            带实体标签的条目。
        """
        result = self._linker.link(item.title, item.body)
        return replace(
            item,
            symbols=item.symbols or result.symbols,
            industries=item.industries or result.industries,
            themes=item.themes or result.themes,
            match_evidence=result.evidence,
        )

    def _classify(self, item: IntelItem) -> IntelItem:
        """事件与域分类。

        规则优先；规则未命中且配了 LLM 回调时才兜底，结果标注
        ``classifier="llm:<model>"``（红线 I-R3）。

        Args:
            item: 情报条目。

        Returns:
            带事件类型的条目。
        """
        if item.event_type is not None:
            return item

        # 只有源方**明确声明**的域才压过推断结果。兜底填的 COMPANY 不算声明——
        # 否则讲降准的条目会永远留在个股域
        declared = item.domain if item.domain_declared else None
        result = self._classifier.classify(item.title, item.body, default_domain=declared)
        if result.event_type is not None:
            return replace(
                item, event_type=result.event_type, domain=result.domain, classifier="rule"
            )

        if self._llm_classify is None:
            return replace(item, domain=result.domain, classifier="rule")

        event, tag = self._llm_classify(item)
        return replace(item, event_type=event, domain=result.domain, classifier=tag)

    def _score(self, item: IntelItem, *, as_of: dt.datetime) -> IntelItem:
        """情绪与重要性打分。

        情绪必须在重要性之前——虽然当前 importance 不依赖 sentiment，
        但黑名单的"累计负面"规则依赖它，顺序反了会漏判。

        Args:
            item: 情报条目。
            as_of: 评估时点。

        Returns:
            完成打分的条目。
        """
        rule_sentiment = self._sentiment.score(item.title, item.body, event_type=item.event_type)
        llm_score = self._llm_sentiment(item) if self._llm_sentiment is not None else None

        scored = replace(
            item,
            sentiment=rule_sentiment,
            sentiment_source="rule",
            llm_sentiment=llm_score,
        )
        breakdown = self._importance.score(scored, as_of=as_of)
        return replace(scored, importance=breakdown.total)

    def domains_seen(self, items: Sequence[IntelItem]) -> frozenset[IntelDomain]:
        """本批情报覆盖的域。

        Args:
            items: 情报条目。

        Returns:
            域集合。
        """
        return frozenset(i.domain for i in items)
