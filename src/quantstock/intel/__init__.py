"""信息情报：多源资讯采集、去重、实体链接、事件分类、外置导入、分域摘要。

情报**不单独产生买入信号**（红线 I-R1）。三条受控通路：
①风险否决（硬约束，单向）②有界软因子 ③证据链解释。
"""

from quantstock.intel.blacklist import BlacklistEntry, IntelBlacklist
from quantstock.intel.classify import EventClassifier, SentimentScorer
from quantstock.intel.dedup import content_hash, dedup, simhash, similarity
from quantstock.intel.digest import DigestBuilder, build_portfolio_alerts
from quantstock.intel.entity import EntityLinker, SymbolDictionary
from quantstock.intel.external import ImportReport, InboxScanner, build_item
from quantstock.intel.pipeline import IntelPipeline, PipelineResult
from quantstock.intel.protocols import NewsSource, SourceRegistry, fetch_all
from quantstock.intel.scoring import ImportanceScorer, ImportanceWeights
from quantstock.intel.store import IntelStore
from quantstock.intel.types import (
    CalendarEvent,
    DomainDigest,
    EventType,
    IntelDigest,
    IntelDomain,
    IntelItem,
    PortfolioAlert,
    SourceHealth,
    SourceTier,
)

__all__ = [
    "BlacklistEntry",
    "CalendarEvent",
    "DigestBuilder",
    "DomainDigest",
    "EntityLinker",
    "EventClassifier",
    "EventType",
    "ImportReport",
    "ImportanceScorer",
    "ImportanceWeights",
    "InboxScanner",
    "IntelBlacklist",
    "IntelDigest",
    "IntelDomain",
    "IntelItem",
    "IntelPipeline",
    "IntelStore",
    "NewsSource",
    "PipelineResult",
    "PortfolioAlert",
    "SentimentScorer",
    "SourceHealth",
    "SourceRegistry",
    "SourceTier",
    "SymbolDictionary",
    "build_item",
    "build_portfolio_alerts",
    "content_hash",
    "dedup",
    "fetch_all",
    "simhash",
    "similarity",
]
