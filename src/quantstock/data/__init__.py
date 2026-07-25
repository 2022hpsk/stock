"""行情与基本面数据：多源适配、归一化、复权、质量校验、PIT 访问。"""

from quantstock.data.adjust import assert_adjust, convert, hfq_to_none, none_to_hfq
from quantstock.data.lake import ParquetLake
from quantstock.data.protocols import FallbackChain, MarketDataSource
from quantstock.data.quality import QualityChecker, QualityReport, Severity
from quantstock.data.types import Bar, Instrument, InstrumentStatus, UniverseMember
from quantstock.data.universe import UniverseRegistry, check_survivorship_bias

__all__ = [
    "Bar",
    "FallbackChain",
    "Instrument",
    "InstrumentStatus",
    "MarketDataSource",
    "ParquetLake",
    "QualityChecker",
    "QualityReport",
    "Severity",
    "UniverseMember",
    "UniverseRegistry",
    "assert_adjust",
    "check_survivorship_bias",
    "convert",
    "hfq_to_none",
    "none_to_hfq",
]
