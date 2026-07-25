"""跨模块共享的基础类型与枚举。

规范见 docs/01-开发规范.md 第四、五条：领域术语必须统一，关键标量用 ``NewType`` 增强类型安全。
"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Final, NewType

__all__ = [
    "AccountId",
    "Adjust",
    "AssetType",
    "Board",
    "Direction",
    "Exchange",
    "Freq",
    "Horizon",
    "IntentId",
    "Money",
    "PlanId",
    "Settlement",
    "Side",
    "Symbol",
    "TradeDate",
    "make_symbol",
    "parse_symbol",
    "split_symbol",
]

# --------------------------------------------------------------------- 标量别名
Symbol = NewType("Symbol", str)
"""统一标的代码，格式 ``<6位代码>.<交易所>``，如 ``600519.SH``。"""

PlanId = NewType("PlanId", str)
IntentId = NewType("IntentId", str)
AccountId = NewType("AccountId", str)

TradeDate = date
"""交易日。语义别名——出现该类型即表示"交易日"而非自然日。"""

Money = Decimal
"""金额语义别名。红线 R1：金额一律 ``Decimal``，禁止 ``float``。"""


# --------------------------------------------------------------------- 枚举
class Exchange(StrEnum):
    """交易所。"""

    SH = "SH"
    SZ = "SZ"
    BJ = "BJ"


class Board(StrEnum):
    """板块。涨跌幅限制与交易规则按板块区分，见 docs/05-风控规范.md。"""

    MAIN = "main"
    """沪深主板"""
    GEM = "gem"
    """创业板"""
    STAR = "star"
    """科创板"""
    BSE = "bse"
    """北交所"""
    ETF = "etf"
    """场内基金"""


class AssetType(StrEnum):
    """资产类型。"""

    STOCK = "stock"
    ETF = "etf"
    LOF = "lof"
    INDEX = "index"


class Settlement(StrEnum):
    """交收制度。

    股票与股票型 ETF 为 T+1；跨境 / 债券 / 黄金 ETF 为 T+0。
    风控规则 A01 按本字段判断，不得一刀切（见 docs/05-风控规范.md A13）。
    """

    T0 = "T0"
    T1 = "T1"


class Adjust(StrEnum):
    """复权口径。红线 R4：任何价格序列必须显式携带本标记。"""

    NONE = "none"
    """不复权。用于下单价格、涨跌停判断与展示。"""
    QFQ = "qfq"
    """前复权。仅用于图表展示，由 hfq 实时换算，不落盘。"""
    HFQ = "hfq"
    """后复权。用于因子计算与回测收益，保证历史不随时间漂移。"""


class Freq(StrEnum):
    """K线频率。"""

    D = "D"
    W = "W"
    M = "M"
    M1 = "M1"
    M5 = "M5"
    M15 = "M15"
    M30 = "M30"
    M60 = "M60"

    @property
    def is_intraday(self) -> bool:
        """是否为日内频率。"""
        return self in _INTRADAY_FREQS


_INTRADAY_FREQS: Final[frozenset[Freq]] = frozenset(
    {Freq.M1, Freq.M5, Freq.M15, Freq.M30, Freq.M60}
)


class Side(StrEnum):
    """委托方向。"""

    BUY = "buy"
    SELL = "sell"


class Direction(StrEnum):
    """信号方向。

    注意 ``REDUCE`` 与 ``FLAT`` 的区别：前者是减仓，后者是清仓/不持有。
    """

    LONG = "long"
    REDUCE = "reduce"
    FLAT = "flat"


class Horizon(StrEnum):
    """策略周期层级。

    多周期融合：``最终权重 = 总仓位中枢(LONG) × 个股相对权重(MEDIUM) × 择时系数(SHORT)``。
    见 docs/02-系统架构.md 第五节。
    """

    LONG = "long"
    MEDIUM = "medium"
    SHORT = "short"


# --------------------------------------------------------------------- Symbol 工具
CODE_LENGTH: Final = 6
"""A 股标的代码固定 6 位。"""

_SYMBOL_RE: Final = re.compile(r"^(?P<code>\d{6})\.(?P<exchange>SH|SZ|BJ)$")

_SH_PREFIXES: Final = ("60", "68", "50", "51", "52", "56", "58", "88", "90")
_SZ_PREFIXES: Final = ("00", "30", "15", "16", "18", "20", "39")
_BJ_PREFIXES: Final = ("43", "83", "87", "92")


def make_symbol(code: str, exchange: Exchange) -> Symbol:
    """由 6 位代码与交易所构造标准 Symbol。

    Args:
        code: 6 位数字代码。
        exchange: 交易所。

    Returns:
        形如 ``600519.SH`` 的标准 Symbol。

    Raises:
        ValueError: code 不是 6 位数字。
    """
    if not (len(code) == CODE_LENGTH and code.isdigit()):
        msg = f"标的代码必须为 6 位数字，收到 {code!r}"
        raise ValueError(msg)
    return Symbol(f"{code}.{exchange.value}")


def split_symbol(symbol: Symbol | str) -> tuple[str, Exchange]:
    """拆解 Symbol 为 (代码, 交易所)。

    Args:
        symbol: 标准格式的 Symbol。

    Returns:
        ``(code, exchange)``。

    Raises:
        ValueError: 格式不合法。
    """
    match = _SYMBOL_RE.match(str(symbol))
    if match is None:
        msg = f"Symbol 格式非法，期望 <6位代码>.<SH|SZ|BJ>，收到 {symbol!r}"
        raise ValueError(msg)
    return match["code"], Exchange(match["exchange"])


def parse_symbol(raw: str) -> Symbol:
    """把各种外部格式的代码归一化为标准 Symbol。

    数据源适配器必须在 ``data`` 层调用本函数归一化，禁止让原始格式流出该层
    （见 docs/01-开发规范.md 第四条）。

    支持的输入形式::

        600519.SH  600519.sh  sh600519  SH600519  sh.600519  600519

    纯 6 位代码按前缀推断交易所。

    Args:
        raw: 外部数据源给出的原始代码。

    Returns:
        标准 Symbol。

    Raises:
        ValueError: 无法识别或无法推断交易所。
    """
    text = raw.strip().upper().replace("_", ".")

    # 形如 600519.SH
    if (match := _SYMBOL_RE.match(text)) is not None:
        return Symbol(f"{match['code']}.{match['exchange']}")

    # 形如 SH600519 / SH.600519
    for exchange in Exchange:
        prefix = exchange.value
        if text.startswith(prefix):
            rest = text[len(prefix) :].lstrip(".")
            if len(rest) == CODE_LENGTH and rest.isdigit():
                return make_symbol(rest, exchange)

    # 形如 600519.SS（部分源用 SS 表示上交所）
    if text.endswith(".SS"):
        code = text[:-3]
        if len(code) == CODE_LENGTH and code.isdigit():
            return make_symbol(code, Exchange.SH)

    # 纯 6 位数字，按前缀推断
    if len(text) == CODE_LENGTH and text.isdigit():
        return make_symbol(text, _infer_exchange(text))

    msg = f"无法识别的标的代码格式：{raw!r}"
    raise ValueError(msg)


def _infer_exchange(code: str) -> Exchange:
    """按代码前缀推断交易所。

    Args:
        code: 6 位数字代码。

    Returns:
        推断出的交易所。

    Raises:
        ValueError: 前缀不属于任何已知区段。
    """
    prefix = code[:2]
    if prefix in _SH_PREFIXES:
        return Exchange.SH
    if prefix in _SZ_PREFIXES:
        return Exchange.SZ
    if prefix in _BJ_PREFIXES:
        return Exchange.BJ
    msg = f"无法由代码 {code!r} 推断交易所，请显式指定"
    raise ValueError(msg)
