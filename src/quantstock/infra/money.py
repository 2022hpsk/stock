"""金额与数量计算工具。

红线 R1：金额、价格、成本、盈亏一律使用 ``Decimal``，禁止 ``float``。
仅因子/统计计算的中间过程允许浮点，结果落到订单或账本时必须经本模块转回并量化。
"""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Final

from quantstock.infra.types import Money

__all__ = [
    "CNY",
    "PRICE",
    "RATIO",
    "ZERO",
    "align_lot",
    "money",
    "quantize_cny",
    "quantize_price",
    "quantize_ratio",
    "to_money",
]

CNY: Final[Decimal] = Decimal("0.01")
"""人民币金额精度：分。"""

PRICE: Final[Decimal] = Decimal("0.0001")
"""价格精度。A股报价最小变动为 0.01 元，多留两位供均价与复权计算。"""

RATIO: Final[Decimal] = Decimal("0.000001")
"""比例/权重精度。"""

ZERO: Final[Money] = Decimal("0")


def money(value: str | int | Decimal) -> Money:
    """构造金额。

    刻意**不接受** ``float``——浮点字面量已经损失精度，转换只会把误差固化下来。
    需要从浮点转换时请显式调用 :func:`to_money` 并承担精度责任。

    Args:
        value: 字符串、整数或已有的 Decimal。

    Returns:
        Decimal 金额。

    Raises:
        TypeError: 传入了 float。
        ValueError: 字符串无法解析为数值。
    """
    if isinstance(value, float):
        msg = (
            "禁止用 float 构造金额（红线 R1）。请传字符串如 money('123.45')；"
            "确需从浮点转换时用 to_money()。"
        )
        raise TypeError(msg)
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        msg = f"无法解析为金额：{value!r}"
        raise ValueError(msg) from exc


def to_money(value: float, *, precision: Decimal = CNY) -> Money:
    """把浮点数转为金额，用于因子/统计计算结果落账。

    调用点应当能说清楚"为什么这里是浮点"——通常是统计计算的输出。

    Args:
        value: 浮点数值。
        precision: 量化精度，默认到分。

    Returns:
        量化后的 Decimal 金额。
    """
    return Decimal(str(value)).quantize(precision, rounding=ROUND_HALF_UP)


def quantize_cny(value: Money) -> Money:
    """金额量化到分，四舍五入。

    Args:
        value: 待量化金额。

    Returns:
        精确到分的金额。
    """
    return value.quantize(CNY, rounding=ROUND_HALF_UP)


def quantize_price(value: Money) -> Money:
    """价格量化。

    Args:
        value: 待量化价格。

    Returns:
        量化后的价格。
    """
    return value.quantize(PRICE, rounding=ROUND_HALF_UP)


def quantize_ratio(value: Money) -> Money:
    """比例量化。

    Args:
        value: 待量化比例。

    Returns:
        量化后的比例。
    """
    return value.quantize(RATIO, rounding=ROUND_HALF_UP)


def align_lot(qty: int, *, lot_size: int = 100, min_qty: int = 0) -> int:
    """把买入数量向下对齐到整手（风控规则 A03）。

    向下取整而非四舍五入：宁可少买，不可超出可用资金。

    Args:
        qty: 期望数量（股）。
        lot_size: 每手股数，主板/ETF 为 100，科创板递增单位为 1。
        min_qty: 最小申报数量，科创板为 200。低于此值返回 0（放弃该笔）。

    Returns:
        对齐后的数量；不足一手或低于最小申报量时返回 0。

    Raises:
        ValueError: lot_size 非正。
    """
    if lot_size <= 0:
        msg = f"lot_size 必须为正整数，收到 {lot_size}"
        raise ValueError(msg)
    if qty <= 0:
        return 0
    aligned = (qty // lot_size) * lot_size
    if aligned < min_qty:
        return 0
    return aligned


def safe_div(numerator: Money, denominator: Money, *, default: Money = ZERO) -> Money:
    """安全除法，分母为零时返回默认值。

    用于计算占比、收益率等——分母为零通常意味着"尚无持仓/无基数"，
    此时返回 0 比抛异常更符合业务语义。

    Args:
        numerator: 分子。
        denominator: 分母。
        default: 分母为零时的返回值。

    Returns:
        商，或分母为零时的默认值。
    """
    if denominator == 0:
        return default
    return numerator / denominator


def round_down_cny(value: Money) -> Money:
    """金额向下取整到分。

    用于计算"最多能买多少"这类必须保守的场景。

    Args:
        value: 待处理金额。

    Returns:
        向下取整到分的金额。
    """
    return value.quantize(CNY, rounding=ROUND_DOWN)
