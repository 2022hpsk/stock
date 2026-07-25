"""复权换算（红线 R4）。

三种口径的分工（见 docs/04-数据规格.md 第三节）：

- **后复权 hfq**：因子计算与回测收益。历史价格不随新的除权事件变动，保证研究可复现。
- **不复权 none**：下单价格、涨跌停判断、界面展示。必须是真实市场价。
- **前复权 qfq**：仅图表展示。由 hfq 与最新复权因子实时换算，**不落盘**——
  它会随每次除权而整体变动，落盘等于每次除权都要重写全部历史。
"""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal

from quantstock.infra.errors import AdjustMismatchError
from quantstock.infra.money import quantize_price
from quantstock.infra.types import Adjust, Money, Symbol

__all__ = [
    "assert_adjust",
    "convert",
    "hfq_to_none",
    "hfq_to_qfq",
    "none_to_hfq",
    "qfq_to_hfq",
]


def assert_adjust(actual: Adjust, expected: Adjust, *, context: str = "") -> None:
    """断言复权口径一致。

    任何接收价格序列的函数都应在入口调用本函数——混用口径造成的错误
    极难在事后发现，因为数字看起来都"像价格"。

    Args:
        actual: 实际口径。
        expected: 期望口径。
        context: 出错时附加的上下文说明。

    Raises:
        AdjustMismatchError: 口径不一致。
    """
    if actual is not expected:
        msg = "复权口径不匹配"
        raise AdjustMismatchError(
            msg, expected=expected.value, actual=actual.value, context=context
        )


def none_to_hfq(price: Money, adj_factor: Decimal) -> Money:
    """不复权价 → 后复权价。

    Args:
        price: 不复权真实价。
        adj_factor: 该日的复权因子。

    Returns:
        后复权价。

    Raises:
        ValueError: 复权因子非正。
    """
    _check_factor(adj_factor)
    return quantize_price(price * adj_factor)


def hfq_to_none(price: Money, adj_factor: Decimal) -> Money:
    """后复权价 → 不复权价。

    Args:
        price: 后复权价。
        adj_factor: 该日的复权因子。

    Returns:
        不复权真实价。

    Raises:
        ValueError: 复权因子非正。
    """
    _check_factor(adj_factor)
    return quantize_price(price / adj_factor)


def hfq_to_qfq(price: Money, latest_factor: Decimal) -> Money:
    """后复权价 → 前复权价。

    公式：``qfq = hfq / adj_factor_latest``。

    只依赖**最新**复权因子，不需要当日因子——这正是前复权的定义：
    以最新价为锚，把历史价格整体缩放。验证最新一天::

        qfq = hfq_latest / factor_latest
            = none_latest × factor_latest / factor_latest
            = none_latest

    即最新日的前复权价恰好等于真实价。

    也正因为分母是"最新因子"，每发生一次除权，全部历史前复权价都会整体变动——
    这就是它不落盘、只在展示时实时换算的原因。

    Args:
        price: 后复权价。
        latest_factor: 最新的复权因子。

    Returns:
        前复权价。

    Raises:
        ValueError: 复权因子非正。
    """
    _check_factor(latest_factor)
    return quantize_price(price / latest_factor)


def qfq_to_hfq(price: Money, latest_factor: Decimal) -> Money:
    """前复权价 → 后复权价。

    Args:
        price: 前复权价。
        latest_factor: 最新的复权因子。

    Returns:
        后复权价。

    Raises:
        ValueError: 复权因子非正。
    """
    _check_factor(latest_factor)
    return quantize_price(price * latest_factor)


def convert(
    price: Money,
    *,
    source: Adjust,
    target: Adjust,
    adj_factor: Decimal,
    latest_factor: Decimal | None = None,
) -> Money:
    """在三种复权口径间换算。

    Args:
        price: 原始价格。
        source: 原口径。
        target: 目标口径。
        adj_factor: 该日复权因子。
        latest_factor: 最新复权因子，转换到/自 qfq 时必需。

    Returns:
        换算后的价格。

    Raises:
        ValueError: 缺少 ``latest_factor`` 或复权因子非法。
    """
    if source is target:
        return price

    # 统一先转成 hfq 作为中枢，再转到目标口径
    if source is Adjust.NONE:
        hfq = none_to_hfq(price, adj_factor)
    elif source is Adjust.QFQ:
        if latest_factor is None:
            msg = "从前复权换算需要提供 latest_factor"
            raise ValueError(msg)
        hfq = qfq_to_hfq(price, latest_factor)
    else:
        hfq = price

    if target is Adjust.HFQ:
        return hfq
    if target is Adjust.NONE:
        return hfq_to_none(hfq, adj_factor)
    if latest_factor is None:
        msg = "换算到前复权需要提供 latest_factor"
        raise ValueError(msg)
    return hfq_to_qfq(hfq, latest_factor)


def check_return_consistency(
    hfq_prices: Iterable[Money],
    none_prices: Iterable[Money],
    *,
    tolerance: Decimal = Decimal("1e-6"),
    symbol: Symbol | None = None,
) -> list[int]:
    """校验后复权与不复权序列的收益率一致性（DQ07）。

    非除权日两者的日收益率应当完全一致；不一致说明复权因子算错了。
    除权日本就应当不一致——那正是复权要修正的东西，调用方应先剔除除权日。

    Args:
        hfq_prices: 后复权收盘价序列。
        none_prices: 不复权收盘价序列，长度须与前者一致。
        tolerance: 允许的偏差。
        symbol: 标的，仅用于错误信息。

    Returns:
        不一致的位置下标列表（指向后一日）。

    Raises:
        ValueError: 两个序列长度不一致。
    """
    hfq = list(hfq_prices)
    none = list(none_prices)
    if len(hfq) != len(none):
        msg = f"序列长度不一致：hfq={len(hfq)}, none={len(none)}"
        raise ValueError(msg)

    mismatches: list[int] = []
    for i in range(1, len(hfq)):
        if hfq[i - 1] <= 0 or none[i - 1] <= 0:
            continue
        hfq_ret = (hfq[i] - hfq[i - 1]) / hfq[i - 1]
        none_ret = (none[i] - none[i - 1]) / none[i - 1]
        if abs(hfq_ret - none_ret) > tolerance:
            mismatches.append(i)
    _ = symbol
    return mismatches


def _check_factor(factor: Decimal) -> None:
    """校验复权因子合法。

    Args:
        factor: 复权因子。

    Raises:
        ValueError: 因子非正。
    """
    if factor <= 0:
        msg = f"复权因子必须为正，收到 {factor}"
        raise ValueError(msg)
