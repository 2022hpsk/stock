"""金额计算测试。

红线 R1：金额必须 Decimal。这里用属性测试覆盖"整手对齐永不超出原始数量"等不变量。
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantstock.infra.money import (
    align_lot,
    money,
    quantize_cny,
    round_down_cny,
    safe_div,
    to_money,
)


class TestMoney:
    def test_money__rejects_float(self) -> None:
        """红线 R1：float 字面量已损失精度，转换只会把误差固化。"""
        with pytest.raises(TypeError, match="禁止用 float 构造金额"):
            money(123.45)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("123.45", Decimal("123.45")), (100, Decimal("100"))],
    )
    def test_money__accepts_str_and_int(self, value: str | int, expected: Decimal) -> None:
        assert money(value) == expected

    def test_money__unparseable__raises(self) -> None:
        with pytest.raises(ValueError, match="无法解析为金额"):
            money("not-a-number")

    def test_to_money__explicit_float_conversion__quantizes(self) -> None:
        assert to_money(123.456) == Decimal("123.46")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1.005", "1.01"),  # 四舍五入而非银行家舍入
            ("1.004", "1.00"),
            ("-1.005", "-1.01"),
        ],
    )
    def test_quantize_cny__half_up(self, raw: str, expected: str) -> None:
        assert quantize_cny(Decimal(raw)) == Decimal(expected)

    def test_round_down_cny__always_floors(self) -> None:
        """“最多能买多少”这类场景必须向下取整。"""
        assert round_down_cny(Decimal("1.009")) == Decimal("1.00")


class TestAlignLot:
    @pytest.mark.parametrize(
        ("qty", "expected"),
        [(0, 0), (99, 0), (100, 100), (150, 100), (199, 100), (1234, 1200)],
    )
    def test_align_lot__default_100(self, qty: int, expected: int) -> None:
        assert align_lot(qty) == expected

    def test_align_lot__negative__returns_zero(self) -> None:
        assert align_lot(-500) == 0

    def test_align_lot__star_market_min_qty(self) -> None:
        """科创板最低 200 股，之后可按 1 股递增。"""
        assert align_lot(150, lot_size=1, min_qty=200) == 0
        assert align_lot(201, lot_size=1, min_qty=200) == 201

    def test_align_lot__bad_lot_size__raises(self) -> None:
        with pytest.raises(ValueError, match="lot_size 必须为正整数"):
            align_lot(100, lot_size=0)

    @given(qty=st.integers(min_value=-10_000, max_value=10_000_000))
    def test_align_lot__never_exceeds_input(self, qty: int) -> None:
        """不变量：对齐后数量永不超过原始数量，且必为整手。

        向下取整是有意为之——宁可少买，不可超出可用资金。
        """
        aligned = align_lot(qty)
        assert aligned <= max(qty, 0)
        assert aligned % 100 == 0
        assert aligned >= 0


class TestSafeDiv:
    def test_safe_div__zero_denominator__returns_default(self) -> None:
        """分母为零通常意味着“尚无持仓/无基数”，返回 0 比抛异常更贴合业务。"""
        assert safe_div(Decimal("100"), Decimal("0")) == Decimal("0")

    def test_safe_div__custom_default(self) -> None:
        assert safe_div(Decimal("1"), Decimal("0"), default=Decimal("-1")) == Decimal("-1")

    def test_safe_div__normal(self) -> None:
        assert safe_div(Decimal("10"), Decimal("4")) == Decimal("2.5")
