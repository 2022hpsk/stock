"""Symbol 归一化与领域类型测试。"""

from __future__ import annotations

import pytest

from quantstock.infra.types import (
    Exchange,
    Freq,
    make_symbol,
    parse_symbol,
    split_symbol,
)


class TestParseSymbol:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("600519.SH", "600519.SH"),
            ("600519.sh", "600519.SH"),
            ("sh600519", "600519.SH"),
            ("SH600519", "600519.SH"),
            ("sh.600519", "600519.SH"),
            ("600519.SS", "600519.SH"),
            ("  600519.SH  ", "600519.SH"),
            ("000001.SZ", "000001.SZ"),
            ("sz000001", "000001.SZ"),
            ("430047.BJ", "430047.BJ"),
        ],
    )
    def test_parse_symbol__known_formats__normalizes(self, raw: str, expected: str) -> None:
        assert parse_symbol(raw) == expected

    @pytest.mark.parametrize(
        ("code", "exchange"),
        [
            ("600519", Exchange.SH),  # 沪市主板
            ("688111", Exchange.SH),  # 科创板
            ("510300", Exchange.SH),  # 沪市 ETF
            ("000001", Exchange.SZ),  # 深市主板
            ("300750", Exchange.SZ),  # 创业板
            ("159915", Exchange.SZ),  # 深市 ETF
            ("430047", Exchange.BJ),  # 北交所
            ("830799", Exchange.BJ),
            ("871981", Exchange.BJ),
        ],
    )
    def test_parse_symbol__bare_code__infers_exchange(self, code: str, exchange: Exchange) -> None:
        assert parse_symbol(code) == f"{code}.{exchange.value}"

    @pytest.mark.parametrize(
        "raw",
        ["", "abc", "60051", "6005199", "600519.XX", "999999", "600519.SH.SZ"],
    )
    def test_parse_symbol__invalid__raises(self, raw: str) -> None:
        with pytest.raises(ValueError, match=r"无法识别|Symbol 格式非法|无法由代码"):
            parse_symbol(raw)


class TestMakeAndSplit:
    def test_make_symbol__valid__builds(self) -> None:
        assert make_symbol("600519", Exchange.SH) == "600519.SH"

    @pytest.mark.parametrize("code", ["60051", "6005199", "60051a", ""])
    def test_make_symbol__bad_code__raises(self, code: str) -> None:
        with pytest.raises(ValueError, match="6 位数字"):
            make_symbol(code, Exchange.SH)

    def test_split_symbol__roundtrips(self) -> None:
        symbol = make_symbol("300750", Exchange.SZ)
        code, exchange = split_symbol(symbol)
        assert (code, exchange) == ("300750", Exchange.SZ)

    def test_split_symbol__malformed__raises(self) -> None:
        with pytest.raises(ValueError, match="格式非法"):
            split_symbol("600519")


class TestFreq:
    @pytest.mark.parametrize("freq", [Freq.M1, Freq.M5, Freq.M15, Freq.M30, Freq.M60])
    def test_is_intraday__minute_freqs__true(self, freq: Freq) -> None:
        assert freq.is_intraday

    @pytest.mark.parametrize("freq", [Freq.D, Freq.W, Freq.M])
    def test_is_intraday__daily_and_above__false(self, freq: Freq) -> None:
        assert not freq.is_intraday
