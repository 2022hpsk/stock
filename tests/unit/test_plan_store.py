"""计划落盘、读回与执行服务测试。

重点验证红线 R6（可追溯可复现）：计划落盘再读回必须**逐字段等价**——
审计快照读回来变了样，等于没有审计。
"""

from __future__ import annotations

import datetime as dt
import json
from decimal import Decimal
from pathlib import Path

import pytest

from quantstock.advisor.store import PlanNotFoundError, PlanStore
from quantstock.advisor.types import (
    IntelEvidence,
    IntelImpact,
    RationaleBundle,
    RejectedCandidate,
    TradePlan,
)
from quantstock.infra.clock import CST, FrozenClock, set_clock
from quantstock.infra.errors import QuantStockError
from quantstock.infra.serde import SerdeError, from_jsonable, to_jsonable
from quantstock.infra.types import PlanId, Side, Symbol
from tests.unit.test_execution import TODAY, A, B, intent, plan

pytestmark = pytest.mark.usefixtures("_frozen_store_clock")


@pytest.fixture(autouse=True)
def _frozen_store_clock() -> None:
    """固定时钟。"""
    set_clock(FrozenClock(dt.datetime(2026, 7, 24, 16, 0, tzinfo=CST)))


def _rich_plan() -> TradePlan:
    """构造一个字段尽量铺满的计划，用来压编解码。"""
    base = plan(
        intent(A, intent_id="i1"),
        intent(B, side=Side.SELL, qty=300, price="55.5", intent_id="i2"),
    )
    first = base.intents[0]
    rationale = RationaleBundle(
        verdict=first.rationale.verdict,
        quant_evidence=first.rationale.quant_evidence,
        technical=first.rationale.technical,
        intel_evidence=(
            IntelEvidence(
                title="白酒动销回暖",
                source="财联社",
                published_at=dt.datetime(2026, 7, 23, 9, 15, tzinfo=CST),
                url="https://example.com/news/1",
                domain="consumer",
                sentiment=0.4,
                importance=3,
                impact=IntelImpact.SUPPORT,
                summary="渠道调研显示动销环比改善",
            ),
        ),
        counter_evidence=first.rationale.counter_evidence,
        falsification=first.rationale.falsification,
        risk_notes=("估值处于近三年高位",),
        confidence=0.62,
        confidence_basis="因子一致性较高",
        intel_absent_note="",
        llm_involved=True,
        llm_adjustment=0.08,
    )
    from dataclasses import replace  # noqa: PLC0415 - 仅测试内使用

    return replace(
        base,
        intents=(replace(first, rationale=rationale), base.intents[1]),
        rejected=(RejectedCandidate(symbol=Symbol("000001.SZ"), reason="流动性不足"),),
        incomplete=((Symbol("601318.SH"), "缺④反面证据"),),
        data_fingerprint="sha256:abc123",
        strategy_versions={"momentum_trend": "1.2.0"},
        param_hash="p-9f8e",
        summary="今日 2 条建议",
    )


class TestSerde:
    """通用编解码。"""

    def test_roundtrip_preserves_every_field(self) -> None:
        original = _rich_plan()
        restored = from_jsonable(TradePlan, to_jsonable(original))
        assert restored == original

    def test_decimal_never_passes_through_float(self) -> None:
        # 金额编成字符串而非 JSON 数字：JSON 数字会被解析成 float（红线 R1）
        encoded = to_jsonable(Decimal("0.1"))
        assert encoded == "0.1"
        assert isinstance(encoded, str)

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(SerdeError, match="naive"):
            to_jsonable(dt.datetime(2026, 7, 24, 9, 30))  # noqa: DTZ001 - 刻意构造违规输入

    def test_decoding_naive_datetime_rejected(self) -> None:
        with pytest.raises(SerdeError, match="时区"):
            from_jsonable(dt.datetime, "2026-07-24T09:30:00")

    def test_unknown_field_is_an_error_not_silently_ignored(self) -> None:
        payload = to_jsonable(_rich_plan())
        payload["surprise"] = 1
        with pytest.raises(SerdeError, match="未知字段"):
            from_jsonable(TradePlan, payload)

    def test_optional_none_roundtrips(self) -> None:
        original = _rich_plan()
        assert original.confirmed_at is None
        assert from_jsonable(TradePlan, to_jsonable(original)).confirmed_at is None

    def test_unsupported_type_rejected(self) -> None:
        with pytest.raises(SerdeError, match="不支持"):
            to_jsonable(object())


class TestPlanStore:
    """计划仓库。"""

    def test_save_then_load_is_identical(self, tmp_path: Path) -> None:
        store = PlanStore(tmp_path)
        original = _rich_plan()
        store.save(original)
        assert store.load(TODAY, original.plan_id) == original

    def test_saved_file_is_valid_json_with_schema_version(self, tmp_path: Path) -> None:
        store = PlanStore(tmp_path)
        path = store.save(_rich_plan())
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == 1
        assert payload["plan"]["plan_id"] == "p1"

    def test_overwrite_refused(self, tmp_path: Path) -> None:
        # 审计快照只追加不修改（红线 R8 同思路）
        store = PlanStore(tmp_path)
        store.save(_rich_plan())
        with pytest.raises(QuantStockError, match="不允许覆盖"):
            store.save(_rich_plan())

    def test_missing_plan_raises(self, tmp_path: Path) -> None:
        with pytest.raises(PlanNotFoundError):
            PlanStore(tmp_path).load(TODAY, PlanId("nope"))

    def test_schema_version_mismatch_refuses_to_execute(self, tmp_path: Path) -> None:
        store = PlanStore(tmp_path)
        path = store.save(_rich_plan())
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = 99
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(QuantStockError, match="schema 版本不匹配"):
            store.load(TODAY, PlanId("p1"))

    def test_confirmation_does_not_touch_the_original_file(self, tmp_path: Path) -> None:
        store = PlanStore(tmp_path)
        original = _rich_plan()
        path = store.save(original)
        before = path.read_bytes()

        confirmed = store.mark_confirmed(original, confirmed_by="张三")
        assert confirmed.is_confirmed
        assert path.read_bytes() == before, "确认信息必须写在独立文件，原始快照不可变"

        reloaded = store.load(TODAY, original.plan_id)
        assert reloaded.confirmed_by == "张三"
        assert reloaded.confirmed_at is not None

    def test_confirmation_requires_a_confirmer(self, tmp_path: Path) -> None:
        store = PlanStore(tmp_path)
        with pytest.raises(QuantStockError, match="确认人"):
            store.mark_confirmed(_rich_plan(), confirmed_by="   ")

    def test_latest_skips_confirm_sidecar_files(self, tmp_path: Path) -> None:
        store = PlanStore(tmp_path)
        original = _rich_plan()
        store.save(original)
        store.mark_confirmed(original, confirmed_by="张三")
        latest = store.latest(TODAY)
        assert latest is not None
        assert latest.plan_id == original.plan_id

    def test_latest_returns_none_when_no_plan(self, tmp_path: Path) -> None:
        assert PlanStore(tmp_path).latest(TODAY) is None

    def test_list_dates_ignores_non_date_directories(self, tmp_path: Path) -> None:
        store = PlanStore(tmp_path)
        store.save(_rich_plan())
        (tmp_path / "scratch").mkdir()
        assert store.list_dates() == [TODAY]

    def test_list_dates_on_missing_root(self, tmp_path: Path) -> None:
        assert PlanStore(tmp_path / "absent").list_dates() == []
