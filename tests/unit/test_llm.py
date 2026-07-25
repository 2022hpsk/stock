"""大模型集成测试。

覆盖 docs/10-大模型集成规格.md 第十一节的全部测试要求：

======================  ==========================================================
测试                    要求
======================  ==========================================================
回放确定性              同一缓存跑两次结果逐笔一致
回测隔离                回测中任何实时调用都抛 ``LLMLiveCallInBacktestError``
降级完整性              off / 超时 / 非法 JSON 三种情况下系统均能正常出结果
影响有界                属性测试：任意 adjustment ∈ [-1,1]，偏移 ≤ α
反幻觉                  引用不存在的材料 → 整个输出作废
泄漏检测                真实名 vs 虚构名的输出差异率 < 10%
成本预算                超预算自动降级并告警
======================  ==========================================================

**CI 永不打真实 API**：全部用打桩供应商。
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from quantstock.infra.clock import CST, FrozenClock, set_clock
from quantstock.infra.errors import (
    LLMError,
    LLMLiveCallInBacktestError,
    LLMOutputInvalidError,
)
from quantstock.infra.types import Symbol
from quantstock.llm.anonymize import Anonymizer, strip_absolute_dates
from quantstock.llm.budget import BudgetGuard, UsageRecord, estimate_cost
from quantstock.llm.cache import LLMCache, canonical_json, compute_cache_key
from quantstock.llm.client import LLMClient, TaskCall
from quantstock.llm.influence import (
    HARD_INFLUENCE_CAP,
    apply_conviction,
    apply_exposure,
)
from quantstock.llm.protocols import CompletionRequest, CompletionResponse
from quantstock.llm.schemas import (
    ExplanationOutput,
    IntelClassification,
    MarketJudgement,
    PositionJudgement,
)
from quantstock.llm.tasks import ExplainTask, IntelClassifyTask, MarketJudgeTask, PositionJudgeTask
from quantstock.llm.validate import extract_json, parse_output, validate_evidence_refs

MAOTAI = Symbol("600519.SH")
CATL = Symbol("300750.SZ")
NOW = dt.datetime(2026, 7, 25, 18, 0, tzinfo=CST)

GOOD_JUDGEMENT = {
    "symbol_ref": "600519.SH",
    "positive_factors": [{"statement": "动量处于高位", "evidence_ref": "m1"}],
    "negative_factors": [{"statement": "估值偏高", "evidence_ref": "m2"}],
    "conflicts": ["动量与估值方向相反"],
    "risk_level": "MEDIUM",
    "conviction_adjustment": 0.4,
    "falsification": ["跌破 MA20 则证伪"],
    "insufficient_evidence": False,
}

MATERIALS = {"m1": "近 60 日动量位于全市场 87% 分位", "m2": "PE_TTM 处于近五年 78% 分位"}


@pytest.fixture(autouse=True)
def _frozen() -> None:
    """固定时钟。"""
    set_clock(FrozenClock(NOW))


class StubProvider:
    """打桩供应商。CI 永远不打真实 API。"""

    def __init__(self, *responses: str, fail: Exception | None = None) -> None:
        self._responses = list(responses) or [json.dumps(GOOD_JUDGEMENT)]
        self._fail = fail
        self.calls: list[CompletionRequest] = []

    @property
    def name(self) -> str:
        return "stub"

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        self.calls.append(request)
        if self._fail is not None:
            raise self._fail
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        return CompletionResponse(
            text=self._responses[index],
            model=request.model,
            input_tokens=1000,
            output_tokens=200,
        )


def make_call(payload: dict[str, object] | None = None) -> TaskCall:
    """构造一次任务调用。"""
    return TaskCall(
        task_id="position_judge",
        prompt_version="v1",
        model_id="claude-sonnet-5",
        system="sys",
        user="user",
        payload=payload if payload is not None else {"materials": MATERIALS},
    )


class TestCacheKey:
    """缓存键。"""

    def test_key_is_stable_across_dict_order(self) -> None:
        # 键顺序不同就算出两个 key 的话，命中率会崩掉，回测退化成实时调用
        left = compute_cache_key(
            task_id="t",
            prompt_version="v1",
            model_id="m",
            temperature=0.0,
            payload={"a": 1, "b": 2},
        )
        right = compute_cache_key(
            task_id="t",
            prompt_version="v1",
            model_id="m",
            temperature=0.0,
            payload={"b": 2, "a": 1},
        )
        assert left == right

    @pytest.mark.parametrize(
        "override",
        [
            {"prompt_version": "v2"},
            {"model_id": "other"},
            {"temperature": 0.5},
            {"task_id": "other"},
            {"payload": {"a": 9}},
        ],
    )
    def test_every_component_changes_the_key(self, override: dict[str, object]) -> None:
        # 改提示词等同于改策略——不能让"润色措辞"悄悄复用旧回测结果
        base = {
            "task_id": "t",
            "prompt_version": "v1",
            "model_id": "m",
            "temperature": 0.0,
            "payload": {"a": 1},
        }
        assert compute_cache_key(**base) != compute_cache_key(**{**base, **override})  # type: ignore[arg-type]

    def test_canonical_json_keeps_chinese_readable(self) -> None:
        assert "茅台" in canonical_json({"name": "茅台"})


class TestCache:
    """快照缓存。"""

    def test_roundtrip(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path)
        entry = cache.make_entry(
            cache_key="abc123",
            task_id="position_judge",
            model_id="m",
            prompt_version="v1",
            temperature=0.0,
            request={"system": "s"},
            response={"text": "t"},
            cost_usd=0.01,
        )
        cache.put(entry)
        assert cache.get("position_judge", "abc123") == entry

    def test_miss_returns_none_and_counts(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path)
        assert cache.get("t", "nope") is None
        assert cache.stats.misses == 1
        assert cache.stats.hit_rate == 0.0

    def test_corrupt_entry_treated_as_miss(self, tmp_path: Path) -> None:
        # 抛错会中断整个回测；未命中会被安全降级
        cache = LLMCache(tmp_path)
        path = cache.path_for("t", "deadbeef")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ not json", encoding="utf-8")
        assert cache.get("t", "deadbeef") is None

    def test_coverage(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path)
        cache.put(
            cache.make_entry(
                cache_key="k1",
                task_id="t",
                model_id="m",
                prompt_version="v1",
                temperature=0.0,
                request={},
                response={},
            )
        )
        assert cache.coverage("t", ["k1", "k2"]) == 0.5
        assert cache.coverage("t", []) == 1.0

    def test_total_cost_and_count(self, tmp_path: Path) -> None:
        cache = LLMCache(tmp_path)
        for i, cost in enumerate([0.01, 0.02]):
            cache.put(
                cache.make_entry(
                    cache_key=f"k{i}",
                    task_id="t",
                    model_id="m",
                    prompt_version="v1",
                    temperature=0.0,
                    request={},
                    response={},
                    cost_usd=cost,
                )
            )
        assert cache.count("t") == 2
        assert cache.total_cost("t") == pytest.approx(0.03)


class TestModes:
    """三种运行模式。"""

    def test_off_never_calls(self, tmp_path: Path) -> None:
        provider = StubProvider()
        client = LLMClient(mode="off", provider=provider, cache_dir=tmp_path)
        result = client.complete(make_call(), PositionJudgement)

        assert result.output is None
        assert provider.calls == []
        assert "关闭" in result.reason

    def test_live_calls_and_writes_cache(self, tmp_path: Path) -> None:
        provider = StubProvider()
        client = LLMClient(mode="live", provider=provider, cache_dir=tmp_path)
        result = client.complete(make_call(), PositionJudgement)

        assert result.used_llm
        assert len(provider.calls) == 1
        assert client.cache is not None
        assert client.cache.has("position_judge", result.cache_key)

    def test_replay_reads_cache_without_calling(self, tmp_path: Path) -> None:
        writer = StubProvider()
        LLMClient(mode="live", provider=writer, cache_dir=tmp_path).complete(
            make_call(), PositionJudgement
        )

        reader = StubProvider()
        client = LLMClient(mode="replay", provider=reader, cache_dir=tmp_path)
        result = client.complete(make_call(), PositionJudgement)

        assert result.used_llm
        assert result.from_cache
        assert reader.calls == [], "回放模式绝不能发起实时调用"

    def test_replay_miss_degrades_quietly(self, tmp_path: Path) -> None:
        # 未命中不是错误：那个决策点没预计算过，回测照常继续
        client = LLMClient(mode="replay", provider=StubProvider(), cache_dir=tmp_path)
        result = client.complete(make_call(), PositionJudgement)

        assert result.output is None
        assert "未命中" in result.reason

    def test_replay_determinism(self, tmp_path: Path) -> None:
        # 同一缓存跑两次必须逐字段一致
        LLMClient(mode="live", provider=StubProvider(), cache_dir=tmp_path).complete(
            make_call(), PositionJudgement
        )
        client = LLMClient(mode="replay", cache_dir=tmp_path)
        first = client.complete(make_call(), PositionJudgement)
        second = client.complete(make_call(), PositionJudgement)
        assert first.output == second.output


class TestBacktestIsolation:
    """回测隔离（红线 LR3）。"""

    def test_live_mode_in_backtest_rejected_at_construction(self, tmp_path: Path) -> None:
        with pytest.raises(LLMLiveCallInBacktestError, match="replay"):
            LLMClient(mode="live", cache_dir=tmp_path, in_backtest=True)

    def test_replay_is_allowed_in_backtest(self, tmp_path: Path) -> None:
        client = LLMClient(mode="replay", cache_dir=tmp_path, in_backtest=True)
        assert client.mode == "replay"

    def test_off_is_allowed_in_backtest(self, tmp_path: Path) -> None:
        assert LLMClient(mode="off", in_backtest=True).mode == "off"


class TestDegradation:
    """降级完整性。任何失败都不该中断决策链。"""

    def test_provider_timeout_degrades(self, tmp_path: Path) -> None:
        provider = StubProvider(fail=TimeoutError("超时"))
        client = LLMClient(mode="live", provider=provider, cache_dir=tmp_path)
        result = client.complete(make_call(), PositionJudgement)

        assert result.output is None
        assert "TimeoutError" in result.reason

    def test_invalid_json_degrades(self, tmp_path: Path) -> None:
        client = LLMClient(mode="live", provider=StubProvider("这不是 JSON"), cache_dir=tmp_path)
        result = client.complete(make_call(), PositionJudgement)
        assert result.output is None
        assert "校验" in result.reason

    def test_schema_violation_degrades(self, tmp_path: Path) -> None:
        # conviction_adjustment 越界 → 契约拒绝 → 不使用 LLM
        bad = json.dumps({**GOOD_JUDGEMENT, "conviction_adjustment": 5.0})
        client = LLMClient(mode="live", provider=StubProvider(bad), cache_dir=tmp_path)
        assert client.complete(make_call(), PositionJudgement).output is None

    def test_no_provider_degrades(self, tmp_path: Path) -> None:
        client = LLMClient(mode="live", cache_dir=tmp_path)
        assert "供应商" in client.complete(make_call(), PositionJudgement).reason

    def test_invalid_response_is_still_cached_for_diagnosis(self, tmp_path: Path) -> None:
        # 一份没通过校验的响应是排查提示词问题的唯一线索
        client = LLMClient(mode="live", provider=StubProvider("垃圾"), cache_dir=tmp_path)
        result = client.complete(make_call(), PositionJudgement)
        assert client.cache is not None
        assert client.cache.has("position_judge", result.cache_key)

    def test_cached_invalid_output_does_not_trigger_live_call(self, tmp_path: Path) -> None:
        LLMClient(mode="live", provider=StubProvider("垃圾"), cache_dir=tmp_path).complete(
            make_call(), PositionJudgement
        )
        reader = StubProvider()
        client = LLMClient(mode="replay", provider=reader, cache_dir=tmp_path)
        result = client.complete(make_call(), PositionJudgement)

        assert result.output is None
        assert reader.calls == []


class TestInfluenceBound:
    """影响有界（红线 LR2）——整个 LLM 集成的安全性压在这里。"""

    @given(
        base=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
        adjustment=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        alpha=st.floats(min_value=0.0, max_value=HARD_INFLUENCE_CAP),
    )
    def test_offset_never_exceeds_alpha(self, base: float, adjustment: float, alpha: float) -> None:
        result = apply_conviction(base, adjustment, alpha=alpha)
        assert abs(result.final_score - base) <= abs(base) * alpha + 1e-9

    @given(
        base=st.floats(min_value=0.01, max_value=10.0, allow_nan=False),
        adjustment=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    )
    def test_sign_is_never_flipped(self, base: float, adjustment: float) -> None:
        # α ≤ 0.2 意味着 (1 + α·adj) ≥ 0.8 > 0，LLM 不可能把正分翻成负分
        result = apply_conviction(base, adjustment, alpha=HARD_INFLUENCE_CAP)
        assert result.final_score > 0

    def test_none_adjustment_is_identity(self) -> None:
        # LLM 挂了 → 原样返回 → 系统退化为纯量化，照常出建议
        result = apply_conviction(0.62, None, alpha=0.15, note="API 超时")
        assert result.final_score == 0.62
        assert not result.applied
        assert "API 超时" in result.explain()

    def test_alpha_above_hard_cap_rejected(self) -> None:
        # 静默截断会让用户以为设置生效了
        with pytest.raises(LLMError, match="硬上限"):
            apply_conviction(1.0, 0.5, alpha=0.5)

    def test_negative_alpha_rejected(self) -> None:
        with pytest.raises(LLMError, match="不能为负"):
            apply_conviction(1.0, 0.5, alpha=-0.1)

    def test_out_of_range_adjustment_is_clamped_not_raised(self) -> None:
        # 纵深防御：走到这里还越界说明有路径绕过了契约校验，裁剪保证影响仍有界
        result = apply_conviction(1.0, 99.0, alpha=0.15)
        assert result.adjustment == 1.0
        assert result.final_score == pytest.approx(1.15)

    def test_explain_shows_the_arithmetic(self) -> None:
        # 界面的 🤖 展开视图要能看到完整算式（红线 LR8）
        text = apply_conviction(0.62, 0.4, alpha=0.15).explain()
        assert "🤖" in text
        assert "0.15" in text

    @given(adjustment=st.floats(min_value=0.001, max_value=1.0, allow_nan=False))
    def test_exposure_can_only_decrease(self, adjustment: float) -> None:
        # 让模型基于新闻情绪加仓，是在最容易被情绪裹挟的时候加杠杆
        result = apply_exposure(0.6, adjustment, alpha=0.15)
        assert result.final_score == 0.6
        assert "截断" in result.note

    def test_exposure_negative_applies(self) -> None:
        result = apply_exposure(0.6, -0.5, alpha=0.20)
        assert result.final_score < 0.6
        assert result.final_score == pytest.approx(0.6 * 0.9)


class TestSchemas:
    """结构化契约（红线 LR5）。"""

    def test_extra_field_rejected(self) -> None:
        # 多出来的字段通常意味着模型没按契约作答
        with pytest.raises(LLMOutputInvalidError):
            parse_output(json.dumps({**GOOD_JUDGEMENT, "surprise": 1}), PositionJudgement)

    def test_insufficient_evidence_forces_zero_adjustment(self) -> None:
        # 说"材料不够"却给方向性调节，是自相矛盾
        payload = {**GOOD_JUDGEMENT, "insufficient_evidence": True}
        parsed = parse_output(json.dumps(payload), PositionJudgement)
        assert parsed.conviction_adjustment == 0.0

    def test_intel_classification_neutralized_when_insufficient(self) -> None:
        parsed = IntelClassification.model_validate(
            {"sentiment": 0.9, "event_type": "buyback", "insufficient_evidence": True}
        )
        assert parsed.sentiment == 0.0
        assert parsed.event_type is None

    def test_market_judgement_neutralized_when_insufficient(self) -> None:
        parsed = MarketJudgement.model_validate(
            {"exposure_adjustment": -0.8, "insufficient_evidence": True}
        )
        assert parsed.exposure_adjustment == 0.0

    def test_explanation_has_no_numeric_outlet(self) -> None:
        # 解释环节不该有能改变数字的通道
        numeric = [
            name
            for name, field in ExplanationOutput.model_fields.items()
            if field.annotation in (float, int)
        ]
        assert numeric == []

    def test_evidence_refs_collected(self) -> None:
        parsed = parse_output(json.dumps(GOOD_JUDGEMENT), PositionJudgement)
        assert parsed.evidence_refs() == frozenset({"m1", "m2"})


class TestJsonExtraction:
    """从模型回复里抽 JSON。"""

    def test_plain_json(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_json(self) -> None:
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_with_preamble(self) -> None:
        assert extract_json('好的，结果如下：{"a": 1}') == {"a": 1}

    def test_no_json_raises(self) -> None:
        with pytest.raises(LLMOutputInvalidError, match="找不到 JSON"):
            extract_json("完全没有 JSON")

    def test_malformed_json_raises(self) -> None:
        with pytest.raises(LLMOutputInvalidError, match="不是合法 JSON"):
            extract_json("{ broken: }")

    def test_json_array_rejected(self) -> None:
        with pytest.raises(LLMOutputInvalidError):
            extract_json("[1, 2, 3]")


class TestAntiHallucination:
    """反幻觉校验。"""

    def test_fabricated_reference_voids_everything(self) -> None:
        # 编造一个看似合理的材料编号来支撑结论，是幻觉最危险的形态
        outcome = validate_evidence_refs(frozenset({"m1", "m99"}), MATERIALS)
        assert not outcome.ok
        assert "m99" in outcome.reason

    def test_real_references_pass(self) -> None:
        assert validate_evidence_refs(frozenset({"m1", "m2"}), MATERIALS)

    def test_task_rejects_hallucinated_judgement(self, tmp_path: Path) -> None:
        bad = json.dumps(
            {
                **GOOD_JUDGEMENT,
                "positive_factors": [{"statement": "编造的依据", "evidence_ref": "不存在"}],
            }
        )
        client = LLMClient(mode="live", provider=StubProvider(bad), cache_dir=tmp_path)
        task = PositionJudgeTask(client, alpha=0.15, model_id="claude-sonnet-5")
        outcome = task.run(MAOTAI, base_score=0.6, materials=MATERIALS, as_of="T日")

        assert outcome.judgement is None
        assert not outcome.used_llm
        assert outcome.influence.final_score == 0.6, "作废后必须退化为纯量化"
        assert "幻觉" in outcome.reason


class TestAnonymize:
    """匿名化（红线 LR4）。"""

    def test_names_and_codes_replaced(self) -> None:
        mapper = Anonymizer()
        mapper.register(MAOTAI, "贵州茅台", "茅台")
        result = mapper.anonymize("贵州茅台（600519.SH）发布公告，茅台股价波动")

        assert "贵州茅台" not in result.text
        assert "600519" not in result.text
        assert "标的A" in result.text

    def test_same_symbol_gets_same_label(self) -> None:
        # 同一次调用里代号必须稳定，否则模型看到的材料自相矛盾
        mapper = Anonymizer()
        mapper.register(MAOTAI, "贵州茅台")
        assert mapper.anonymize("贵州茅台").text == mapper.anonymize("贵州茅台").text

    def test_distinct_symbols_get_distinct_labels(self) -> None:
        mapper = Anonymizer()
        mapper.register(MAOTAI, "贵州茅台")
        mapper.register(CATL, "宁德时代")
        assert mapper.label_of(MAOTAI) != mapper.label_of(CATL)

    def test_long_name_replaced_first(self) -> None:
        # 否则"中国平安"会被"平安"抢先命中，留下"中国标的A"
        mapper = Anonymizer()
        mapper.register(Symbol("601318.SH"), "中国平安", "平安")
        assert "中国标的" not in mapper.anonymize("中国平安发布公告").text

    def test_restore_roundtrip(self) -> None:
        mapper = Anonymizer()
        mapper.register(MAOTAI, "贵州茅台")
        anonymized = mapper.anonymize("贵州茅台发布公告").text
        assert "600519.SH" in mapper.restore(anonymized)

    def test_resolve_label(self) -> None:
        mapper = Anonymizer()
        label = mapper.register(MAOTAI, "贵州茅台")
        assert mapper.resolve(label) == MAOTAI
        assert mapper.resolve("标的Z") is None

    def test_more_than_26_symbols(self) -> None:
        mapper = Anonymizer()
        labels = {mapper.register(Symbol(f"{600000 + i}.SH")) for i in range(30)}
        assert len(labels) == 30

    @pytest.mark.parametrize(
        "text",
        [
            "2023年3月15日发布公告",
            "2023-03-15 发布公告",
            "2023年业绩说明",
            "2023/03 数据",
        ],
    )
    def test_absolute_dates_stripped(self, text: str) -> None:
        # 提示词里出现具体日期等于告诉模型现在是哪一天
        stripped = strip_absolute_dates(text)
        assert "2023" not in stripped
        assert "〈日期〉" in stripped

    def test_relative_expressions_untouched(self) -> None:
        assert strip_absolute_dates("T-3 日的成交量") == "T-3 日的成交量"


class TestLeakageDetection:
    """训练集泄漏检测（docs/10 §5.2）。

    真实名与虚构名两版 prompt，输出差异率超阈值即判定模型在用内部知识。
    这里用打桩验证**检测机制本身**成立；接真实模型后同一套断言直接生效。
    """

    LEAK_THRESHOLD = 0.10

    def _sentiments(self, provider: StubProvider, tmp_path: Path, name: str) -> float:
        client = LLMClient(mode="live", provider=provider, cache_dir=tmp_path / name)
        task = IntelClassifyTask(client, model_id="claude-haiku-4-5-20251001")
        outcome = task.run(item_id="n1", title=f"{name}发布季度经营数据", body="营收同比持平")
        assert outcome.classification is not None
        return outcome.classification.sentiment

    def test_material_bound_model_gives_same_answer_for_either_name(self, tmp_path: Path) -> None:
        # 一个只看材料的模型，换公司名不该改变判断
        response = json.dumps({"event_type": "earnings_report", "sentiment": 0.05})
        real = self._sentiments(StubProvider(response), tmp_path, "贵州茅台")
        fake = self._sentiments(StubProvider(response), tmp_path, "虚构甲公司")
        assert abs(real - fake) <= self.LEAK_THRESHOLD

    def test_detector_catches_a_leaking_model(self, tmp_path: Path) -> None:
        # 换个名字就大幅改判 → 说明它在用先验知识而非材料
        real = self._sentiments(StubProvider(json.dumps({"sentiment": 0.85})), tmp_path, "贵州茅台")
        fake = self._sentiments(
            StubProvider(json.dumps({"sentiment": 0.0})), tmp_path, "虚构甲公司"
        )
        assert abs(real - fake) > self.LEAK_THRESHOLD

    def test_anonymized_backtest_prompt_hides_the_name(self, tmp_path: Path) -> None:
        provider = StubProvider()
        client = LLMClient(mode="live", provider=provider, cache_dir=tmp_path)
        mapper = Anonymizer()
        mapper.register(MAOTAI, "贵州茅台")
        task = PositionJudgeTask(
            client,
            alpha=0.15,
            model_id="claude-sonnet-5",
            anonymize=True,
            strip_dates=True,
        )
        task.run(
            MAOTAI,
            base_score=0.6,
            materials={"m1": "贵州茅台 2023年3月15日 发布公告"},
            as_of="T日",
            anonymizer=mapper,
        )
        sent = provider.calls[0].messages[0].content
        assert "贵州茅台" not in sent
        assert "2023" not in sent
        assert "标的A" in sent


class TestBudget:
    """成本预算（红线 LR7）。"""

    def test_estimate_cost(self) -> None:
        cost = estimate_cost("claude-sonnet-5", input_tokens=1_000_000, output_tokens=0)
        assert cost == pytest.approx(3.0)

    def test_unknown_model_priced_at_the_top_tier(self) -> None:
        # 宁可提前降级也不要超支
        unknown = estimate_cost("mystery", input_tokens=1_000_000, output_tokens=0)
        assert unknown == pytest.approx(15.0)

    def test_degrades_when_daily_budget_exceeded(self, tmp_path: Path) -> None:
        guard = BudgetGuard(daily_usd=1.0, monthly_usd=100.0, usage_dir=tmp_path)
        guard.record(
            UsageRecord(
                task_id="t",
                model_id="m",
                input_tokens=0,
                output_tokens=0,
                cost_usd=1.5,
                at=NOW,
            )
        )
        assert guard.state.degraded
        assert not guard.can_spend()
        assert "当日费用" in guard.state.degraded_reason

    def test_projected_cost_blocks_before_overspending(self, tmp_path: Path) -> None:
        # 一次调用就打穿余额的情况必须提前拦下
        guard = BudgetGuard(daily_usd=1.0, monthly_usd=100.0, usage_dir=tmp_path)
        assert not guard.can_spend(2.0)

    def test_budget_survives_restart(self, tmp_path: Path) -> None:
        # 否则反复重启就能绕过预算
        first = BudgetGuard(daily_usd=10.0, monthly_usd=100.0, usage_dir=tmp_path)
        first.record(
            UsageRecord(
                task_id="t", model_id="m", input_tokens=0, output_tokens=0, cost_usd=4.0, at=NOW
            )
        )
        second = BudgetGuard(daily_usd=10.0, monthly_usd=100.0, usage_dir=tmp_path)
        assert second.state.daily_usd == pytest.approx(4.0)

    def test_client_degrades_when_over_budget(self, tmp_path: Path) -> None:
        guard = BudgetGuard(daily_usd=0.0001, monthly_usd=100.0, usage_dir=tmp_path)
        guard.record(
            UsageRecord(
                task_id="t", model_id="m", input_tokens=0, output_tokens=0, cost_usd=1.0, at=NOW
            )
        )
        provider = StubProvider()
        client = LLMClient(mode="live", provider=provider, cache_dir=tmp_path, budget=guard)
        result = client.complete(make_call(), PositionJudgement)

        assert result.output is None
        assert "预算" in result.reason
        assert provider.calls == []

    def test_reset_degradation(self, tmp_path: Path) -> None:
        guard = BudgetGuard(daily_usd=1.0, monthly_usd=10.0, usage_dir=tmp_path)
        guard.record(
            UsageRecord(
                task_id="t", model_id="m", input_tokens=0, output_tokens=0, cost_usd=2.0, at=NOW
            )
        )
        guard.reset_degradation()
        assert not guard.state.degraded


class TestBackfill:
    """历史预计算。"""

    def test_backfill_populates_cache(self, tmp_path: Path) -> None:
        provider = StubProvider()
        client = LLMClient(mode="live", provider=provider, cache_dir=tmp_path)
        calls = [make_call({"materials": MATERIALS, "n": i}) for i in range(3)]

        done, cost = client.backfill(calls, PositionJudgement)
        assert done == 3
        assert cost > 0
        assert client.coverage(calls) == 1.0

    def test_backfill_is_resumable(self, tmp_path: Path) -> None:
        provider = StubProvider()
        client = LLMClient(mode="live", provider=provider, cache_dir=tmp_path)
        calls = [make_call({"n": i}) for i in range(3)]
        client.backfill(calls, PositionJudgement)

        before = len(provider.calls)
        client.backfill(calls, PositionJudgement)
        assert len(provider.calls) == before, "已算过的必须跳过"

    def test_backfill_requires_live_mode(self, tmp_path: Path) -> None:
        client = LLMClient(mode="replay", cache_dir=tmp_path)
        with pytest.raises(LLMError, match="live"):
            client.backfill([make_call()], PositionJudgement)

    def test_coverage_reports_partial(self, tmp_path: Path) -> None:
        client = LLMClient(mode="live", provider=StubProvider(), cache_dir=tmp_path)
        client.complete(make_call({"n": 0}), PositionJudgement)
        calls = [make_call({"n": 0}), make_call({"n": 1})]
        assert client.coverage(calls) == 0.5


class TestTasks:
    """三类任务。"""

    def test_position_judge_applies_bounded_influence(self, tmp_path: Path) -> None:
        client = LLMClient(mode="live", provider=StubProvider(), cache_dir=tmp_path)
        task = PositionJudgeTask(client, alpha=0.15, model_id="claude-sonnet-5")
        outcome = task.run(MAOTAI, base_score=0.60, materials=MATERIALS, as_of="T日")

        assert outcome.used_llm
        assert outcome.influence.final_score == pytest.approx(0.60 * (1 + 0.15 * 0.4))
        assert outcome.falsification() == ("跌破 MA20 则证伪",)
        assert outcome.conflicts() == ("动量与估值方向相反",)

    def test_position_judge_degrades_to_pure_quant(self, tmp_path: Path) -> None:
        client = LLMClient(mode="off")
        task = PositionJudgeTask(client, alpha=0.15, model_id="m")
        outcome = task.run(MAOTAI, base_score=0.60, materials=MATERIALS, as_of="T日")

        assert not outcome.used_llm
        assert outcome.influence.final_score == 0.60
        assert outcome.falsification() == ()

    def test_market_judge_only_reduces(self, tmp_path: Path) -> None:
        bullish = json.dumps(
            {
                "regime": "RISK_ON",
                "drivers": [{"statement": "情绪回暖", "evidence_ref": "m1"}],
                "exposure_adjustment": 0.9,
            }
        )
        client = LLMClient(mode="live", provider=StubProvider(bullish), cache_dir=tmp_path)
        task = MarketJudgeTask(client, alpha=0.15, model_id="claude-sonnet-5")
        outcome = task.run(base_exposure=0.6, materials=MATERIALS, as_of="T日")

        assert outcome.influence.final_score == 0.6

    def test_market_judge_reduces_on_risk(self, tmp_path: Path) -> None:
        bearish = json.dumps(
            {
                "regime": "RISK_OFF",
                "drivers": [{"statement": "风险事件密集", "evidence_ref": "m2"}],
                "exposure_adjustment": -0.6,
                "risk_level": "HIGH",
            }
        )
        client = LLMClient(mode="live", provider=StubProvider(bearish), cache_dir=tmp_path)
        task = MarketJudgeTask(client, alpha=0.20, model_id="m")
        outcome = task.run(base_exposure=0.6, materials=MATERIALS, as_of="T日")
        assert outcome.influence.final_score < 0.6

    def test_intel_classify_tags_llm_output(self, tmp_path: Path) -> None:
        # 红线 I-R3：LLM 产出必须可识别
        response = json.dumps({"event_type": "buyback", "sentiment": 0.4})
        client = LLMClient(mode="live", provider=StubProvider(response), cache_dir=tmp_path)
        task = IntelClassifyTask(client, model_id="claude-haiku-4-5-20251001")
        outcome = task.run(item_id="n1", title="公司拟回购股份")

        assert outcome.used_llm
        assert outcome.classifier_tag.startswith("llm:")

    def test_intel_classify_keeps_rule_when_insufficient(self, tmp_path: Path) -> None:
        response = json.dumps({"insufficient_evidence": True})
        client = LLMClient(mode="live", provider=StubProvider(response), cache_dir=tmp_path)
        task = IntelClassifyTask(client, model_id="m")
        outcome = task.run(item_id="n1", title="一句没有信息量的话")

        assert outcome.classification is None
        assert outcome.classifier_tag == "rule"

    def test_explain_falls_back_to_rules(self, tmp_path: Path) -> None:
        # 去掉 LLM 后信息一条不少，只是行文朴素
        task = ExplainTask(LLMClient(mode="off"), model_id="m")
        outcome = task.run(
            verdict="减持至目标权重",
            pillars={"①量化依据": ["打分下滑 27 个分位"], "④反面证据": ["股息率仍处高位"]},
            as_of="T日",
        )
        assert not outcome.llm_generated
        assert any("打分下滑" in line for line in outcome.as_lines())
        assert any("股息率" in line for line in outcome.as_lines())

    def test_explain_never_rewrites_the_verdict(self, tmp_path: Path) -> None:
        # 模型可能"顺手润色"结论，而结论必须与结构化结果逐字一致
        response = json.dumps({"verdict": "强烈建议清仓！", "narrative": ["打分下滑明显。"]})
        client = LLMClient(mode="live", provider=StubProvider(response), cache_dir=tmp_path)
        task = ExplainTask(client, model_id="m")
        outcome = task.run(
            verdict="减持至目标权重", pillars={"①量化依据": ["打分下滑"]}, as_of="T日"
        )
        assert outcome.verdict == "减持至目标权重"
        assert outcome.llm_generated
        assert "🤖" in outcome.as_lines()[0]


class TestPromptHygiene:
    """提示词卫生。"""

    def test_all_prompts_carry_the_closed_world_clause(self) -> None:
        from quantstock.llm.prompts import (  # noqa: PLC0415
            INTEL_CLASSIFY_SYSTEM,
            MARKET_JUDGE_SYSTEM,
            POSITION_JUDGE_SYSTEM,
        )

        for prompt in (INTEL_CLASSIFY_SYSTEM, POSITION_JUDGE_SYSTEM, MARKET_JUDGE_SYSTEM):
            assert "<materials>" in prompt
            assert "insufficient_evidence" in prompt
            assert "先验知识" in prompt

    def test_no_prompt_asks_for_predictions(self) -> None:
        # "这只股票会涨吗"正是被未来知识严重污染的问法
        from quantstock.llm.prompts import (  # noqa: PLC0415
            EXPLAIN_SYSTEM,
            INTEL_CLASSIFY_SYSTEM,
            MARKET_JUDGE_SYSTEM,
            POSITION_JUDGE_SYSTEM,
        )

        for prompt in (
            INTEL_CLASSIFY_SYSTEM,
            POSITION_JUDGE_SYSTEM,
            MARKET_JUDGE_SYSTEM,
            EXPLAIN_SYSTEM,
        ):
            assert "预测" in prompt, "每份提示词都要显式禁止预测"

    def test_materials_render_with_ids(self) -> None:
        from quantstock.llm.prompts import render_materials  # noqa: PLC0415

        rendered = render_materials(MATERIALS)
        assert "[m1]" in rendered
        assert "[m2]" in rendered

    def test_empty_materials_render(self) -> None:
        from quantstock.llm.prompts import render_materials  # noqa: PLC0415

        assert "无材料" in render_materials({})
