"""界面全链路测试：从空数据湖走到一份可执行的交易计划。

**这类测试的价值已经被证明过一次**。项目曾经有 1047 个通过的单元测试和 93%
的覆盖率，却完全没发现主链路是断的——每一层都被单独测过，但没有任何东西
把它们连起来跑。见 ``test_chain.py`` 的模块说明。

界面这一层同样有断链的风险，而且形态更隐蔽：每个路由单独返回 200，
但序列化器把 ``Decimal`` 转成了 float、四支柱少了一根、
执行页拿不到计划 ID——这些只有真的走一遍才会暴露。

所以这里做的是一次**完整的用户旅程**：

    造数据 → 更新行情 → 生成建议 → 读回计划 → 执行预检 → 逐单执行 → 回测

每一步都通过 HTTP 接口，用的是界面实际会走的那条路径。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from quantstock.config.models import RootConfig
from quantstock.config.settings import Secrets, Settings
from quantstock.services.data_service import CORE_UNIVERSE, DataService
from quantstock.web.app import create_app
from tests.unit.test_chain import write_market_data


@pytest.fixture
def seeded(tmp_path: Path) -> Settings:
    """准备好数据湖的配置。

    走的是 ``CsvSource`` 这条离线入口——测试禁止打真实网络，
    而合成的随机游走数据足以驱动整条链路。

    Args:
        tmp_path: 临时目录。

    Returns:
        Settings 实例。
    """
    from quantstock.data.sources.csv_source import CsvSource  # noqa: PLC0415 - 仅测试用

    config = RootConfig.model_validate(
        {
            "app": {"var_dir": str(tmp_path / "var")},
            "data": {"source_chain": ["csv"], "start_date": "2024-01-01"},
            "execution": {"broker": "manual"},
            "intel": {"enabled": False},
        }
    )
    settings = Settings(config=config, secrets=Secrets(), config_dir=tmp_path / "config")
    settings.config_dir.mkdir(parents=True, exist_ok=True)

    write_market_data(settings.var_dir / "csv")
    service = DataService(settings, source=CsvSource(settings.var_dir / "csv"))
    service.sync_instruments()
    service.update(CORE_UNIVERSE, start=dt.date(2024, 1, 1))
    return settings


@pytest.fixture
def client(seeded: Settings) -> Iterator[TestClient]:
    app = create_app(settings=seeded)
    with TestClient(app) as c:
        c.headers["X-Access-Token"] = app.state.app_state.access_token
        yield c


@pytest.fixture
def plan(client: TestClient) -> dict[str, Any]:
    """生成一份交易计划。

    Args:
        client: 测试客户端。

    Returns:
        建议响应。
    """
    response = client.post("/api/advisor/advise", json={"save": True})
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


class TestFullJourney:
    """一次完整的用户旅程。"""

    def test_data_status_reports_a_ready_lake(self, client: TestClient) -> None:
        body = client.get("/api/data/status").json()
        assert body["is_ready"] is True
        assert body["symbols"] > 0
        assert body["latest_date"]

    def test_bars_endpoint_feeds_the_candle_chart(self, client: TestClient) -> None:
        body = client.get("/api/data/bars", params={"symbol": "600519.SH", "limit": 120}).json()
        assert body["count"] > 0
        # 复权口径必须回给界面。不标口径的 K 线对不上看盘软件时，
        # 用户会以为是数据错了（红线 R4）
        assert body["adjust"] in {"hfq", "qfq", "none"}
        bar = body["bars"][0]
        assert set(bar) >= {"date", "open", "high", "low", "close", "volume"}

    def test_prices_are_strings_not_floats(self, client: TestClient) -> None:
        # JSON 数字是 IEEE 754 双精度。价格一旦变成 float，界面上就会出现
        # 1596.5200000000001，而用户看到的是钱（红线 R1）
        bar = client.get("/api/data/bars", params={"symbol": "600519.SH"}).json()["bars"][0]
        for field in ("open", "high", "low", "close"):
            assert isinstance(bar[field], str)
            Decimal(bar[field])  # 必须能无损还原

    def test_advise_produces_a_plan(self, plan: dict[str, Any]) -> None:
        assert plan["plan"]["plan_id"]
        assert plan["summary"]

    def test_every_intent_is_actionable(self, plan: dict[str, Any]) -> None:
        # "建议减仓"不可执行，"卖出 400 股，限价 1578~1596"才可以
        for intent in plan["plan"]["intents"]:
            assert intent["qty"] > 0
            assert Decimal(intent["price_low"]) <= Decimal(intent["price_high"])
            assert Decimal(intent["estimated_amount"]) > 0

    def test_every_intent_carries_four_pillars(self, plan: dict[str, Any]) -> None:
        # 四支柱是**强制**的：缺任何一根，这条建议在界面上就成了
        # "系统说买"而没有可核查的理由
        for intent in plan["plan"]["intents"]:
            r = intent["rationale"]
            assert r["quant_evidence"], "缺支柱① 量化依据"
            assert r["technical"]["statements"], "缺支柱② 持仓与技术分析"
            # 支柱③ 无情报时必须明写说明，不能留空——
            # 留空让人分不清"没查"和"查了没有"
            assert r["intel_evidence"] or r["intel_absent_note"], "缺支柱③ 情报证据"
            assert r["counter_evidence"] or r["falsification"], "缺支柱④ 反面证据"
            assert r["is_complete"] is True

    def test_intel_evidence_always_carries_source_and_time(self, plan: dict[str, Any]) -> None:
        # 红线 I-R4：不可复述无出处的内容。没有链接的"情报"是不可核实的传闻
        for intent in plan["plan"]["intents"]:
            for item in intent["rationale"]["intel_evidence"]:
                assert item["url"].strip()
                assert item["published_at"]

    def test_traceability_fields_are_present(self, plan: dict[str, Any]) -> None:
        # 红线 R6：没有这三样，审计页就没法复现"当天为什么给出这个建议"
        p = plan["plan"]
        assert p["data_fingerprint"]
        assert p["param_hash"]
        assert p["strategy_versions"]

    def test_plan_is_reloadable_by_date(self, client: TestClient, plan: dict[str, Any]) -> None:
        trade_date = plan["plan"]["trade_date"]
        assert trade_date in client.get("/api/advisor/dates").json()["dates"]

        reloaded = client.get(f"/api/advisor/plan/{trade_date}").json()
        assert reloaded["plan_id"] == plan["plan"]["plan_id"]
        assert len(reloaded["intents"]) == len(plan["plan"]["intents"])

    def test_missing_plan_is_404_not_500(self, client: TestClient) -> None:
        assert client.get("/api/advisor/plan/1999-01-04").status_code == 404

    def test_preview_then_execute(self, client: TestClient, plan: dict[str, Any]) -> None:
        p = plan["plan"]
        if not p["intents"]:
            pytest.skip("本次未产生建议")

        prices = {i["symbol"]: i["price_high"] for i in p["intents"]}
        preview = client.post(
            "/api/execution/preview",
            json={"trade_date": p["trade_date"], "plan_id": p["plan_id"], "prices": prices},
        ).json()
        assert len(preview["items"]) == len(p["intents"])
        assert preview["broker"] == "manual"
        for item in preview["items"]:
            assert Decimal(item["limit_price"]) > 0

        report = client.post(
            "/api/execution/execute",
            json={
                "trade_date": p["trade_date"],
                "plan_id": p["plan_id"],
                "prices": prices,
                "decisions": [
                    {"intent_id": i["intent_id"], "accepted": True} for i in p["intents"]
                ],
            },
        )
        assert report.status_code == 200, report.text
        body = report.json()
        assert body["confirmed_by"] == "ui"
        assert body["submitted"] + body["skipped"] == len(p["intents"])

    def test_omitted_decisions_are_skipped_not_executed(
        self, client: TestClient, plan: dict[str, Any]
    ) -> None:
        # **默认执行会把"没来得及看"变成"下单了"**，这个方向的错误不可逆。
        # 一条决定都不给时，必须一笔都不提交
        p = plan["plan"]
        if not p["intents"]:
            pytest.skip("本次未产生建议")

        body = client.post(
            "/api/execution/execute",
            json={
                "trade_date": p["trade_date"],
                "plan_id": p["plan_id"],
                "prices": {i["symbol"]: i["price_high"] for i in p["intents"]},
                "decisions": [],
            },
        ).json()
        assert body["submitted"] == 0
        assert body["skipped"] == len(p["intents"])

    def test_backtest_runs_and_records_a_trial(self, client: TestClient) -> None:
        before = client.get("/api/backtest/trials").json()["count"]
        response = client.post(
            "/api/backtest/run",
            json={"start": "2025-01-02", "end": "2026-07-24", "rebalance_days": 5},
        )
        assert response.status_code == 200, response.text
        report = response.json()

        assert report["trading_days"] > 0
        assert report["trial_id"]
        assert isinstance(report["explain"], str)
        # 每次尝试都必须落账，包括不好看的。只留最优结果会让 DSR 系统性偏乐观
        assert client.get("/api/backtest/trials").json()["count"] == before + 1

    def test_backtest_reports_both_twr_and_mwr(self, client: TestClient) -> None:
        # 只看一个会得出相反结论：策略很好但在高点加了仓，
        # TWR 漂亮而 MWR 是亏的
        stats = client.post(
            "/api/backtest/run", json={"start": "2025-01-02", "end": "2026-07-24"}
        ).json()["stats"]
        assert "twr" in stats
        assert "mwr" in stats

    def test_random_walk_does_not_produce_free_money(self, client: TestClient) -> None:
        # 合成数据是纯随机游走。动量策略在噪声上本来就不该赚钱——
        # 若这里跑出了高 Sharpe，说明链路某处泄漏了未来信息
        stats = client.post(
            "/api/backtest/run", json={"start": "2025-01-02", "end": "2026-07-24"}
        ).json()["stats"]
        assert stats["sharpe"] < 2.0, f"随机游走上跑出 Sharpe {stats['sharpe']}，疑似未来函数"

    def test_admission_gate_becomes_available_after_a_trial(self, client: TestClient) -> None:
        client.post("/api/backtest/run", json={"start": "2025-01-02", "end": "2026-07-24"})
        verdict = client.get("/api/backtest/admission").json()

        assert verdict["available"] is True
        assert 0.0 <= verdict["dsr"] <= 1.0
        # 一次试验就想进实盘候选池是不可能的：PBO 还没算，参数高原也没验
        assert verdict["admitted"] is False
        assert verdict["reasons"]


class TestEventsDuringJourney:
    """全链路过程中的事件推送。"""

    def test_advise_publishes_task_events(self, client: TestClient, seeded: Settings) -> None:
        del seeded
        hub = client.app.state.app_state.events  # type: ignore[attr-defined]
        before = hub.last_seq

        client.post("/api/advisor/advise", json={"save": True})

        kinds = [e.kind for e in hub.replay(frozenset({"tasks"}), since=before)]
        assert "progress" in kinds
        assert "done" in kinds


class TestExecutionFeedsReview:
    """执行 → 复盘的闭环。

    **这是执行报告落盘的目的**。报告此前只存在于一次调用的返回值里，
    进程一退就没了——于是"计划 8 笔、执行 5 笔、跳过的 3 笔各是什么原因"
    这类问题事后完全无法回答，而这正是半自动系统里最值得回答的一类。
    """

    def _execute(self, client: TestClient, plan: dict[str, Any], *, accept: bool) -> dict[str, Any]:
        """执行一份计划。

        Args:
            client: 测试客户端。
            plan: 建议响应。
            accept: 全部接受还是全部跳过。

        Returns:
            执行报告。
        """
        p = plan["plan"]
        body: dict[str, Any] = client.post(
            "/api/execution/execute",
            json={
                "trade_date": p["trade_date"],
                "plan_id": p["plan_id"],
                "prices": {i["symbol"]: i["price_high"] for i in p["intents"]},
                "decisions": [
                    {
                        "intent_id": i["intent_id"],
                        "accepted": accept,
                        "skip_reason": None if accept else "bad_timing",
                    }
                    for i in p["intents"]
                ],
            },
        ).json()
        return body

    def test_execution_shows_up_in_review(self, client: TestClient, plan: dict[str, Any]) -> None:
        if not plan["plan"]["intents"]:
            pytest.skip("本次未产生建议")

        assert client.get("/api/review/dates").json()["dates"] == []
        self._execute(client, plan, accept=True)

        dates = client.get("/api/review/dates").json()["dates"]
        assert plan["plan"]["trade_date"] in dates

        deviation = client.get(f"/api/review/deviation/{plan['plan']['trade_date']}").json()
        assert deviation["available"] is True
        assert deviation["planned"] == len(plan["plan"]["intents"])

    def test_skip_reasons_survive_to_review(self, client: TestClient, plan: dict[str, Any]) -> None:
        # 跳过原因是复盘的原料。只存成交记录的话，
        # "人工干预到底是帮忙还是添乱"永远算不出来
        if not plan["plan"]["intents"]:
            pytest.skip("本次未产生建议")

        self._execute(client, plan, accept=False)

        deviation = client.get(f"/api/review/deviation/{plan['plan']['trade_date']}").json()
        assert deviation["skipped"] == len(plan["plan"]["intents"])
        assert deviation["by_reason"].get("bad_timing") == len(plan["plan"]["intents"])

    def test_review_summary_counts_the_execution(
        self, client: TestClient, plan: dict[str, Any]
    ) -> None:
        if not plan["plan"]["intents"]:
            pytest.skip("本次未产生建议")

        self._execute(client, plan, accept=True)
        summary = client.get("/api/review/summary").json()

        assert summary["plans"] == 1
        assert summary["total_executed"] > 0
        # 样本远不足十次，必须仍然拒绝给结论
        assert summary["has_enough_samples"] is False


class TestAbortedExecution:
    """被硬闸中止的执行。

    绝对金额硬闸（A10）**中止整个计划而不是跳过那一笔**：单笔超限通常意味着
    计算基数出错，此时其余单笔同样不可信。这条行为是对的，但它有两个
    容易被忽略的副作用，下面各钉一条。
    """

    def _tiny_limit_client(self, seeded: Settings) -> TestClient:
        """构造一个硬闸阈值极低的客户端。

        Args:
            seeded: 已备好数据的配置。

        Returns:
            测试客户端。
        """
        seeded.config.risk.hard_limits.max_single_order_amount = Decimal("1")
        app = create_app(settings=seeded)
        client = TestClient(app)
        client.headers["X-Access-Token"] = app.state.app_state.access_token
        return client

    @staticmethod
    def _run(client: TestClient, plan: dict[str, Any]) -> dict[str, Any]:
        """全部接受地执行一份计划。

        Args:
            client: 测试客户端。
            plan: 交易计划。

        Returns:
            执行报告。
        """
        body: dict[str, Any] = client.post(
            "/api/execution/execute",
            json={
                "trade_date": plan["trade_date"],
                "plan_id": plan["plan_id"],
                "prices": {i["symbol"]: i["price_high"] for i in plan["intents"]},
                "decisions": [
                    {"intent_id": i["intent_id"], "accepted": True} for i in plan["intents"]
                ],
            },
        ).json()
        return body

    def test_abort_is_reported_not_silent(self, seeded: Settings) -> None:
        # 只回「提交 0 笔」而不说为什么，用户会以为系统坏了
        client = self._tiny_limit_client(seeded)
        plan = client.post("/api/advisor/advise", json={"save": True}).json()["plan"]
        if not plan["intents"]:
            pytest.skip("本次未产生建议")

        body = self._run(client, plan)

        assert body["aborted"] is True
        assert "硬闸" in body["abort_reason"]
        assert body["submitted"] == 0

    def test_aborted_plan_is_not_marked_confirmed(self, seeded: Settings) -> None:
        # 硬闸中止**不抛异常**而是返回 aborted=True，所以"先执行再标记确认"
        # 挡不住它——一份被拦下、一笔都没发出的计划会被标成"已人工确认并执行"，
        # 审计流水上再也分不清它到底有没有真的下过单
        client = self._tiny_limit_client(seeded)
        plan = client.post("/api/advisor/advise", json={"save": True}).json()["plan"]
        if not plan["intents"]:
            pytest.skip("本次未产生建议")

        self._run(client, plan)

        reloaded = client.get(f"/api/advisor/plan/{plan['trade_date']}").json()
        assert reloaded["is_confirmed"] is False
        assert reloaded["confirmed_by"] == ""

    def test_aborted_execution_is_still_recorded(self, seeded: Settings) -> None:
        # 被拦下来这件事本身就要留痕：事后要能查到"那天试过，被硬闸挡了"
        client = self._tiny_limit_client(seeded)
        plan = client.post("/api/advisor/advise", json={"save": True}).json()["plan"]
        if not plan["intents"]:
            pytest.skip("本次未产生建议")

        self._run(client, plan)

        deviation = client.get(f"/api/review/deviation/{plan['trade_date']}").json()
        assert deviation["available"] is True
        assert deviation["aborted"] is True


class TestAuditAndPortfolio:
    """P15 审计与 P9 组合。"""

    def test_reproduction_is_the_evidence_for_r6(
        self, client: TestClient, plan: dict[str, Any]
    ) -> None:
        # 把指纹存下来只证明"当时算过"。重算一遍并比对，才证明"现在还能算出
        # 同样的结果"——这才是红线 R6 真正被满足的证据
        date = plan["plan"]["trade_date"]
        body = client.post(f"/api/audit/reproduce/{date}").json()

        assert body["verdict"] == "identical", body["explain"]
        assert body["fingerprint"]["match"] is True
        assert body["param_hash"]["match"] is True
        assert body["intents"]["match"] is True

    def test_reproduction_does_not_overwrite_the_archive(
        self, client: TestClient, plan: dict[str, Any]
    ) -> None:
        # 复现若覆盖了存档，就再也没有"当时"可比了
        date = plan["plan"]["trade_date"]
        original_id = plan["plan"]["plan_id"]
        client.post(f"/api/audit/reproduce/{date}")
        assert client.get(f"/api/advisor/plan/{date}").json()["plan_id"] == original_id

    def test_audit_chain_links_intent_to_orders(
        self, client: TestClient, plan: dict[str, Any]
    ) -> None:
        p = plan["plan"]
        if not p["intents"]:
            pytest.skip("本次未产生建议")

        client.post(
            "/api/execution/execute",
            json={
                "trade_date": p["trade_date"],
                "plan_id": p["plan_id"],
                "prices": {i["symbol"]: i["price_high"] for i in p["intents"]},
                "decisions": [
                    {"intent_id": i["intent_id"], "accepted": False, "skip_reason": "bad_timing"}
                    for i in p["intents"]
                ],
            },
        )

        chain = client.get(f"/api/audit/plan/{p['trade_date']}").json()["chain"]
        assert len(chain) == len(p["intents"])
        # intent_id 是贯穿建议 → 订单 → 成交的锚点（红线 R6）
        assert all(link["orders"] for link in chain)
        assert all("跳过" in link["outcome"] for link in chain)

    def test_missing_audit_date_is_404(self, client: TestClient) -> None:
        assert client.get("/api/audit/plan/1999-01-04").status_code == 404

    def test_portfolio_weights_reflect_the_plan(
        self, client: TestClient, plan: dict[str, Any]
    ) -> None:
        body = client.get("/api/portfolio/weights").json()
        assert body["plan_id"] == plan["plan"]["plan_id"]
        if plan["plan"]["intents"]:
            assert body["rows"]

    def test_portfolio_constraints_list_breaches_individually(self, client: TestClient) -> None:
        # 只给一个"不合规"的总判定没法行动。要知道是哪一条、超了多少
        body = client.get("/api/portfolio/constraints").json()
        assert "satisfied" in body
        assert isinstance(body["breaches"], list)
        assert body["limits"]["max_single_position"] > 0

    def test_portfolio_flags_single_position_breach(self, client: TestClient) -> None:
        # 把全部资金押在一只标的上，必须被 B01 逮住
        client.post("/api/account/deposit", json={"amount": "100000"})
        client.post(
            "/api/account/trade",
            json={"symbol": "600519.SH", "side": "buy", "qty": 60, "price": "1000"},
        )
        body = client.get("/api/portfolio/constraints").json()

        assert body["satisfied"] is False
        assert any(b["rule_id"] == "B01" for b in body["breaches"])
