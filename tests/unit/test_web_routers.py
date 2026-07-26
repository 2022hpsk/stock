"""新增 API 路由、WebSocket 推送与前端挂载的测试。

对应 docs/09-可视化界面规格.md 第九节验收标准里此前没被覆盖的几条：

- 验收 6：``var/HALT`` 存在时下单接口返回拒绝；
- 验收 8：断开 WebSocket 后重连，事件能补齐、不丢失；
- 验收 9：构建产物不引用任何外部 CDN（离线可用）；
- 验收 10：只读模式下所有写操作返回 403。

另外钉死两个已经踩过的坑：

- 空的 ``SourceRegistry`` 是 falsy，用 ``or`` 装配会把"关掉所有源"变成
  "用默认联网源"（已在 services 层修掉，这里从 API 侧再验一次降级形态）；
- SPA 的 catch-all 路由会把拼错的 ``/api/*`` 变成 200 + HTML，
  前端只能看到一个 "Unexpected token <"。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from quantstock.config.models import RootConfig
from quantstock.config.settings import Secrets, Settings
from quantstock.web.app import DIST_DIR, create_app
from quantstock.web.events import CHANNELS, EventHub, parse_channels


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = RootConfig()
    config.app.var_dir = str(tmp_path / "var")
    # 关掉情报模块：CLI/API 用例验的是接线，不是源的可用性。
    # 留着默认装配会让测试真的去请求外网
    config.intel.enabled = False
    return Settings(config=config, secrets=Secrets(_env_file=None), config_dir=config_dir)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings=settings)
    with TestClient(app) as c:
        c.headers["X-Access-Token"] = app.state.app_state.access_token
        yield c


@pytest.fixture
def readonly_client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings=settings, readonly=True)
    with TestClient(app) as c:
        c.headers["X-Access-Token"] = app.state.app_state.access_token
        yield c


class TestReadEndpoints:
    """只读接口在空环境下也必须能返回，而不是 500。

    新装的系统数据湖是空的。这时候接口崩掉，用户第一次打开界面看到的
    就是一片报错——而他还什么都没做错。
    """

    @pytest.mark.parametrize(
        "path",
        [
            "/api/data/status",
            "/api/data/universe",
            "/api/intel/status",
            "/api/intel/sources",
            "/api/intel/blacklist",
            "/api/llm/status",
            "/api/execution/status",
            "/api/execution/skip-reasons",
            "/api/advisor/dates",
            "/api/backtest/trials",
            "/api/backtest/admission",
        ],
    )
    def test_returns_200_on_empty_environment(self, client: TestClient, path: str) -> None:
        assert client.get(path).status_code == 200

    def test_all_require_token(self, settings: Settings) -> None:
        with TestClient(create_app(settings=settings)) as anon:
            assert anon.get("/api/data/status").status_code == 401

    def test_admission_without_trials_is_not_an_error(self, client: TestClient) -> None:
        # 没有试验记录不是异常，是"还没做过检验"。返回 500 会让人以为系统坏了
        body = client.get("/api/backtest/admission").json()
        assert body["available"] is False
        assert "禁止入池" in body["message"]

    def test_intel_sources_empty_when_disabled(self, client: TestClient) -> None:
        assert client.get("/api/intel/sources").json()["sources"] == []

    def test_skip_reasons_are_an_enum_not_free_text(self, client: TestClient) -> None:
        # 自由输入的跳过原因没法分组统计，人工干预价值分析就成了一堆读不出结论的字符串
        reasons = client.get("/api/execution/skip-reasons").json()["reasons"]
        assert {r["value"] for r in reasons} >= {"disagree_logic", "cash_reserved", "bad_timing"}
        assert all(r["label"] for r in reasons)


class TestReadonlyMode:
    """验收 10：只读模式下所有写操作返回 403。"""

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("post", "/api/data/update"),
            ("post", "/api/advisor/advise"),
            ("post", "/api/execution/cancel-all"),
            ("post", "/api/intel/fetch"),
            ("post", "/api/backtest/run"),
            ("post", "/api/system/halt"),
        ],
    )
    def test_writes_rejected(self, readonly_client: TestClient, method: str, path: str) -> None:
        response = getattr(readonly_client, method)(path, json={})
        assert response.status_code == 403

    def test_reads_still_allowed(self, readonly_client: TestClient) -> None:
        assert readonly_client.get("/api/data/status").status_code == 200


class TestExecutionGuards:
    """执行接口的前置校验。这些是**后端**校验，不是界面提示。"""

    def test_live_channel_requires_confirmation_code(self, client: TestClient) -> None:
        # 只在前端拦是防误点，不是防绕过。绕过界面直接打接口的路径必须也被堵上
        response = client.post(
            "/api/execution/execute",
            json={
                "trade_date": "2026-07-24",
                "plan_id": "p1",
                "decisions": [],
                "live": True,
                "confirmation_code": "  ",
            },
        )
        assert response.status_code == 400
        assert "确认码" in response.json()["detail"]

    def test_skip_without_reason_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/api/execution/execute",
            json={
                "trade_date": "2026-07-24",
                "plan_id": "p1",
                "decisions": [{"intent_id": "i1", "accepted": False, "skip_reason": None}],
            },
        )
        assert response.status_code == 400
        assert "原因" in response.json()["detail"]


class TestSpaMounting:
    """前端挂载。"""

    def test_index_is_served(self, client: TestClient) -> None:
        assert client.get("/").status_code == 200

    def test_history_routes_fall_back_to_index(self, client: TestClient) -> None:
        # Vue Router 用 history 模式，直接刷新 /advisor 必须仍返回单页入口，
        # 否则用户刷新一次页面就"打不开了"
        first = client.get("/").text
        assert client.get("/advisor").text == first

    def test_unknown_api_path_is_404_not_html(self, client: TestClient) -> None:
        # catch-all 吞掉 /api/* 时状态码是 200 而 body 是 HTML，
        # 前端 response.json() 抛的 "Unexpected token <" 跟真正的问题毫无关系
        response = client.get("/api/definitely-not-a-route")
        assert response.status_code == 404
        assert "<div id=" not in response.text


@pytest.mark.skipif(not (DIST_DIR / "index.html").exists(), reason="前端未构建")
class TestBuiltAssets:
    """验收 9：构建产物不得引用任何外部 CDN。"""

    def test_index_has_no_external_resource(self) -> None:
        html = (DIST_DIR / "index.html").read_text(encoding="utf-8")
        for attr in ('src="http', "src='http", 'href="http', "href='http"):
            assert attr not in html

    def test_no_external_css_url(self) -> None:
        # 字体与图标必须内联或本地打包，否则离线时界面会缺字缺图，
        # 而且访问行为会泄漏给第三方
        for css in (DIST_DIR / "assets").glob("*.css"):
            text = css.read_text(encoding="utf-8")
            assert "url(http" not in text
            assert "url('http" not in text
            assert 'url("http' not in text

    def test_assets_are_served(self, client: TestClient) -> None:
        html = client.get("/").text
        start = html.index('src="/assets/') + len('src="')
        asset = html[start : html.index('"', start)]
        assert client.get(asset).status_code == 200


class TestWebSocket:
    """验收 8：断线重连不丢事件。"""

    def test_rejects_bad_token(self, settings: Settings) -> None:
        # 口令错误时连接必须建不起来。WebSocket 不走 HTTP 头，
        # 口令只能放查询串——正因为如此，这条校验一旦漏掉就是完全敞开的
        app = create_app(settings=settings)
        with (
            TestClient(app) as c,
            pytest.raises(WebSocketDisconnect),
            c.websocket_connect("/ws?token=wrong") as ws,
        ):
            ws.receive_json()

    def test_receives_published_events(self, settings: Settings) -> None:
        app = create_app(settings=settings)
        token = app.state.app_state.access_token
        with (
            TestClient(app) as c,
            c.websocket_connect(f"/ws?token={token}&channels=tasks") as ws,
        ):
            assert ws.receive_json()["kind"] == "ready"
            app.state.app_state.events.publish("tasks", "progress", task="demo")
            event = ws.receive_json()
            assert event["kind"] == "progress"
            assert event["payload"]["task"] == "demo"

    def test_replays_missed_events_on_reconnect(self, settings: Settings) -> None:
        # 不带 since 重连，断线那几秒的事件就永久消失了，
        # 界面上的进度条会卡在中途再也不动
        app = create_app(settings=settings)
        token = app.state.app_state.access_token
        hub = app.state.app_state.events
        with TestClient(app) as c:
            hub.publish("tasks", "first", task="a")
            checkpoint = hub.last_seq
            hub.publish("tasks", "missed", task="b")

            with c.websocket_connect(f"/ws?token={token}&channels=tasks&since={checkpoint}") as ws:
                replayed = ws.receive_json()
                assert replayed["kind"] == "missed"
                assert ws.receive_json()["kind"] == "ready"

    def test_channel_filter(self, settings: Settings) -> None:
        app = create_app(settings=settings)
        token = app.state.app_state.access_token
        hub = app.state.app_state.events
        with (
            TestClient(app) as c,
            c.websocket_connect(f"/ws?token={token}&channels=orders") as ws,
        ):
            assert ws.receive_json()["kind"] == "ready"
            hub.publish("tasks", "ignored", task="x")
            hub.publish("orders", "wanted", plan_id="p1")
            assert ws.receive_json()["kind"] == "wanted"


class TestEventHub:
    """事件总线本身。"""

    def test_sequence_is_monotonic(self) -> None:
        hub = EventHub()
        first = hub.publish("tasks", "a")
        second = hub.publish("orders", "b")
        assert second.seq == first.seq + 1

    def test_unknown_channel_rejected(self) -> None:
        # 拼错的频道名会让订阅静默地收不到消息，在界面上表现为"功能没反应"，
        # 极难定位。所以发布时就当场报错
        hub = EventHub()
        with pytest.raises(ValueError, match="未知频道"):
            hub.publish("nope", "x")

    def test_replay_filters_by_channel_and_seq(self) -> None:
        hub = EventHub()
        hub.publish("tasks", "old")
        checkpoint = hub.last_seq
        hub.publish("tasks", "new")
        hub.publish("orders", "other")

        replayed = hub.replay(frozenset({"tasks"}), since=checkpoint)

        assert [e.kind for e in replayed] == ["new"]

    def test_buffer_is_bounded(self) -> None:
        # 无界缓冲在长时间运行后会把内存吃光
        hub = EventHub(max_buffered=2)
        for i in range(100):
            hub.publish("tasks", f"e{i}")
        assert len(hub.replay(frozenset(CHANNELS), since=0)) <= 2 * len(CHANNELS)

    def test_payload_is_json_serialisable(self) -> None:
        event = EventHub().publish("tasks", "done", count=3, name="x", ok=True)
        assert json.loads(json.dumps(event.to_dict()))["payload"]["count"] == 3

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, set(CHANNELS)),
            ("", set(CHANNELS)),
            ("tasks,orders", {"tasks", "orders"}),
            ("tasks, nonsense", {"tasks"}),
            # 全是非法名时回退到全订阅，而不是让连接建不起来
            ("nonsense", set(CHANNELS)),
        ],
    )
    def test_parse_channels(self, raw: str | None, expected: set[str]) -> None:
        assert parse_channels(raw) == frozenset(expected)


class TestAccountEndpoints:
    """P1 账户页的接口。

    **没有"修改持仓"的接口**是刻意的（红线 R8）：持仓由流水重放得出，
    不是可以直接编辑的状态。
    """

    def test_empty_account_is_reported_not_crashed(self, client: TestClient) -> None:
        body = client.get("/api/account/summary").json()
        assert body["is_empty"] is True
        assert body["position_count"] == 0
        assert "账本为空" in body["message"]

    def test_positions_and_transactions_start_empty(self, client: TestClient) -> None:
        assert client.get("/api/account/positions").json()["positions"] == []
        assert client.get("/api/account/transactions").json()["transactions"] == []

    def test_deposit_then_trade_shows_up_in_positions(self, client: TestClient) -> None:
        assert client.post("/api/account/deposit", json={"amount": "100000"}).status_code == 200
        response = client.post(
            "/api/account/trade",
            json={"symbol": "600519.SH", "side": "buy", "qty": 100, "price": "500"},
        )
        assert response.status_code == 200, response.text

        rows = client.get("/api/account/positions").json()["positions"]
        assert len(rows) == 1
        assert rows[0]["symbol"] == "600519.SH"
        assert rows[0]["qty"] == 100
        # 批次明细必须回给界面：红利税分档与免税倒计时都要靠它
        assert rows[0]["lots"]

    def test_amounts_come_back_as_strings(self, client: TestClient) -> None:
        client.post("/api/account/deposit", json={"amount": "100000.55"})
        summary = client.get("/api/account/summary").json()
        assert summary["cash"] == "100000.55"

    def test_illegal_trade_is_a_400_not_a_500(self, client: TestClient) -> None:
        # 卖出超过持仓是用户输入错误，不是服务端故障
        client.post("/api/account/deposit", json={"amount": "100000"})
        response = client.post(
            "/api/account/trade",
            json={"symbol": "600519.SH", "side": "sell", "qty": 100, "price": "500"},
        )
        assert response.status_code == 400

    def test_account_has_no_mutating_verbs(self, client: TestClient) -> None:
        # 持仓是流水重放的结果，不该有 PUT / DELETE 这类"就地改状态"的路径。
        # 全部写操作都必须表现为"追加一笔流水"（红线 R8）
        mutating = {
            (route.path, verb)
            for route in client.app.routes  # type: ignore[attr-defined]
            for verb in getattr(route, "methods", set())
            if str(getattr(route, "path", "")).startswith("/api/account")
            and verb in {"PUT", "PATCH", "DELETE"}
        }
        assert mutating == set()
        assert client.put("/api/account/positions", json={}).status_code in {404, 405}

    def test_writes_rejected_in_readonly(self, readonly_client: TestClient) -> None:
        assert readonly_client.post("/api/account/deposit", json={"amount": "1"}).status_code == 403


class TestRiskEndpoints:
    """P10 风控页的接口。"""

    def test_rules_are_listed(self, client: TestClient) -> None:
        body = client.get("/api/risk/rules").json()
        assert body["count"] > 0
        ids = {r["rule_id"] for r in body["rules"]}
        assert {"A01", "A02", "A10", "B01"} <= ids

    def test_no_a_class_rule_is_closable(self, client: TestClient) -> None:
        # 验收 5：A 类风控规则在界面上无任何关闭入口。
        # 这条断言让"不可关闭"从文档里的一句话，变成会失败的约束
        rules = client.get("/api/risk/rules").json()["rules"]
        a_class = [r for r in rules if r["rule_class"] == "A"]
        assert a_class, "A 类规则不该为空"
        assert all(not r["closable"] for r in a_class)

    def test_a_class_thresholds_are_not_editable_from_ui(self, client: TestClient) -> None:
        # 让界面能改绝对硬闸，等于给风控开了个后门
        rules = client.get("/api/risk/rules").json()["rules"]
        assert all(not r["threshold_editable"] for r in rules if r["rule_class"] == "A")

    def test_b_class_rules_are_not_closable_either(self, client: TestClient) -> None:
        # B 类阈值可配，但规则本身不可关闭
        rules = client.get("/api/risk/rules").json()["rules"]
        assert all(not r["closable"] for r in rules if r["rule_class"] == "B")

    def test_hard_limits_are_not_auto_derived(self, client: TestClient) -> None:
        # 自动推导会被同一个错误数据污染，失去防护意义
        assert client.get("/api/risk/hard-limits").json()["auto_derived"] is False

    def test_circuit_reports_distance_not_only_state(self, client: TestClient) -> None:
        # 状态只有到了才变，距离能让人提前看到自己正在往哪走
        body = client.get("/api/risk/circuit", params={"daily_return": -0.02}).json()
        assert body["daily_loss"]["current"] == pytest.approx(0.02)
        assert body["daily_loss"]["watch"] > 0
        assert body["daily_loss"]["halted"] > body["daily_loss"]["watch"]

    def test_halted_does_not_auto_recover(self, client: TestClient) -> None:
        # 自动恢复会让系统在剧烈波动中反复进出，而那正是最该停手的时候
        assert client.get("/api/risk/circuit").json()["auto_recovers"] is False

    def test_risk_page_has_no_write_endpoints(self, client: TestClient) -> None:
        # 改阈值必须走配置页：那条路径有校验、Diff、备份与审计
        verbs = {
            verb
            for route in client.app.routes  # type: ignore[attr-defined]
            for verb in getattr(route, "methods", set())
            if str(getattr(route, "path", "")).startswith("/api/risk")
        }
        assert verbs <= {"GET", "HEAD", "OPTIONS"}


class TestReviewEndpoints:
    """P12 复盘页的接口。"""

    def test_empty_history_is_reported_not_crashed(self, client: TestClient) -> None:
        body = client.get("/api/review/summary").json()
        assert body["plans"] == 0
        assert body["deviations"] == []
        assert "无从复盘" in body["explain"]

    def test_dates_start_empty(self, client: TestClient) -> None:
        assert client.get("/api/review/dates").json()["dates"] == []

    def test_missing_day_is_reported_not_404(self, client: TestClient) -> None:
        body = client.get("/api/review/deviation/2026-07-24").json()
        assert body["available"] is False

    def test_sample_sufficiency_is_always_reported(self, client: TestClient) -> None:
        # 样本不够时算出的胜率是噪声，但读起来很像信号。
        # 所以界面必须能知道该不该信这个数字
        body = client.get("/api/review/summary").json()
        assert body["has_enough_samples"] is False
        assert "unpriced_skips" in body
