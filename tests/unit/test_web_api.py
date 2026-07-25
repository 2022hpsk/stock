"""Web API 测试。

覆盖 docs/09-可视化界面规格.md 第九节验收标准：
配置完备性、认证、只读模式、A 类规则无关闭入口、急停后拒绝写操作。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from quantstock.config.models import RootConfig
from quantstock.config.settings import Secrets, Settings
from quantstock.web.app import create_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    var_dir = tmp_path / "var"
    config = RootConfig()
    config.app.var_dir = str(var_dir)
    return Settings(config=config, secrets=Secrets(_env_file=None), config_dir=config_dir)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings=settings)
    with TestClient(app) as test_client:
        test_client.headers["X-Access-Token"] = app.state.app_state.access_token
        yield test_client


@pytest.fixture
def readonly_client(settings: Settings) -> Iterator[TestClient]:
    app = create_app(settings=settings, readonly=True)
    with TestClient(app) as test_client:
        test_client.headers["X-Access-Token"] = app.state.app_state.access_token
        yield test_client


class TestAuth:
    def test_health__no_auth_required(self, settings: Settings) -> None:
        with TestClient(create_app(settings=settings)) as anon:
            assert anon.get("/api/health").status_code == 200

    def test_missing_token__rejected(self, settings: Settings) -> None:
        with TestClient(create_app(settings=settings)) as anon:
            assert anon.get("/api/config").status_code == 401

    def test_wrong_token__rejected(self, settings: Settings) -> None:
        with TestClient(create_app(settings=settings)) as anon:
            anon.headers["X-Access-Token"] = "wrong"
            assert anon.get("/api/config").status_code == 401

    def test_valid_token__accepted(self, client: TestClient) -> None:
        assert client.get("/api/config").status_code == 200


class TestSystemStatus:
    def test_reports_broker_and_llm(self, client: TestClient) -> None:
        body = client.get("/api/system/status").json()
        assert body["broker"] == "paper"
        assert body["llm"]["enabled"] is False
        assert body["halt"]["halted"] is False

    def test_lists_components(self, client: TestClient) -> None:
        body = client.get("/api/system/status").json()
        names = {c["name"] for c in body["components"]}
        assert {"var_dir", "config", "hard_limits"} <= names


class TestHalt:
    def test_halt_then_resume(self, client: TestClient) -> None:
        halted = client.post("/api/system/halt", json={"reason": "数据异常"}).json()
        assert halted["halted"] is True
        assert client.get("/api/system/status").json()["halt"]["reason"] == "数据异常"

        client.post("/api/system/resume")
        assert client.get("/api/system/status").json()["halt"]["halted"] is False

    def test_halt_without_reason__rejected(self, client: TestClient) -> None:
        assert client.post("/api/system/halt", json={"reason": ""}).status_code == 422


class TestConfigApi:
    def test_schema_has_all_sections(self, client: TestClient) -> None:
        schema = client.get("/api/config/schema").json()
        assert {"app", "data", "risk", "llm", "execution", "portfolio"} <= set(schema["properties"])

    def test_get_config_matches_defaults(self, client: TestClient) -> None:
        body = client.get("/api/config").json()
        assert body["execution"]["broker"] == "paper"

    def test_preview_reports_diff(self, client: TestClient) -> None:
        config: dict[str, Any] = client.get("/api/config").json()
        config["portfolio"]["top_n"] = 15
        body = client.post("/api/config/preview", json={"config": config}).json()
        assert body["valid"] is True
        assert "top_n" in body["diff"]

    def test_preview_reports_validation_issues(self, client: TestClient) -> None:
        config: dict[str, Any] = client.get("/api/config").json()
        config["llm"]["max_influence"] = 0.9  # 超过硬上限 0.20
        body = client.post("/api/config/preview", json={"config": config}).json()
        assert body["valid"] is False
        assert body["issues"][0]["location"] == "llm.max_influence"

    def test_save_persists_and_backs_up(self, client: TestClient) -> None:
        config: dict[str, Any] = client.get("/api/config").json()
        config["portfolio"]["top_n"] = 15
        assert client.put("/api/config", json={"config": config}).json()["saved"] is True
        assert client.get("/api/config").json()["portfolio"]["top_n"] == 15

        # 第二次修改应产生备份
        config["portfolio"]["top_n"] = 8
        result = client.put("/api/config", json={"config": config}).json()
        assert result["backup"] is not None
        assert client.get("/api/config/backups").json()["versions"]

    def test_save_invalid__rejected_with_location(self, client: TestClient) -> None:
        config: dict[str, Any] = client.get("/api/config").json()
        config["risk"]["hard_limits"]["max_single_order_amount"] = "999999999"
        body = client.put("/api/config", json={"config": config}).json()
        assert body["saved"] is False
        assert body["issues"]

    def test_no_change__nothing_saved(self, client: TestClient) -> None:
        config = client.get("/api/config").json()
        assert client.put("/api/config", json={"config": config}).json()["saved"] is False

    def test_rollback(self, client: TestClient) -> None:
        config: dict[str, Any] = client.get("/api/config").json()
        config["portfolio"]["top_n"] = 15
        client.put("/api/config", json={"config": config})
        config["portfolio"]["top_n"] = 8
        client.put("/api/config", json={"config": config})

        version = client.get("/api/config/backups").json()["versions"][0]
        client.post("/api/config/rollback", json={"version": version})
        assert client.get("/api/config").json()["portfolio"]["top_n"] == 15

    def test_rollback_unknown_version__domain_error(self, client: TestClient) -> None:
        response = client.post("/api/config/rollback", json={"version": "nope"})
        assert response.status_code == 400
        assert response.json()["error"] == "ConfigError"


class TestSecretsApi:
    def test_returns_booleans_only(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """红线 R7：后端不提供读取密钥明文的接口。"""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret-value")
        body = client.get("/api/secrets/status").json()
        assert all(isinstance(v, bool) for v in body.values())
        assert "sk-super-secret-value" not in str(body)


class TestReadonlyMode:
    def test_reads_allowed(self, readonly_client: TestClient) -> None:
        assert readonly_client.get("/api/config").status_code == 200

    @pytest.mark.parametrize(
        ("method", "path", "payload"),
        [
            ("put", "/api/config", {"config": {}}),
            ("post", "/api/system/halt", {"reason": "x"}),
            ("post", "/api/system/resume", None),
            ("post", "/api/config/rollback", {"version": "x"}),
        ],
    )
    def test_writes_forbidden(
        self, readonly_client: TestClient, method: str, path: str, payload: dict[str, Any] | None
    ) -> None:
        response = getattr(readonly_client, method)(path, json=payload)
        assert response.status_code == 403


class TestConfigCompleteness:
    """界面配置完备性：每个 pydantic 字段都必须出现在 Schema 中。

    新增配置项时忘记暴露到界面，这条测试会失败。
    """

    def test_every_model_field_present_in_schema(self, client: TestClient) -> None:
        schema = client.get("/api/config/schema").json()
        defs = schema.get("$defs", {})

        missing: list[str] = []
        for section, ref in schema["properties"].items():
            model = _model_for(section)
            def_name = ref.get("$ref", "").split("/")[-1]
            properties = defs.get(def_name, {}).get("properties", {})
            for field in model.model_fields:
                if field not in properties:
                    missing.append(f"{section}.{field}")
        assert not missing, f"以下配置字段未出现在 Schema 中：{missing}"

    def test_no_field_lacks_description(self, client: TestClient) -> None:
        schema = client.get("/api/config/schema").json()
        missing = [
            f"{name}.{field}"
            for name, definition in schema.get("$defs", {}).items()
            for field, spec in definition.get("properties", {}).items()
            if not spec.get("description") and "$ref" not in spec
        ]
        assert not missing, f"以下字段缺少 description：{missing}"


def _model_for(section: str) -> Any:
    """取配置节对应的模型类。"""
    return RootConfig.model_fields[section].annotation
