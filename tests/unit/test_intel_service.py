"""情报服务与 CLI 测试。

对应 docs/07-信息情报模块.md 第十节的端到端验收：

1. ``intel fetch`` 一次拉全域并落库，重复执行幂等；
3. 往收件箱放一个 ``.md``，自动入库；
5. 一条"立案调查"公告 → 自动进黑名单、禁止买入、可溯源到原文链接；
8. 关闭全部情报源后系统仍可用，仅标注"情报缺失"。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from quantstock.cli.main import app
from quantstock.config.models import RootConfig
from quantstock.config.settings import Secrets, Settings
from quantstock.infra.clock import FrozenClock, set_clock
from quantstock.infra.types import Symbol
from quantstock.intel.entity import SymbolDictionary
from quantstock.intel.protocols import NewsSource, SourceRegistry
from quantstock.intel.types import EventType, IntelDomain, IntelItem, SourceHealth, SourceTier
from quantstock.services.intel_service import IntelService
from tests.unit.test_intel import CATL, MAOTAI, NOW, item

runner = CliRunner()

pytestmark = pytest.mark.usefixtures("_frozen_intel_service_clock")


@pytest.fixture(autouse=True)
def _frozen_intel_service_clock() -> None:
    """固定时钟。"""
    set_clock(FrozenClock(NOW))


def make_settings(tmp_path: Path) -> Settings:
    """构造指向临时目录的配置。

    Args:
        tmp_path: 临时目录。

    Returns:
        Settings 实例。
    """
    config = RootConfig.model_validate({"app": {"var_dir": str(tmp_path / "var")}})
    return Settings(config=config, secrets=Secrets(), config_dir=tmp_path / "config")


class _ProbeSource:
    """产出一条立案调查公告的源。"""

    name = "sse"
    domains = (IntelDomain.COMPANY,)

    def fetch(self, since: dt.datetime) -> list[IntelItem]:
        return [
            item(
                "贵州茅台收到中国证监会立案调查通知书",
                source="sse",
                tier=SourceTier.OFFICIAL,
                url="https://sse.com.cn/notice/2026-0725",
            )
        ]

    def health_check(self) -> SourceHealth:
        return SourceHealth(source=self.name, ok=True, fetched=1)


class _BrokenSource:
    """总是失败的源。"""

    name = "cls"
    domains = (IntelDomain.MARKET,)

    def fetch(self, since: dt.datetime) -> list[IntelItem]:
        msg = "限流"
        raise TimeoutError(msg)

    def health_check(self) -> SourceHealth:
        return SourceHealth(source=self.name, ok=False, error="限流")


def make_service(tmp_path: Path, *sources: NewsSource) -> IntelService:
    """构造情报服务。

    Args:
        tmp_path: 临时目录。
        *sources: 情报源。

    Returns:
        服务实例。
    """
    return IntelService(
        make_settings(tmp_path),
        registry=SourceRegistry(list(sources)),
        dictionaries=[SymbolDictionary(MAOTAI, "贵州茅台", aliases=("茅台",))],
        holdings=[MAOTAI],
        watchlist=[CATL],
    )


class TestFetch:
    """采集（验收 1、5）。"""

    def test_probe_announcement_blacklists_with_traceable_link(self, tmp_path: Path) -> None:
        service = make_service(tmp_path, _ProbeSource())
        result = service.fetch()

        assert MAOTAI in result.blacklisted
        assert service.is_blocked(MAOTAI)

        entries = service.blacklist_entries()
        assert entries[0].urls == ("https://sse.com.cn/notice/2026-0725",)
        assert "🔗" in entries[0].explain()

    def test_fetch_is_idempotent(self, tmp_path: Path) -> None:
        service = make_service(tmp_path, _ProbeSource())
        service.fetch()
        service.fetch()
        assert len(service.store.read(NOW.date())) == 1

    def test_broken_source_does_not_block_the_run(self, tmp_path: Path) -> None:
        # 验收 8：情报是增强项，不是阻断项
        service = make_service(tmp_path, _ProbeSource(), _BrokenSource())
        result = service.fetch()

        assert len(result.items) == 1
        assert result.digest is not None
        assert "cls" in result.digest.failed_sources

    def test_no_sources_at_all_still_works(self, tmp_path: Path) -> None:
        service = make_service(tmp_path)
        result = service.fetch()
        assert result.items == ()
        assert result.digest is not None
        assert len(result.digest.missing_domains) == len(IntelDomain)

    def test_blacklist_survives_restart(self, tmp_path: Path) -> None:
        make_service(tmp_path, _ProbeSource()).fetch()
        reopened = make_service(tmp_path)
        assert reopened.is_blocked(MAOTAI)

    def test_domain_filter(self, tmp_path: Path) -> None:
        service = make_service(tmp_path, _ProbeSource(), _BrokenSource())
        result = service.fetch(domains=[IntelDomain.COMPANY])
        assert result.digest is not None
        assert "cls" not in result.digest.coverage


class TestInbox:
    """收件箱（验收 3）。"""

    def test_dropped_markdown_gets_ingested(self, tmp_path: Path) -> None:
        service = make_service(tmp_path)
        service.scan_inbox()  # 建目录
        (service.inbox_dir / "2026-07-25-央行降准.md").write_text(
            "央行宣布降准 0.5 个百分点。", encoding="utf-8"
        )

        result = service.fetch()
        assert [i.title for i in result.items] == ["央行降准"]
        assert result.items[0].source_tier is SourceTier.USER

    def test_inbox_readme_is_created(self, tmp_path: Path) -> None:
        service = make_service(tmp_path)
        service.scan_inbox()
        assert (service.inbox_dir / "README.md").exists()

    def test_broken_inbox_file_is_reported_not_fatal(self, tmp_path: Path) -> None:
        service = make_service(tmp_path)
        service.scan_inbox()
        (service.inbox_dir / "bad.json").write_text("{ nope", encoding="utf-8")

        result = service.fetch()
        assert result.digest is not None
        assert not result.digest.coverage["external:inbox"].ok


class TestEvidence:
    """证据链（支柱③）。"""

    def test_evidence_carries_link_and_time(self, tmp_path: Path) -> None:
        # 红线 I-R4
        service = make_service(tmp_path, _ProbeSource())
        service.fetch()

        evidence = service.evidence_for(MAOTAI)
        assert len(evidence) == 1
        assert evidence[0].url
        assert evidence[0].publish_at.tzinfo is not None

    def test_evidence_respects_pit(self, tmp_path: Path) -> None:
        # 验收 7：回测不得看到未来情报
        service = make_service(tmp_path, _ProbeSource())
        service.fetch()
        past = NOW - dt.timedelta(days=1)
        assert service.evidence_for(MAOTAI, as_of=past) == []

    def test_absent_note_is_explicit(self, tmp_path: Path) -> None:
        # 留空让人分不清"没查"和"查了没有"
        service = make_service(tmp_path)
        assert "无相关消息" in service.absent_note()

    def test_no_evidence_for_unrelated_symbol(self, tmp_path: Path) -> None:
        service = make_service(tmp_path, _ProbeSource())
        service.fetch()
        assert service.evidence_for(Symbol("000001.SZ")) == []


class TestDigestAndStatus:
    """摘要与状态。"""

    def test_digest_from_stored_items(self, tmp_path: Path) -> None:
        service = make_service(tmp_path, _ProbeSource())
        service.fetch()

        digest = service.digest()
        assert digest.total_items == 1
        assert digest.portfolio_alerts
        assert digest.portfolio_alerts[0].symbol == MAOTAI

    def test_render_digest(self, tmp_path: Path) -> None:
        service = make_service(tmp_path, _ProbeSource())
        service.fetch()
        text = "\n".join(service.render_digest(service.digest()))
        assert "持仓相关" in text
        assert "情报健康" in text

    def test_status_snapshot(self, tmp_path: Path) -> None:
        service = make_service(tmp_path, _ProbeSource())
        service.fetch()
        status = service.status()

        assert status.sources == 1
        assert status.latest_date == NOW.date()
        assert status.blacklisted == 1
        assert "黑名单 1 只" in status.message

    def test_note_records_a_manual_item(self, tmp_path: Path) -> None:
        service = make_service(tmp_path)
        entry = service.note("某公司大额订单落地", symbols=["002415.SZ"], importance=80)
        assert entry.source == "external:cli"
        assert Symbol("002415.SZ") in entry.symbols

    def test_import_rows_skips_bad_rows(self, tmp_path: Path) -> None:
        service = make_service(tmp_path)
        items = service.import_rows(
            [{"title": "好的"}, {"title": "", "body": ""}], source_name="myfeed"
        )
        assert len(items) == 1
        assert items[0].source == "external:myfeed"


class TestIntelCli:
    """CLI。"""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        (config_dir / "base.yaml").write_text(
            yaml.safe_dump({"app": {"var_dir": str(tmp_path / "var")}}, allow_unicode=True),
            encoding="utf-8",
        )
        return config_dir

    def test_fetch_runs_with_no_sources(self, workspace: Path) -> None:
        result = runner.invoke(app, ["intel", "fetch", "-c", str(workspace)])
        assert result.exit_code == 0, result.output
        assert "无情报的域" in result.output

    def test_note_then_digest(self, workspace: Path) -> None:
        noted = runner.invoke(
            app,
            [
                "intel",
                "note",
                "某公司中标 5 亿元大额订单",
                "-c",
                str(workspace),
                "--symbol",
                "002415.SZ",
                "--url",
                "https://example.com/x",
                "--importance",
                "80",
            ],
        )
        assert noted.exit_code == 0, noted.output
        assert "已录入" in noted.output

        shown = runner.invoke(app, ["intel", "digest", "-c", str(workspace)])
        assert shown.exit_code == 0, shown.output
        assert "中标" in shown.output

    def test_note_without_url_warns(self, workspace: Path) -> None:
        # 没有链接的条目进不了建议解释（红线 I-R4）
        result = runner.invoke(app, ["intel", "note", "随手记一笔", "-c", str(workspace)])
        assert result.exit_code == 0, result.output
        assert "未提供原文链接" in result.output

    def test_inbox_preview_does_not_move_files(self, workspace: Path, tmp_path: Path) -> None:
        inbox = tmp_path / "var" / "intel" / "inbox"
        inbox.mkdir(parents=True)
        (inbox / "note.md").write_text("公司拟回购股份", encoding="utf-8")

        result = runner.invoke(app, ["intel", "inbox", "-c", str(workspace), "--preview"])
        assert result.exit_code == 0, result.output
        assert (inbox / "note.md").exists()

    def test_import_csv(self, workspace: Path, tmp_path: Path) -> None:
        path = tmp_path / "feed.csv"
        path.write_text("title,importance\n甲公司中标,60\n乙公司回购,50\n", encoding="utf-8")

        result = runner.invoke(app, ["intel", "import", str(path), "-c", str(workspace)])
        assert result.exit_code == 0, result.output
        assert "已导入 2 条" in result.output

    def test_import_missing_file_exits_nonzero(self, workspace: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["intel", "import", str(tmp_path / "nope.csv"), "-c", str(workspace)]
        )
        assert result.exit_code == 1
        assert "文件不存在" in result.output

    def test_blacklist_empty(self, workspace: Path) -> None:
        result = runner.invoke(app, ["intel", "blacklist", "-c", str(workspace)])
        assert result.exit_code == 0, result.output
        assert "当前无情报黑名单" in result.output

    def test_blacklist_lists_entries(self, workspace: Path, tmp_path: Path) -> None:
        service = IntelService(
            make_settings(tmp_path),
            registry=SourceRegistry([_ProbeSource()]),
            dictionaries=[SymbolDictionary(MAOTAI, "贵州茅台")],
        )
        service.fetch()

        result = runner.invoke(app, ["intel", "blacklist", "-c", str(workspace)])
        assert result.exit_code == 0, result.output
        assert "600519.SH" in result.output
        assert "禁止买入" in result.output


def test_event_type_coverage_is_complete() -> None:
    """每个事件类型都要有情绪先验，漏一个会静默变成 0。"""
    from quantstock.intel.classify import EVENT_SENTIMENT_PRIOR  # noqa: PLC0415

    missing = [e.value for e in EventType if e not in EVENT_SENTIMENT_PRIOR]
    assert missing == []


class TestIngestRunsThePipeline:
    """外置导入必须走完整流水线（端到端发现的缺陷）。

    曾经 ``intel inbox`` / ``note`` / ``import`` 都是直接 ``store.write()``，
    跳过了分类、打分与黑名单评估：条目进了库，但 ``event_type=None``、
    ``importance=0``，一条"立案调查"公告既不触发禁买、也不进日报重大消息区。

    单元测试全都走 ``fetch()``，所以一直没发现；端到端跑一遍立刻暴露。
    """

    def test_ingest_classifies_scores_and_blacklists(self, tmp_path: Path) -> None:
        service = make_service(tmp_path)
        raw = item(
            "贵州茅台收到中国证监会立案调查通知书",
            source="external:inbox",
            tier=SourceTier.OFFICIAL,
            url="https://sse.com.cn/n/1",
        )
        assert raw.event_type is None, "入库前是未分类的原始条目"

        result = service.ingest([raw])
        stored = result.items[0]

        assert stored.event_type is EventType.REGULATORY_PROBE
        assert stored.importance >= 80
        assert stored.sentiment < 0
        assert service.is_blocked(MAOTAI)

    def test_inbox_path_reaches_the_blacklist(self, tmp_path: Path) -> None:
        # 验收 5 的完整路径：往收件箱丢个文件 → 自动禁买
        service = make_service(tmp_path)
        service.scan_inbox()
        (service.inbox_dir / "probe.md").write_text(
            "---\n"
            "symbols: [600519.SH]\n"
            "source_tier: OFFICIAL\n"
            "url: https://sse.com.cn/n/1\n"
            "title: 贵州茅台收到中国证监会立案调查通知书\n"
            "---\n正文\n",
            encoding="utf-8",
        )
        report = service.scan_inbox()
        service.ingest(report.items)
        assert service.is_blocked(MAOTAI)

    def test_undeclared_domain_does_not_pin_the_item(self, tmp_path: Path) -> None:
        # 外置导入未写 domain 时兜底为 COMPANY。若流水线把兜底值当成"源方声明"，
        # 一条讲降准的随手记会被永远钉在个股域，宏观段里根本看不到
        service = make_service(tmp_path)
        note = service.note("央行宣布降准 0.5 个百分点，释放长期资金约 1 万亿元")
        assert note.domain is IntelDomain.COMPANY
        assert not note.domain_declared

        stored = service.ingest([note]).items[0]
        assert stored.event_type is EventType.MONETARY
        assert stored.domain is IntelDomain.MACRO

    def test_declared_domain_is_respected(self, tmp_path: Path) -> None:
        service = make_service(tmp_path)
        note = service.note("央行宣布降准", domain="policy")
        assert note.domain_declared
        assert service.ingest([note]).items[0].domain is IntelDomain.POLICY
