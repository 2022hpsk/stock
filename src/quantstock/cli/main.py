"""命令行入口。

CLI 是与 Web UI 平级的**薄**客户端：只做参数解析、调用 ``services``、渲染结果，
不含任何业务逻辑（见 docs/09-可视化界面规格.md §1.1）。
"""

from __future__ import annotations

import datetime as dt
import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from quantstock.config import load_settings
from quantstock.config.models import RootConfig
from quantstock.infra.clock import today
from quantstock.infra.errors import QuantStockError
from quantstock.infra.logging import setup_logging
from quantstock.infra.types import Side, Symbol
from quantstock.services.execution_service import (
    ConfirmationDecision,
    ExecutionReport,
    ExecutionService,
    IntentPreview,
    SkipReason,
)
from quantstock.services.intel_service import IntelDomain, IntelService, parse_payload
from quantstock.services.system_service import SystemService

app = typer.Typer(
    name="quantstock",
    help="A股/场内基金 个人量化投研与半自动交易系统",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(name="config", help="配置查看与校验", no_args_is_help=True)
app.add_typer(config_app)

console = Console()
err_console = Console(stderr=True)

ConfigDirOption = Annotated[
    Path,
    typer.Option("--config-dir", "-c", help="配置目录路径"),
]


@app.callback()
def _root(
    log_level: Annotated[
        str, typer.Option("--log-level", help="日志级别：DEBUG/INFO/WARNING/ERROR")
    ] = "INFO",
    log_format: Annotated[
        str, typer.Option("--log-format", help="日志格式：json/console")
    ] = "console",
) -> None:
    """全局选项。在任何子命令执行前初始化日志。"""
    setup_logging(level=log_level, fmt=log_format)


@app.command()
def version() -> None:
    """显示版本号。"""
    try:
        console.print(f"quantstock {pkg_version('quantstock')}")
    except PackageNotFoundError:
        console.print("quantstock (开发模式，未安装)")


@config_app.command("show")
def config_show(
    config_dir: ConfigDirOption = Path("config"),
    as_json: Annotated[bool, typer.Option("--json", help="以 JSON 输出")] = False,
) -> None:
    """显示当前生效配置（密钥已脱敏）。"""
    try:
        settings = load_settings(config_dir)
    except QuantStockError as exc:
        err_console.print(f"[red]配置加载失败[/red]：{exc}")
        raise typer.Exit(code=1) from exc

    dumped = settings.redacted_dump()
    if as_json:
        console.print_json(json.dumps(dumped, ensure_ascii=False, default=str))
        return

    table = Table(title="生效配置", show_lines=False)
    table.add_column("配置节", style="cyan", no_wrap=True)
    table.add_column("内容")
    for section, value in dumped.items():
        table.add_row(section, json.dumps(value, ensure_ascii=False, default=str, indent=2))
    console.print(table)


@config_app.command("check")
def config_check(config_dir: ConfigDirOption = Path("config")) -> None:
    """校验配置文件是否合法。用于提交前与部署前自检。"""
    try:
        load_settings(config_dir)
    except QuantStockError as exc:
        err_console.print(f"[red]✗ 配置校验失败[/red]\n{exc}")
        raise typer.Exit(code=1) from exc
    console.print("[green]✓ 配置校验通过[/green]")


@config_app.command("schema")
def config_schema(
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="输出文件路径，缺省打印到终端")
    ] = None,
) -> None:
    """导出配置的 JSON Schema。

    界面的配置表单由该 Schema 自动生成——新增配置项只需改 pydantic 模型，
    界面自动出现对应控件（见 docs/09-可视化界面规格.md §4.1）。
    """
    schema = json.dumps(RootConfig.model_json_schema(), ensure_ascii=False, indent=2)
    if output is None:
        console.print_json(schema)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(schema + "\n", encoding="utf-8")
    console.print(f"[green]✓[/green] Schema 已写入 {output}")


@app.command()
def ui(
    config_dir: ConfigDirOption = Path("config"),
    host: Annotated[str, typer.Option("--host", help="监听地址")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="监听端口")] = 8686,
    readonly: Annotated[
        bool, typer.Option("--readonly", help="只读模式：可看不可操作，适合手机查看")
    ] = False,
    open_browser: Annotated[bool, typer.Option("--open", help="自动打开浏览器")] = False,
) -> None:
    """启动本地可视化界面。所有操作（含全部配置）都可在界面完成。"""
    import uvicorn  # noqa: PLC0415 - 延迟导入，未装 web 依赖时其它命令仍可用

    from quantstock.web.app import create_app  # noqa: PLC0415

    if host not in {"127.0.0.1", "localhost", "::1"}:
        # 界面是下单入口，暴露到局域网意味着同网段任何设备都能操作账户
        err_console.print(
            f"[yellow]⚠ 警告[/yellow]：监听地址为 {host} 而非本地回环，"
            "界面将暴露给网络中的其它设备。请确认这是你想要的。"
        )

    web_app = create_app(config_dir=config_dir, readonly=readonly)
    token = web_app.state.app_state.access_token
    url = f"http://{host}:{port}"

    console.print()
    console.print(f"  界面地址  [cyan]{url}[/cyan]")
    console.print(f"  访问口令  [bold yellow]{token}[/bold yellow]")
    console.print(f"  模式      {'只读' if readonly else '可读写'}")
    console.print()

    if open_browser:
        import webbrowser  # noqa: PLC0415 - 仅在需要时导入

        webbrowser.open(url)

    uvicorn.run(web_app, host=host, port=port, log_config=None)


@app.command()
def halt(
    reason: Annotated[str, typer.Option("--reason", "-r", help="急停原因，必填")],
    config_dir: ConfigDirOption = Path("config"),
) -> None:
    """一键急停：此后所有下单路径一律拒绝，直到显式 resume。"""
    settings = load_settings(config_dir)
    state = SystemService(settings).halt(reason=reason, by="cli")
    console.print(f"[red]● 已急停[/red]  原因：{state.reason}")
    console.print(f"  标志文件：{settings.var_dir / 'HALT'}")
    console.print("  解除请执行：[cyan]quantstock resume[/cyan]")


@app.command()
def resume(
    config_dir: ConfigDirOption = Path("config"),
    yes: Annotated[bool, typer.Option("--yes", "-y", help="跳过二次确认")] = False,
) -> None:
    """解除急停。"""
    settings = load_settings(config_dir)
    service = SystemService(settings)
    current = service.halt_switch.state()
    if not current.halted:
        console.print("[green]系统当前未处于急停状态[/green]")
        return
    console.print(f"当前急停原因：{current.reason}（{current.halted_at}）")
    if not yes and not typer.confirm("确认解除急停？"):
        console.print("已取消")
        return
    service.resume(by="cli")
    console.print("[green]✓ 已解除急停[/green]")


@app.command()
def status(config_dir: ConfigDirOption = Path("config")) -> None:
    """显示系统状态。"""
    settings = load_settings(config_dir)
    snapshot = SystemService(settings).status()

    console.print(f"quantstock {snapshot.version}   {snapshot.checked_at}")
    if snapshot.halt.halted:
        console.print(f"[red]● 已急停[/red]  {snapshot.halt.reason}")
    else:
        console.print("[green]● 运行中[/green]")
    console.print(f"交易通道：{snapshot.broker}")
    console.print(
        f"大模型：{'启用 (' + snapshot.llm_mode + ')' if snapshot.llm_enabled else '关闭'}"
    )

    table = Table(show_header=True)
    table.add_column("组件", style="cyan")
    table.add_column("状态")
    table.add_column("说明")
    for component in snapshot.components:
        table.add_row(component.name, "✓" if component.ok else "✗", component.detail)
    console.print(table)


# ---------------------------------------------------------------- 执行
plan_app = typer.Typer(help="交易计划：查看与执行。")
app.add_typer(plan_app, name="plan")


@plan_app.command("show")
def plan_show(
    config_dir: ConfigDirOption = Path("config"),
    date: Annotated[str, typer.Option("--date", "-d", help="交易日 YYYY-MM-DD，默认今日")] = "",
) -> None:
    """展示某交易日的计划（含四支柱解释）。"""
    settings = load_settings(config_dir)
    service = ExecutionService(settings)
    trade_date = dt.date.fromisoformat(date) if date else today()
    plan = service.store.latest(trade_date)
    if plan is None:
        console.print(f"[yellow]{trade_date} 没有已保存的计划[/yellow]")
        raise typer.Exit(1)

    console.print(f"计划 {plan.plan_id}   {plan.trade_date}   {len(plan.intents)} 条建议")
    console.print(f"数据指纹 {plan.data_fingerprint or '—'}   参数哈希 {plan.param_hash or '—'}")
    for intent in plan.intents:
        action = "买入" if intent.side is Side.BUY else "卖出"
        console.print(
            f"\n[bold]{action} {intent.symbol}[/bold]  {intent.qty} 股  "
            f"{intent.price_low}~{intent.price_high}  ({intent.urgency.value})"
        )
        console.print(f"  {intent.rationale.verdict}")
        for line in intent.rationale.technical.statements():
            console.print(f"    · {line}")
        for note in intent.rationale.falsification:
            console.print(f"    [dim]证伪：{note}[/dim]")
    if plan.rejected:
        console.print(
            f"\n[dim]被否决 {len(plan.rejected)} 只：为什么没买与为什么买了同样重要[/dim]"
        )
        for rejected in plan.rejected:
            console.print(f"  [dim]{rejected.symbol}  {rejected.reason}[/dim]")


@app.command()
def execute(
    config_dir: ConfigDirOption = Path("config"),
    date: Annotated[str, typer.Option("--date", "-d", help="交易日 YYYY-MM-DD，默认今日")] = "",
    only: Annotated[str, typer.Option("--only", help="只执行指定标的，逗号分隔")] = "",
    live: Annotated[bool, typer.Option("--live", help="使用真实资金通道（红线 R5）")] = False,
) -> None:
    """逐单确认并执行当日计划。

    每一条都要你亲手过一遍；跳过时必须选原因——复盘要按原因统计人工干预的价值。
    """
    settings = load_settings(config_dir)
    service = ExecutionService(settings)
    trade_date = dt.date.fromisoformat(date) if date else today()

    plan = service.store.latest(trade_date)
    if plan is None:
        console.print(f"[yellow]{trade_date} 没有已保存的计划，请先生成[/yellow]")
        raise typer.Exit(1)

    only_symbols = (
        frozenset(Symbol(s.strip()) for s in only.split(",") if s.strip()) if only else None
    )
    preview = service.preview(plan, current_prices={})
    if preview.halted:
        console.print(f"[red]● 系统处于急停状态：{preview.halt_reason}[/red]")
        raise typer.Exit(1)

    console.print(
        f"计划 {preview.plan_id}   通道 [cyan]{preview.broker}[/cyan]   "
        f"买入 {preview.total_buy}   卖出 {preview.total_sell}"
    )
    if preview.requires_live_flag and not live:
        console.print("[yellow]该通道涉及真实资金，需加 --live 才会真正提交[/yellow]")

    confirmed_by = typer.prompt("确认人（记入审计，红线 R5）")
    decisions = [
        _confirm_one(item)
        for item in preview.items
        if only_symbols is None or item.symbol in only_symbols
    ]

    report = service.execute(
        plan,
        decisions=decisions,
        current_prices={},
        confirmed_by=confirmed_by,
        only_symbols=only_symbols,
        live=live,
    )
    _print_report(report)


def _confirm_one(item: IntentPreview) -> ConfirmationDecision:
    """对单条建议做逐单确认。

    Args:
        item: 执行前视图条目。

    Returns:
        人工决定。
    """
    action = "买入" if item.side is Side.BUY else "卖出"
    console.print(
        f"\n[bold]{action} {item.symbol}[/bold]  {item.qty} 股  "
        f"限价 {item.limit_price}  约 {item.estimated_amount} 元"
    )
    console.print(f"  {item.verdict}")
    if item.drift is not None:
        style = "red" if item.needs_review else "dim"
        console.print(f"  [{style}]{item.drift.message}[/{style}]")

    if typer.confirm("  执行这一条？", default=not item.needs_review):
        return ConfirmationDecision(intent_id=item.intent_id, accepted=True)

    reasons = [r.value for r in SkipReason]
    console.print(f"  跳过原因：{'  '.join(f'[{i}] {r}' for i, r in enumerate(reasons, 1))}")
    choice = typer.prompt("  选择编号", type=int, default=len(reasons))
    index = min(max(choice, 1), len(reasons)) - 1
    return ConfirmationDecision(
        intent_id=item.intent_id,
        accepted=False,
        skip_reason=SkipReason(reasons[index]),
        skip_note=typer.prompt("  备注（可留空）", default="", show_default=False),
    )


def _print_report(report: ExecutionReport) -> None:
    """打印执行报告。

    Args:
        report: 执行报告。
    """
    if report.aborted:
        console.print(f"\n[red]● 已中止，未提交任何订单[/red]\n  {report.abort_reason}")
        return

    console.print(
        f"\n[green]✓ 完成[/green]  提交 {len(report.submitted)} 笔  跳过 {len(report.skipped)} 笔"
    )
    if report.manual_checklist:
        console.print("\n[bold]手工执行清单（照抄到券商 App）[/bold]")
        for line in report.manual_checklist:
            console.print(f"  {line}")
    if counts := report.skip_reasons():
        console.print("\n[dim]跳过原因统计：" + "  ".join(f"{k}×{v}" for k, v in counts.items()))


@app.command("cancel-all")
def cancel_all(
    config_dir: ConfigDirOption = Path("config"),
    yes: Annotated[bool, typer.Option("--yes", "-y", help="跳过二次确认")] = False,
) -> None:
    """撤销所有未成交委托。急停后应立即执行。"""
    settings = load_settings(config_dir)
    if not yes and not typer.confirm("确认撤销全部未成交委托？"):
        console.print("已取消")
        return
    count = ExecutionService(settings).cancel_all()
    console.print(f"[green]✓ 已发出撤单指令[/green]  影响 {count} 笔")


# ---------------------------------------------------------------- 情报
intel_app = typer.Typer(help="情报：采集、外置导入、摘要、黑名单。", no_args_is_help=True)
app.add_typer(intel_app, name="intel")


@intel_app.command("fetch")
def intel_fetch(
    config_dir: ConfigDirOption = Path("config"),
    domain: Annotated[str, typer.Option("--domain", help="只采集某个域，ALL 表示全部")] = "ALL",
    lookback: Annotated[int, typer.Option("--lookback", help="向前追溯天数")] = 7,
) -> None:
    """采集全域情报并落库。重复执行幂等。"""
    settings = load_settings(config_dir)
    service = IntelService(settings)
    domains = None if domain.upper() == "ALL" else [IntelDomain(domain.lower())]

    result = service.fetch(domains=domains, lookback_days=lookback)
    console.print(f"[green]✓[/green] {result.summary}")

    if result.blacklisted:
        console.print("\n[red]● 新增情报黑名单（禁止买入）[/red]")
        for entry in service.blacklist_entries():
            if entry.symbol in result.blacklisted:
                console.print(f"  {entry.explain()}")

    if result.digest is not None and (missing := result.digest.missing_domains):
        # 情报缺失只是降级，不阻断建议——但必须说出来
        console.print(f"\n[yellow]⚠ 今日无情报的域：{'、'.join(d.value for d in missing)}[/yellow]")


@intel_app.command("digest")
def intel_digest(
    config_dir: ConfigDirOption = Path("config"),
    date: Annotated[str, typer.Option("--date", "-d", help="交易日 YYYY-MM-DD")] = "",
    session: Annotated[str, typer.Option("--session", help="pre 盘前 / post 盘后")] = "post",
) -> None:
    """输出分域摘要。命中持仓的事件置顶。"""
    settings = load_settings(config_dir)
    service = IntelService(settings)
    trade_date = dt.date.fromisoformat(date) if date else today()

    digest = service.digest(trade_date=trade_date, session=session)
    for line in service.render_digest(digest):
        console.print(line)


@intel_app.command("note")
def intel_note(
    text: Annotated[str, typer.Argument(help="情报内容")],
    config_dir: ConfigDirOption = Path("config"),
    symbol: Annotated[list[str] | None, typer.Option("--symbol", help="关联标的，可重复")] = None,
    domain: Annotated[str, typer.Option("--domain", help="情报域")] = "",
    event: Annotated[str, typer.Option("--event", help="事件类型")] = "",
    importance: Annotated[int, typer.Option("--importance", help="重要性 0-100")] = 0,
    sentiment: Annotated[float, typer.Option("--sentiment", help="情绪 -1~1")] = 0.0,
    url: Annotated[str, typer.Option("--url", help="原文链接")] = "",
) -> None:
    """直接录入一条情报（外置导入方式二）。"""
    settings = load_settings(config_dir)
    service = IntelService(settings)

    item = service.note(
        text,
        symbols=symbol or [],
        domain=domain,
        event_type=event,
        importance=importance,
        sentiment=sentiment,
        url=url,
    )
    service.store.write([item])
    console.print(f"[green]✓ 已录入[/green]  {item.cite()}")
    if not url:
        # 没有链接的条目进不了建议解释（红线 I-R4），提前说清楚
        console.print("[yellow]⚠ 未提供原文链接，该条不会进入建议解释的情报支柱[/yellow]")


@intel_app.command("inbox")
def intel_inbox(
    config_dir: ConfigDirOption = Path("config"),
    preview: Annotated[bool, typer.Option("--preview", help="只看不导入，不移动文件")] = False,
) -> None:
    """扫描外置导入收件箱（外置导入方式一）。"""
    settings = load_settings(config_dir)
    service = IntelService(settings)

    report = service.scan_inbox(move=not preview)
    console.print(f"收件箱：{service.inbox_dir}")
    console.print(f"[green]✓[/green] {report.summary}")
    for item in report.items:
        console.print(f"  · {item.cite()}")
    for path, reason in report.failed:
        console.print(f"  [red]✗ {path.name}：{reason}[/red]")
    if not preview and report.items:
        service.store.write(report.items)


@intel_app.command("import")
def intel_import(
    path: Annotated[Path, typer.Argument(help="CSV / JSON 文件路径")],
    config_dir: ConfigDirOption = Path("config"),
    source_name: Annotated[str, typer.Option("--source-name", help="来源名")] = "import",
) -> None:
    """批量导入文件（外置导入方式二）。"""
    settings = load_settings(config_dir)
    service = IntelService(settings)

    if not path.exists():
        console.print(f"[red]文件不存在：{path}[/red]")
        raise typer.Exit(1)

    rows = parse_payload(path, path.read_text(encoding="utf-8"))
    items = service.import_rows(rows, source_name=source_name)
    service.store.write(items)
    console.print(f"[green]✓ 已导入 {len(items)} 条[/green]（原文件 {len(rows)} 行）")


@intel_app.command("blacklist")
def intel_blacklist(config_dir: ConfigDirOption = Path("config")) -> None:
    """查看当前生效的情报黑名单。"""
    settings = load_settings(config_dir)
    entries = IntelService(settings).blacklist_entries()
    if not entries:
        console.print("当前无情报黑名单")
        return

    console.print(f"[red]● 情报黑名单 {len(entries)} 只（禁止买入）[/red]")
    for entry in entries:
        console.print(f"  {entry.explain()}")


if __name__ == "__main__":  # pragma: no cover
    app()
