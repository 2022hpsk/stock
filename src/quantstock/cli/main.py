"""命令行入口。

CLI 是与 Web UI 平级的**薄**客户端：只做参数解析、调用 ``services``、渲染结果，
不含任何业务逻辑（见 docs/09-可视化界面规格.md §1.1）。
"""

from __future__ import annotations

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
from quantstock.infra.errors import QuantStockError
from quantstock.infra.logging import setup_logging
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


if __name__ == "__main__":  # pragma: no cover
    app()
