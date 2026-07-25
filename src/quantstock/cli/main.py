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


if __name__ == "__main__":  # pragma: no cover
    app()
