"""执行端计划执行器（F15）。

**这个文件刻意只依赖 Python 标准库**，并且不 import 本项目的任何模块——
它要能直接拷进 QMT 客户端内置的 Python 环境或 PTrade 的券商机房环境运行，
那些环境不允许装第三方包，也不可能塞进整个 quantstock。

用法（在执行端机器上）::

    python plan_executor.py --bridge-dir D:/quantstock-bridge --dry-run
    python plan_executor.py --bridge-dir D:/quantstock-bridge --live

工作流程：

1. 读取研究端写出的 ``plan-<date>.json``；
2. 校验 schema 版本与订单字段；
3. 逐单下单（接入券商 SDK 处见 ``submit_order``）；
4. 回写 ``fills-<date>.json``。

**幂等**：已执行的 ``intent_id`` 记在 ``executed.log`` 里，重复运行不会重复下单。
这是本脚本最重要的一条保证——执行端往往会被手工重跑。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EXECUTED_LOG = "executed.log"
REQUIRED_FIELDS = ("order_id", "intent_id", "symbol", "side", "qty", "price")


def load_plan(path: Path) -> list[dict[str, Any]]:
    """读取并校验计划文件。

    Args:
        path: 计划文件路径。

    Returns:
        订单列表。

    Raises:
        SystemExit: 文件缺失、格式非法或 schema 版本不匹配。
    """
    if not path.exists():
        sys.exit(f"计划文件不存在：{path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"计划文件无法解析：{exc}")

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        # 版本不匹配直接停下：字段含义可能已变，猜着执行等于拿真钱赌
        sys.exit(f"schema 版本不匹配：期望 {SCHEMA_VERSION}，实际 {version}。请更新执行器。")

    orders = payload.get("orders", [])
    for order in orders:
        missing = [f for f in REQUIRED_FIELDS if f not in order]
        if missing:
            sys.exit(f"订单缺少必需字段 {missing}：{order}")
        if int(order["qty"]) <= 0:
            sys.exit(f"订单数量非法：{order}")
        if order["side"] not in ("buy", "sell"):
            sys.exit(f"订单方向非法：{order}")
    return list(orders)


def load_executed(bridge_dir: Path) -> set[str]:
    """读取已执行的 intent 集合。

    Args:
        bridge_dir: 桥接目录。

    Returns:
        已执行的 intent_id 集合。
    """
    log = bridge_dir / EXECUTED_LOG
    if not log.exists():
        return set()
    return {line.strip() for line in log.read_text(encoding="utf-8").splitlines() if line.strip()}


def mark_executed(bridge_dir: Path, intent_id: str) -> None:
    """记录某 intent 已执行。

    **先记录再下单**——宁可漏下一单（可人工补），也不要因为下单成功
    但记录失败而重复下单。重复下单是不可逆的。

    Args:
        bridge_dir: 桥接目录。
        intent_id: 意图标识。
    """
    with (bridge_dir / EXECUTED_LOG).open("a", encoding="utf-8") as handle:
        handle.write(f"{intent_id}\n")
        handle.flush()


def submit_order(order: dict[str, Any], *, live: bool) -> dict[str, Any]:
    """提交单笔订单。

    **接入券商 SDK 的位置就在这里。** 例如 QMT::

        from xtquant import xttrader
        xt_id = xttrader.order_stock(account, symbol, order_type, qty, price_type, price)

    保持这个函数是脚本里唯一碰券商 API 的地方——换券商时只改它。

    Args:
        order: 订单字典。
        live: 是否真实下单。False 时只打印不执行。

    Returns:
        成交回报字典。
    """
    label = "买入" if order["side"] == "buy" else "卖出"
    print(  # noqa: T201 - 执行端脚本，终端输出即为界面
        f"  [{label}] {order['symbol']}  {order['qty']} 股  限价 {order['price']}"
        + ("" if live else "   （dry-run，未真实下单）")
    )

    if not live:
        return {}

    # ---- 券商 SDK 调用处 ----
    # broker_order_id = xttrader.order_stock(...)  # noqa: ERA001 - 接入点示例，刻意保留
    # 下面返回的是模拟回报，接入时替换为真实成交查询
    return {
        "order_id": order["order_id"],
        "symbol": order["symbol"],
        "side": order["side"],
        "qty": order["qty"],
        "price": order["price"],
        "fee": "0",
    }


def write_fills(bridge_dir: Path, fills: list[dict[str, Any]], stamp: str) -> Path:
    """回写成交回报。

    Args:
        bridge_dir: 桥接目录。
        fills: 成交列表。
        stamp: 日期戳。

    Returns:
        回报文件路径。
    """
    path = bridge_dir / f"fills-{stamp}.json"
    path.write_text(
        json.dumps(
            {"generated_at": dt.datetime.now().isoformat(), "fills": fills},  # noqa: DTZ005
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    """入口。

    Args:
        argv: 命令行参数。

    Returns:
        进程退出码。
    """
    parser = argparse.ArgumentParser(description="quantstock 执行端计划执行器")
    parser.add_argument("--bridge-dir", required=True, help="与研究端共享的桥接目录")
    parser.add_argument("--date", default=None, help="计划日期 YYYY-MM-DD，默认今日")
    parser.add_argument("--live", action="store_true", help="真实下单。不加则为 dry-run")
    args = parser.parse_args(argv)

    bridge = Path(args.bridge_dir)
    stamp = args.date or dt.date.today().isoformat()  # noqa: DTZ011
    orders = load_plan(bridge / f"plan-{stamp}.json")
    executed = load_executed(bridge)

    print(f"计划 {stamp}：共 {len(orders)} 笔")  # noqa: T201
    if not args.live:
        print("*** dry-run 模式，不会真实下单 ***")  # noqa: T201

    fills: list[dict[str, Any]] = []
    skipped = 0
    for order in orders:
        intent_id = str(order["intent_id"])
        if intent_id in executed:
            print(f"  跳过 {order['symbol']}：该意图已执行过")  # noqa: T201
            skipped += 1
            continue

        if args.live:
            mark_executed(bridge, intent_id)
        result = submit_order(order, live=args.live)
        if result:
            fills.append(result)

    if fills:
        path = write_fills(bridge, fills, stamp)
        print(f"成交回报已写入 {path}")  # noqa: T201

    print(f"完成：下单 {len(fills)} 笔，跳过 {skipped} 笔")  # noqa: T201
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
