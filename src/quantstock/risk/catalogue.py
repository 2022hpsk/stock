"""风控规则目录（docs/05-风控规范.md、docs/09 P10）。

**为什么需要一份声明式目录**：规则的判定逻辑散落在 ``engine.py`` 与
``halt.py`` 的各个分支里。这对执行没问题，但让"A 类不可关闭"这条约束
只存在于文档的一句话里——界面上凭什么知道哪些开关不该画出来？

把规则的**元数据**（编号、分级、阈值来源、是否可关）集中声明后：

- 界面能据此渲染规则表，并对 A 类**根本不渲染开关**（验收 5）；
- 可以写一条测试断言"没有任何 A 类规则是 closable 的"，
  让它成为会失败的约束而不是会被遗忘的注释；
- 新增规则时忘记登记会被覆盖率测试抓到。

目录**不含判定逻辑**——那仍然在引擎里。这里只回答"这条规则是什么、
归谁管、能不能关"。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["RULES", "RuleClass", "RuleSpec", "rules_by_class"]


class RuleClass(StrEnum):
    """规则分级。

    分级决定的是**用户能对它做什么**，不是它有多重要。
    """

    A = "A"
    """市场规则与绝对硬闸。**不可关闭、不可在界面调整**——
    T+1、涨跌停、整手这些是交易所定的，关掉它只会让委托被券商拒单；
    绝对金额硬闸则是比例风控失效时的最后一道防线。"""
    B = "B"
    """组合约束。阈值可配，但**规则本身不可关闭**。"""
    C = "C"
    """建议性检查。可关闭，关闭后只影响提示，不影响下单合法性。"""


@dataclass(frozen=True, slots=True)
class RuleSpec:
    """一条规则的元数据。"""

    rule_id: str
    name: str
    rule_class: RuleClass
    description: str
    threshold_key: str = ""
    """对应的配置项路径。空表示阈值由市场规则决定，不可配。"""

    @property
    def closable(self) -> bool:
        """能否整条关闭。

        只有 C 类可以。界面必须据此决定要不要画那个开关——
        **画出来再拒绝**和**根本不画**是两回事，前者会让人一直去试。
        """
        return self.rule_class is RuleClass.C

    @property
    def threshold_editable(self) -> bool:
        """阈值能否在界面上改。

        A 类不行：它们的阈值来自交易所规则或手工设定的绝对硬闸，
        让界面能改绝对硬闸，等于给风控开了个后门。
        """
        return self.rule_class is not RuleClass.A and bool(self.threshold_key)


RULES: tuple[RuleSpec, ...] = (
    # ---- A 类：市场规则与绝对硬闸，不可关闭、不可在界面调整 ----
    RuleSpec(
        rule_id="A01",
        name="T+1 可卖量",
        rule_class=RuleClass.A,
        description="当日买入的股票次日才能卖出。卖出数量不得超过可卖量，否则会被券商拒单。",
    ),
    RuleSpec(
        rule_id="A02",
        name="涨跌停",
        rule_class=RuleClass.A,
        description="涨停不可买入、跌停不可卖出——挂了也成交不了，还会让回测虚增收益。",
    ),
    RuleSpec(
        rule_id="A03",
        name="整手买入",
        rule_class=RuleClass.A,
        description="A 股买入必须是 100 股的整数倍（科创板 200 股起、可按 1 股递增）。",
    ),
    RuleSpec(
        rule_id="A04",
        name="停牌与行情可得性",
        rule_class=RuleClass.A,
        description="停牌或当日无行情的标的一律不可交易。拿不准的价格算不出对的仓位。",
    ),
    RuleSpec(
        rule_id="A07",
        name="资金充足性",
        rule_class=RuleClass.A,
        description="买入金额不得超过可用资金。透支下单会被拒，还可能触发券商风控。",
    ),
    RuleSpec(
        rule_id="A10",
        name="绝对金额硬闸",
        rule_class=RuleClass.A,
        description=(
            "单笔上限、单日上限、笔数上限，阈值**手工设定**。"
            "比例风控挡不住计算基数出错——总资产算成十倍时，"
            "「单票不超过 15%」照样会放出一笔十倍大的委托。"
        ),
        threshold_key="risk.hard_limits",
    ),
    RuleSpec(
        rule_id="A11",
        name="账户合理性校验",
        rule_class=RuleClass.A,
        description="总资产落在合理区间、单日变动不超过阈值。用来发现数据或账本本身出了错。",
        threshold_key="risk.hard_limits",
    ),
    RuleSpec(
        rule_id="A12",
        name="急停开关",
        rule_class=RuleClass.A,
        description=(
            "``var/HALT`` 存在时一切下单路径拒绝，且**不会自动恢复**——"
            "必须人工解除。自动恢复会让系统在剧烈波动中反复进出，"
            "而那正是最该停手的时候。"
        ),
    ),
    # ---- B 类：组合约束，阈值可配、规则不可关 ----
    RuleSpec(
        rule_id="B01",
        name="单票仓位上限",
        rule_class=RuleClass.B,
        description="单一标的占总资产的比例上限。",
        threshold_key="accounts.<id>.max_single_position",
    ),
    RuleSpec(
        rule_id="B02",
        name="行业集中度上限",
        rule_class=RuleClass.B,
        description="单一申万一级行业占总资产的比例上限。",
        threshold_key="accounts.<id>.max_industry_exposure",
    ),
    RuleSpec(
        rule_id="B06",
        name="单笔最小金额",
        rule_class=RuleClass.B,
        description="低于此金额的委托不下——手续费占比过高，摊薄后收益为负。",
        threshold_key="accounts.<id>.min_position_value",
    ),
    RuleSpec(
        rule_id="B08",
        name="流动性下限",
        rule_class=RuleClass.B,
        description="近 20 日均成交额低于下限的标的不买。买得进卖不出比买不进更糟。",
        threshold_key="risk.min_liquidity",
    ),
    RuleSpec(
        rule_id="B09",
        name="单日换手上限",
        rule_class=RuleClass.B,
        description="单日换手率上限。A 股单边成本约 0.1%，高频换手一年就是几十个点。",
        threshold_key="accounts.<id>.max_daily_turnover",
    ),
    # ---- C 类：建议性检查，可关闭 ----
    RuleSpec(
        rule_id="C01",
        name="情报风险否决",
        rule_class=RuleClass.C,
        description=(
            "命中情报黑名单的标的禁止买入。这是**单向**的——"
            "情报只能否决买入，不能触发买入（红线 I-R1）。"
        ),
        threshold_key="intel.enabled",
    ),
    RuleSpec(
        rule_id="C02",
        name="止损距离提示",
        rule_class=RuleClass.C,
        description="持仓接近止损位时提示。只提示，不自动下单。",
        threshold_key="accounts.<id>.stop_loss_pct",
    ),
)
"""全部风控规则。新增规则时必须在这里登记，否则界面上看不到它。"""


def rules_by_class(rule_class: RuleClass) -> tuple[RuleSpec, ...]:
    """按分级取规则。

    Args:
        rule_class: 分级。

    Returns:
        该级的全部规则。
    """
    return tuple(r for r in RULES if r.rule_class is rule_class)
