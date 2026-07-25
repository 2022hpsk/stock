"""配置模型。

规范见 docs/01-开发规范.md 第八条与 docs/09-可视化界面规格.md 第四节。

**每个字段都必须写 ``description`` 与取值约束**——界面的配置表单由本模块导出的
JSON Schema 自动生成，没有 description 的字段在界面上就是一个没有说明的裸输入框。
新增配置项只需改这里，界面自动出现对应控件。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "AdvisorConfig",
    "AppConfig",
    "CircuitBreakerConfig",
    "DataConfig",
    "ExecutionConfig",
    "FactorConfig",
    "HardLimitConfig",
    "IntelConfig",
    "LLMConfig",
    "LLMTaskConfig",
    "LabelConfig",
    "NotifyConfig",
    "PortfolioConfig",
    "QualityConfig",
    "RiskConfig",
    "RootConfig",
    "ScheduleConfig",
]

Ratio = Annotated[float, Field(ge=0.0, le=1.0)]
"""0~1 之间的比例。"""

_FLOAT_TOLERANCE: Final = 1e-9
"""比例求和校验的浮点容差。比例本身是配置项而非金额，用浮点无碍（红线 R1 只约束金额）。"""

PositiveInt = Annotated[int, Field(gt=0)]


class _Base(BaseModel):
    """所有配置模型的基类。"""

    model_config = ConfigDict(
        extra="forbid",  # 拼错的配置项必须报错，而不是被静默忽略
        validate_assignment=True,
        frozen=False,
        str_strip_whitespace=True,
    )


# --------------------------------------------------------------------- 应用
class AppConfig(_Base):
    """基础运行配置。"""

    timezone: Literal["Asia/Shanghai"] = Field(
        default="Asia/Shanghai",
        description="系统时区。红线 R3 要求全系统统一为 Asia/Shanghai，不可更改。",
    )
    var_dir: str = Field(
        default="./var",
        description="运行时数据目录，存放数据湖、计划、审计流水、账本。该目录不入版本库。",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="日志级别。排查问题时临时调为 DEBUG。",
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="日志格式。json 便于检索，console 便于本地阅读。",
    )
    random_seed: int = Field(
        default=20260101,
        description="全局随机种子。固定种子保证同一输入重跑结果完全一致（红线 R6）。",
    )


# --------------------------------------------------------------------- 数据
class DataConfig(_Base):
    """行情与基本面数据配置。"""

    start_date: str = Field(
        default="2015-01-01",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="数据湖起始日期。越早数据量越大，初始化越慢。",
    )
    source_chain: list[str] = Field(
        default=["tushare", "akshare", "baostock"],
        min_length=1,
        description=(
            "数据源降级链，按顺序尝试。全部失败时当日不出建议——"
            "数据不可信时拒绝出建议，而不是用降级数据硬出建议。"
        ),
    )
    adjust_for_research: Literal["hfq"] = Field(
        default="hfq",
        description=(
            "因子与回测使用的复权口径。必须为后复权，保证历史不随新的除权事件漂移（红线 R4）。"
        ),
    )
    adjust_for_trading: Literal["none"] = Field(
        default="none",
        description="下单与展示使用的复权口径。必须为不复权真实价（红线 R4）。",
    )
    minute_freqs: list[Literal["M1", "M5", "M15", "M30", "M60"]] = Field(
        default=["M1", "M5", "M30"],
        description="需要采集的分钟频率。只对关注池采集，控制数据体量。",
    )
    minute_retention_days: PositiveInt = Field(
        default=730,
        description="分钟线保留天数，超期滚动删除。",
    )
    rate_limit_per_min: PositiveInt = Field(
        default=60,
        description="对数据源的每分钟请求上限，避免被限流封禁。",
    )
    cache_ttl_sec: PositiveInt = Field(
        default=3600,
        description="热点数据的进程内缓存存活时间。",
    )
    init_tier: Literal["core", "all"] = Field(
        default="core",
        description=(
            "初始化范围。core 只拉沪深300+中证500+主要ETF（约900只，20分钟可用）；"
            "all 拉全市场（数小时，支持断点续传）。"
        ),
    )


class QualityConfig(_Base):
    """数据质量校验配置（DQ01–DQ11，见 docs/04-数据规格.md 第六节）。"""

    fail_on: list[str] = Field(
        default=["DQ01", "DQ02", "DQ03", "DQ04", "DQ11"],
        description="致命级校验项，不通过则拒绝入库。DQ11 为幸存者偏差检测，不可移除。",
    )
    coverage_threshold: Ratio = Field(
        default=0.99,
        description="活跃标的当日数据覆盖率下限，低于此值告警并将缺失标的移出当日 universe。",
    )
    cross_check_sample_pct: Ratio = Field(
        default=0.01,
        description="与备用数据源交叉比对的抽样比例。",
    )
    cross_check_max_deviation: Ratio = Field(
        default=0.005,
        description="交叉比对的收盘价最大允许偏差，超出则告警。",
    )
    abort_advise_missing_ratio: Ratio = Field(
        default=0.02,
        description="关键字段缺失比例超过此值时当日不出建议。宁可不做，不可用脏数据决策。",
    )


# --------------------------------------------------------------------- 因子与标签
class LabelConfig(_Base):
    """预测目标定义（F2.5）。

    没有 label 就无法做因子有效性检验。
    """

    type: Literal["forward_return", "forward_excess_return", "risk_adjusted"] = Field(
        default="forward_excess_return",
        description="预测目标类型。excess 为相对基准的超额收益。",
    )
    horizon_days: PositiveInt = Field(
        default=20,
        description="预测未来多少个交易日的收益。",
    )
    benchmark: str = Field(
        default="hs300",
        description="计算超额收益时的基准。",
    )
    exclude_unbuyable_at_entry: bool = Field(
        default=True,
        description=(
            "剔除入场日涨停或停牌（无法买入）的样本。"
            "**强烈建议保持开启**——这类样本若计入 IC 会严重高估因子有效性，"
            "很多“神因子”本质只是在预测涨停。"
        ),
    )
    exclude_unsellable_at_exit: bool = Field(
        default=True,
        description="剔除出场日跌停或停牌（无法卖出）的样本。",
    )
    winsorize_label: Ratio = Field(
        default=0.01,
        description="label 双侧缩尾比例，防止极端值主导回归。",
    )


class FactorConfig(_Base):
    """因子计算配置。"""

    winsorize_mad_n: float = Field(
        default=3.0,
        gt=0,
        description="去极值倍数，超出中位数 ± n×MAD 的值被截断。",
    )
    fillna: Literal["industry_median", "zero", "drop"] = Field(
        default="industry_median",
        description="缺失值填充方式。",
    )
    standardize: Literal["zscore", "rank", "none"] = Field(
        default="zscore",
        description="横截面标准化方式。",
    )
    neutralize: list[Literal["industry", "market_cap"]] = Field(
        default=["industry", "market_cap"],
        description="中性化维度，剔除行业与市值暴露。",
    )
    ic_window_days: PositiveInt = Field(
        default=756,
        description="计算 IC 均值与 IR 的滚动窗口（交易日），756 约为 3 年。",
    )
    max_correlation: Ratio = Field(
        default=0.7,
        description="因子间相关系数上限，超出的因子对必须择一或正交化，否则等同重复下注。",
    )
    label: LabelConfig = Field(default_factory=LabelConfig, description="预测目标定义。")


# --------------------------------------------------------------------- 风控
class HardLimitConfig(_Base):
    """绝对金额硬闸（风控规则 A10/A11）。

    比例风控无法防御计算基数出错——若总资产被算成真实值的 10 倍，
    所有比例约束仍会通过，但实际下单金额是灾难性的。

    **这些阈值必须由用户按自己实际资金规模手工设定**，程序不自动推导：
    自动推导会被同一个错误数据污染，失去防护意义。
    """

    enabled: bool = Field(
        default=True,
        description="是否启用绝对金额硬闸。真实交易通道下不可关闭。",
    )
    max_single_order_amount: Decimal = Field(
        default=Decimal("100000"),
        gt=0,
        description="单笔委托金额上限（元）。按你的实际资金规模设定。",
    )
    max_daily_total_amount: Decimal = Field(
        default=Decimal("300000"),
        gt=0,
        description="单日累计委托金额上限（元）。",
    )
    max_daily_order_count: PositiveInt = Field(
        default=20,
        description="单日最大委托笔数。异常放大通常意味着程序出错。",
    )
    max_single_order_qty: PositiveInt = Field(
        default=100000,
        description="单笔委托股数上限。",
    )
    min_account_value_sanity: Decimal = Field(
        default=Decimal("1000"),
        gt=0,
        description="账户总资产合理性下限（元）。低于此值判定账户数据异常，停止下单。",
    )
    max_account_value_sanity: Decimal = Field(
        default=Decimal("100000000"),
        gt=0,
        description="账户总资产合理性上限（元）。高于此值判定账户数据异常，停止下单。",
    )
    max_account_value_daily_change: Ratio = Field(
        default=0.30,
        description="账户总资产单日变动上限。超出且无对应资金流水则判定同步异常。",
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> HardLimitConfig:
        """校验阈值之间的一致性。

        Returns:
            校验通过的自身。

        Raises:
            ValueError: 阈值自相矛盾。
        """
        if self.max_single_order_amount > self.max_daily_total_amount:
            msg = (
                f"单笔上限 {self.max_single_order_amount} 不应大于单日上限 "
                f"{self.max_daily_total_amount}"
            )
            raise ValueError(msg)
        if self.min_account_value_sanity >= self.max_account_value_sanity:
            msg = "账户总资产合理性下限必须小于上限"
            raise ValueError(msg)
        return self


class CircuitBreakerConfig(_Base):
    """组合级熔断阈值（见 docs/05-风控规范.md §4.2）。"""

    watch_daily_loss: Ratio = Field(default=0.03, description="进入 WATCH 的当日亏损阈值。")
    watch_drawdown_20d: Ratio = Field(default=0.12, description="进入 WATCH 的 20 日回撤阈值。")
    halted_daily_loss: Ratio = Field(
        default=0.05, description="进入 HALTED 的当日亏损阈值。HALTED 下禁止一切买入。"
    )
    halted_drawdown_20d: Ratio = Field(default=0.18, description="进入 HALTED 的 20 日回撤阈值。")
    recover_days: PositiveInt = Field(
        default=3, description="回撤收敛后需连续观察多少个交易日才恢复 NORMAL。"
    )
    recover_drawdown: Ratio = Field(default=0.08, description="恢复 NORMAL 所需的回撤收敛水平。")

    @model_validator(mode="after")
    def _check_ordering(self) -> CircuitBreakerConfig:
        """校验 WATCH 阈值必须比 HALTED 更宽松。

        Returns:
            校验通过的自身。

        Raises:
            ValueError: 阈值顺序错误。
        """
        if self.watch_daily_loss >= self.halted_daily_loss:
            msg = "WATCH 的当日亏损阈值必须小于 HALTED 的阈值"
            raise ValueError(msg)
        if self.watch_drawdown_20d >= self.halted_drawdown_20d:
            msg = "WATCH 的回撤阈值必须小于 HALTED 的阈值"
            raise ValueError(msg)
        return self


class RiskConfig(_Base):
    """风控配置。

    注意：A 类规则（T+1、涨跌停、整手、停牌、急停等市场规则）**不在配置中**，
    因为它们不可关闭也不可调整。界面上同样不提供关闭入口。
    """

    hard_limits: HardLimitConfig = Field(
        default_factory=HardLimitConfig, description="绝对金额硬闸（A10/A11）。"
    )
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig, description="组合级熔断状态机阈值。"
    )
    correlation_window: PositiveInt = Field(
        default=60, description="持仓相关性计算窗口（交易日）。"
    )
    correlation_threshold: Ratio = Field(default=0.85, description="持仓两两相关系数告警阈值。")


# --------------------------------------------------------------------- 大模型
class LLMTaskConfig(_Base):
    """单个 LLM 任务的配置。"""

    enabled: bool = Field(default=True, description="是否启用该任务。")
    model: Literal["fast", "main", "deep"] = Field(
        default="main", description="使用的模型档位。fast 便宜、deep 最强。"
    )
    prompt_version: str = Field(
        default="v1",
        description="提示词版本。改版视为策略变更，会进入 param_hash 影响可复现性。",
    )


class LLMConfig(_Base):
    """大模型配置（见 docs/10-大模型集成规格.md）。

    核心边界：LLM 做「非结构化文本 → 结构化特征」，量化做「结构化特征 → 决策」。
    """

    enabled: bool = Field(
        default=False,
        description="LLM 总开关。关闭后系统退化为纯量化，**功能完整可用**（红线 LR2）。",
    )
    mode: Literal["off", "live", "replay"] = Field(
        default="live",
        description=(
            "运行模式。live 实际调用并写缓存；replay 只读缓存保证可复现；"
            "**回测由引擎强制设为 replay**，实时调用会抛异常（红线 LR3）。"
        ),
    )
    max_influence: float = Field(
        default=0.15,
        ge=0.0,
        le=0.20,
        description=(
            "LLM 对最终打分的影响系数 α：final = base × (1 + α × adjustment)。"
            "硬上限 0.20，超出直接拒绝启动（红线 LR2）。"
        ),
    )
    temperature: float = Field(
        default=0.0, ge=0.0, le=1.0, description="采样温度。默认 0 以最大化确定性。"
    )
    anonymize_in_backtest: bool = Field(
        default=True,
        description=(
            "回测中把标的名替换为代号。**强烈建议保持开启**——"
            "模型的知识截止日期晚于回测区间，认出具体公司会导致训练集泄漏（红线 LR4）。"
        ),
    )
    strip_absolute_dates: bool = Field(
        default=True,
        description="提示词中用相对日期表述，降低模型定位到具体历史时点的能力。",
    )
    daily_budget_usd: float = Field(
        default=2.0, ge=0, description="每日 API 费用上限（美元），超出自动降级。"
    )
    monthly_budget_usd: float = Field(default=30.0, ge=0, description="每月 API 费用上限（美元）。")
    rate_limit_per_min: PositiveInt = Field(default=20, description="每分钟调用上限。")
    model_fast: str = Field(
        default="claude-haiku-4-5-20251001", description="fast 档位模型 ID，用于批量情报分类。"
    )
    model_main: str = Field(
        default="claude-sonnet-5", description="main 档位模型 ID，用于研判与解释生成。"
    )
    model_deep: str = Field(
        default="claude-opus-5", description="deep 档位模型 ID，用于复杂研判，默认不启用。"
    )
    tasks: dict[str, LLMTaskConfig] = Field(
        default_factory=lambda: {
            "intel_classify": LLMTaskConfig(model="fast"),
            "position_judge": LLMTaskConfig(model="main"),
            "market_judge": LLMTaskConfig(model="main"),
            "explain": LLMTaskConfig(model="main"),
        },
        description="各任务的独立配置。",
    )

    @model_validator(mode="after")
    def _check_mode(self) -> LLMConfig:
        """总开关关闭时强制 mode=off，避免配置自相矛盾。

        Returns:
            校验后的自身。
        """
        if not self.enabled and self.mode != "off":
            object.__setattr__(self, "mode", "off")
        return self


# --------------------------------------------------------------------- 建议与执行
class IntelConfig(_Base):
    """情报配置。

    情报**不单独产生买入信号**（红线 I-R1），这里的每一项都只在
    "解释 / 有界软调节 / 单向风险否决"三条通路上生效。
    """

    enabled: bool = Field(
        default=True,
        description=(
            "是否启用情报模块。关闭后系统仍能完整出建议，日报只标注「情报缺失」——"
            "情报是增强项而非阻断项。"
        ),
    )
    lookback_days: PositiveInt = Field(default=7, description="采集与证据检索的回溯天数。")
    retention_days: PositiveInt = Field(
        default=1095, description="情报保留期（自然日），默认 3 年。"
    )
    dedup_similarity: Ratio = Field(
        default=0.65,
        description=(
            "近似去重的 SimHash 相似度阈值。不要想当然设成 0.9："
            "SimHash 是 64 位有损指纹，实测同一事件的媒体改写落在 0.64~0.77，"
            "阈值过高会让近似去重根本不触发。"
        ),
    )
    dedup_window_hours: PositiveInt = Field(
        default=6,
        description="近似去重的时间窗。窗外的相同措辞视为旧闻重发或事件进展，不合并。",
    )
    max_score_influence: Ratio = Field(
        default=0.20,
        le=0.20,
        description=("情报因子对最终打分的影响上限（红线 I-R1）。硬上限 0.20，配置不得超过。"),
    )
    blacklist_ttl_days: PositiveInt = Field(default=60, description="情报黑名单有效期（自然日）。")
    blacklist_importance_threshold: int = Field(
        default=80,
        ge=0,
        le=100,
        description="风险类事件触发黑名单的 importance 门槛。",
    )
    negative_streak: PositiveInt = Field(
        default=3, description="窗口内累计负面事件达到该数即拉黑。"
    )
    user_importance_cap: int = Field(
        default=90,
        ge=0,
        le=100,
        description=(
            "人工导入条目的 importance 上限。防止单条手工输入压过全部量化信号——"
            "这是本系统最容易被自己绕过风控的地方。"
        ),
    )
    sentiment_use_llm: bool = Field(
        default=False,
        description=(
            "是否额外用 LLM 打情绪分。规则分始终保留，"
            "两者分歧 > 0.5 时日报标注「情绪判定存在分歧」（红线 I-R3）。"
        ),
    )
    domains: list[str] = Field(
        default_factory=lambda: [
            "macro",
            "policy",
            "industry",
            "company",
            "market",
            "overseas",
            "calendar",
        ],
        description="启用的情报域。",
    )


class AdvisorConfig(_Base):
    """建议生成配置。"""

    plan_valid_days: PositiveInt = Field(
        default=1, description="交易计划有效期（交易日）。过期计划拒绝执行。"
    )
    price_drift_threshold: Ratio = Field(
        default=0.03,
        description="T+1 开盘价相对建议区间的漂移阈值，超出则该单标记 STALE 需二次确认。",
    )
    require_full_rationale: bool = Field(
        default=True,
        description=(
            "要求每条建议具备完整四支柱解释（量化依据/持仓技术分析/情报证据/反面证据）。"
            "缺支柱的建议不进入交易计划——宁可不建议，也不给无法解释的建议。"
        ),
    )
    price_band_pct: Ratio = Field(
        default=0.006, description="建议限价区间宽度，相对参考价的单侧比例。"
    )


class ExecutionConfig(_Base):
    """执行配置。"""

    broker: Literal["paper", "manual", "file_bridge", "qmt", "ptrade"] = Field(
        default="paper",
        description=(
            "交易通道。默认 paper 模拟撮合；manual 输出手工执行清单；"
            "file_bridge 通过计划文件对接执行端。真实通道需显式 --live。"
        ),
    )
    require_manual_confirm: bool = Field(
        default=True,
        description="是否要求逐单人工确认。真实通道下强制为 true，不可关闭（红线 R5）。",
    )
    cancel_unfilled_before: str = Field(
        default="14:50",
        pattern=r"^\d{2}:\d{2}$",
        description="收盘前自动撤销未成交委托的时间点。",
    )
    split_order_threshold_pct: Ratio = Field(
        default=0.02,
        description="单笔占当日成交量超过此比例时拆单，降低冲击成本。",
    )

    @model_validator(mode="after")
    def _enforce_manual_confirm(self) -> ExecutionConfig:
        """真实交易通道下强制人工确认（红线 R5）。

        Returns:
            校验后的自身。

        Raises:
            ValueError: 真实通道下试图关闭人工确认。
        """
        if self.broker in {"qmt", "ptrade"} and not self.require_manual_confirm:
            msg = "真实交易通道下 require_manual_confirm 不可关闭（红线 R5）"
            raise ValueError(msg)
        return self


class PortfolioConfig(_Base):
    """组合构建配置。"""

    weighting: Literal["equal", "score_weighted", "inverse_vol", "risk_parity", "mean_variance"] = (
        Field(
            default="inverse_vol",
            description=(
                "权重分配方法。个人账户持仓通常 ≤12 只，均值方差的收益远小于其估计误差风险，"
                "默认用最稳健的波动率倒数。"
            ),
        )
    )
    top_n: PositiveInt = Field(default=10, description="选股数量上限。")
    core_ratio: Ratio = Field(
        default=0.60, description="核心仓比例（ETF/低波稳健）。核心+卫星应为 1。"
    )
    satellite_ratio: Ratio = Field(default=0.40, description="卫星仓比例（个股进攻）。")
    rebalance_band: Ratio = Field(
        default=0.02,
        description="调仓缓冲带。目标与当前权重偏离小于此值不调仓，避免频繁小额交易。",
    )

    @model_validator(mode="after")
    def _check_structure(self) -> PortfolioConfig:
        """核心与卫星比例之和必须为 1。

        Returns:
            校验后的自身。

        Raises:
            ValueError: 比例之和不为 1。
        """
        total = self.core_ratio + self.satellite_ratio
        if abs(total - 1.0) > _FLOAT_TOLERANCE:
            msg = f"core_ratio + satellite_ratio 必须为 1，当前为 {total}"
            raise ValueError(msg)
        return self


class ScheduleConfig(_Base):
    """定时任务时间点（Asia/Shanghai）。"""

    daily_bars: str = Field(default="15:05", pattern=r"^\d{2}:\d{2}$", description="日线采集。")
    intel_pre_market: str = Field(
        default="08:30", pattern=r"^\d{2}:\d{2}$", description="盘前情报采集与摘要。"
    )
    intel_post_market: str = Field(
        default="18:30", pattern=r"^\d{2}:\d{2}$", description="盘后情报采集与摘要。"
    )
    factors: str = Field(default="15:30", pattern=r"^\d{2}:\d{2}$", description="因子计算。")
    advise: str = Field(default="16:00", pattern=r"^\d{2}:\d{2}$", description="生成每日建议。")
    notify: str = Field(default="16:25", pattern=r"^\d{2}:\d{2}$", description="推送日报。")


class NotifyConfig(_Base):
    """通知配置。"""

    channels: list[Literal["console", "email", "wecom", "telegram", "file"]] = Field(
        default=["console"], description="启用的通知渠道。"
    )
    on_events: list[str] = Field(
        default=[
            "daily_report",
            "risk_alert",
            "circuit_breaker",
            "data_failure",
            "intel_critical",
        ],
        description="订阅的事件类型。",
    )
    quiet_hours: list[str] = Field(
        default=["22:00", "07:00"],
        min_length=2,
        max_length=2,
        description="免打扰时段 [开始, 结束]。CRITICAL 级别不受限制。",
    )


# --------------------------------------------------------------------- 根配置
class RootConfig(_Base):
    """全部配置的根。

    界面的配置页由本模型导出的 JSON Schema 自动生成
    （``RootConfig.model_json_schema()``）。
    """

    app: AppConfig = Field(default_factory=AppConfig, description="基础运行配置。")
    data: DataConfig = Field(default_factory=DataConfig, description="数据采集配置。")
    quality: QualityConfig = Field(default_factory=QualityConfig, description="数据质量校验配置。")
    factors: FactorConfig = Field(default_factory=FactorConfig, description="因子计算配置。")
    portfolio: PortfolioConfig = Field(
        default_factory=PortfolioConfig, description="组合构建配置。"
    )
    risk: RiskConfig = Field(default_factory=RiskConfig, description="风控配置。")
    intel: IntelConfig = Field(default_factory=IntelConfig, description="情报配置。")
    llm: LLMConfig = Field(default_factory=LLMConfig, description="大模型配置。")
    advisor: AdvisorConfig = Field(default_factory=AdvisorConfig, description="建议生成配置。")
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig, description="执行配置。")
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig, description="定时任务时间点。")
    notify: NotifyConfig = Field(default_factory=NotifyConfig, description="通知配置。")
