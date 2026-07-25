"""统一异常树。

规范见 docs/01-开发规范.md 第六条：

- 外部 I/O 必须包裹为本项目异常，不得让第三方异常泄漏到业务层。
- 风控拒绝不是异常流程，只有显式要求时才抛 ``RiskRejectedError``。
- 涉及资金的操作失败必须 fail-fast。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AccountSyncError",
    "AdjustMismatchError",
    "BrokerConnectionError",
    "ConfigError",
    "DataError",
    "DataQualityError",
    "DataSourceError",
    "ExecutionError",
    "HardLimitExceededError",
    "IntelError",
    "LLMError",
    "LLMLiveCallInBacktestError",
    "LLMOutputInvalidError",
    "LedgerError",
    "LookAheadError",
    "OrderRejectedError",
    "QuantStockError",
    "RiskRejectedError",
    "StrategyError",
    "TradingHaltedError",
]


class QuantStockError(Exception):
    """所有本项目异常的根。

    Args:
        message: 人类可读的错误说明，应包含具体数值以便排查。
        context: 结构化上下文，会被日志与审计流水原样记录。
    """

    def __init__(self, message: str, /, **context: Any) -> None:  # noqa: ANN401 - 上下文本就是任意结构
        """初始化。

        Args:
            message: 人类可读的错误说明。
            **context: 结构化上下文，键值任意，会被日志与审计原样记录。
        """
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = context

    def __str__(self) -> str:
        """渲染为 ``消息 (键=值, ...)`` 形式，便于日志排查。"""
        if not self.context:
            return self.message
        detail = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({detail})"


# --------------------------------------------------------------------- 配置
class ConfigError(QuantStockError):
    """配置缺失、非法或自相矛盾。"""


# --------------------------------------------------------------------- 数据
class DataError(QuantStockError):
    """数据层的通用错误。"""


class DataSourceError(DataError):
    """外部数据源不可用、限流或返回异常。"""


class DataQualityError(DataError):
    """数据质量校验未通过（DQ01–DQ11）。"""


class AdjustMismatchError(DataError):
    """复权口径混用（红线 R4）。"""


class LookAheadError(DataError):
    """检测到未来函数：访问了 ``as_of`` 之后才可见的数据（红线 R2）。"""


# --------------------------------------------------------------------- 情报
class IntelError(QuantStockError):
    """情报采集、解析或导入失败。"""


# --------------------------------------------------------------------- 大模型
class LLMError(QuantStockError):
    """大模型调用相关错误。"""


class LLMLiveCallInBacktestError(LLMError):
    """回测中试图发起实时 LLM 调用（红线 LR3）。

    回测必须走 ``replay`` 模式只读缓存，否则结果不可复现且可能引入未来信息。
    """


class LLMOutputInvalidError(LLMError):
    """LLM 输出未通过结构化校验或反幻觉校验（红线 LR5）。"""


# --------------------------------------------------------------------- 策略
class StrategyError(QuantStockError):
    """策略配置或信号生成失败。"""


# --------------------------------------------------------------------- 风控
class RiskRejectedError(QuantStockError):
    """风控拒绝。

    注意：常规流程下 ``RiskEngine.pre_trade_check`` 返回 ``RiskDecision`` 而不抛异常；
    只有调用方显式要求 ``raise_on_reject=True`` 时才抛出本异常。
    """


class HardLimitExceededError(RiskRejectedError):
    """触发绝对金额硬闸（规则 A10/A11）。

    与比例风控相互独立，触发时必须中止整个计划而非跳过单笔。
    """


class TradingHaltedError(RiskRejectedError):
    """急停标志存在，拒绝一切下单（规则 A12）。"""


# --------------------------------------------------------------------- 账本
class LedgerError(QuantStockError):
    """账本流水非法或重放失败（红线 R8）。"""


class AccountSyncError(QuantStockError):
    """账户同步或对账失败。"""


# --------------------------------------------------------------------- 执行
class ExecutionError(QuantStockError):
    """委托执行相关错误。"""


class BrokerConnectionError(ExecutionError):
    """券商通道连接失败。"""


class OrderRejectedError(ExecutionError):
    """券商拒绝委托。"""
