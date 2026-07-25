"""风控规则引擎与熔断状态机。

费用与红利税模型不在这里——它是纯粹的领域计算、不含风控语义，
且被 ``account.ledger`` 与 ``backtest.engine`` 共用，放在包内会形成
risk → portfolio → account → risk 的循环。见 ``quantstock.costs``。
"""

from quantstock.risk.engine import CircuitState, RiskDecision, RiskEngine
from quantstock.risk.halt import HaltSwitch, HardLimitGuard

__all__ = [
    "CircuitState",
    "HaltSwitch",
    "HardLimitGuard",
    "RiskDecision",
    "RiskEngine",
]
