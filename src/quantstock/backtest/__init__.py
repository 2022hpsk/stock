"""事件驱动回测引擎、成本模型、绩效指标、过拟合防御。"""

from quantstock.backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    MarketView,
    Order,
    RejectReason,
)
from quantstock.backtest.metrics import PerformanceStats, compute_performance
from quantstock.backtest.robustness import (
    CapacityEstimate,
    CostSensitivity,
    SensitivityReport,
    cost_sensitivity,
    estimate_capacity,
    parameter_sensitivity,
)
from quantstock.backtest.trials import (
    DSR_FLOOR,
    PBO_CEILING,
    AdmissionVerdict,
    Trial,
    TrialLog,
    TrialRecorder,
    admission_check,
    deflated_sharpe_ratio,
    parameter_plateau,
    probability_of_backtest_overfitting,
)

__all__ = [
    "DSR_FLOOR",
    "PBO_CEILING",
    "AdmissionVerdict",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "CapacityEstimate",
    "CostSensitivity",
    "MarketView",
    "Order",
    "PerformanceStats",
    "RejectReason",
    "SensitivityReport",
    "Trial",
    "TrialLog",
    "TrialRecorder",
    "admission_check",
    "compute_performance",
    "cost_sensitivity",
    "deflated_sharpe_ratio",
    "estimate_capacity",
    "parameter_plateau",
    "parameter_sensitivity",
    "probability_of_backtest_overfitting",
]
