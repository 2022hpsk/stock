"""行情数据源适配器。

三个实现，覆盖不同场景：

- ``CsvSource``：从本地 CSV 读取。**无网络也能跑通全链路**，
  也是"我已经有一份券商/同花顺导出数据"的用户的入口；
- ``BaoStockSource``：免费、稳定、含完整历史与退市标的，历史批量初始化的主力；
- ``AkShareSource``：接口覆盖广，用于增量更新与补缺。

适配器**必须在本层把外部代码格式归一化为标准 Symbol**，
禁止让 ``sh.600519`` / ``600519`` 这类原始格式流出 data 层
（见 docs/01-开发规范.md 第四条）。
"""

from quantstock.data.sources.akshare_source import AkShareSource
from quantstock.data.sources.baostock_source import BaoStockSource
from quantstock.data.sources.csv_source import CsvSource

__all__ = ["AkShareSource", "BaoStockSource", "CsvSource"]
