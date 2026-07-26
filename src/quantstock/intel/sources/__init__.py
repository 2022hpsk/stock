"""情报源适配器。

- ``RssSource``：通用 RSS/Atom，用户可在配置里加任意站点。**唯一无需第三方
  数据库依赖的源**，也是最不容易失效的；
- ``ClsTelegraphSource``：财联社电报，A 股快讯主力（经 AkShare）；
- ``EastMoneySource``：东方财富个股新闻与全球快讯（经 AkShare）。

三者都遵守 I-R6：限流礼貌抓取、失败只影响自己覆盖的域。
"""

from quantstock.intel.sources.akshare_feeds import ClsTelegraphSource, EastMoneySource
from quantstock.intel.sources.rss import RssSource

__all__ = ["ClsTelegraphSource", "EastMoneySource", "RssSource"]
