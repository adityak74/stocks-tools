"""Announce Stocks"""

from stock_tools.lib.announcer.announce import Announcer

announcer = Announcer()
announcer.announce_stock_prices(["NVDA"])
