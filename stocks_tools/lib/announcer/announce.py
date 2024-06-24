import os
from sys import platform
from stocks_tools.lib.robinhood_client.rh import RobinhoodClient
from dotenv import dotenv_values
import logging

logging.basicConfig(level=logging.INFO)
cfg = dotenv_values(".env")


class Announcer:
    """Announcer class"""

    def __init__(self):
        # if not Mac OSX throw error
        if platform not in ["darwin"]:
            raise ValueError("Announcer works only on macOS")
        self.rh_client = RobinhoodClient()
        self.rh_client.login()

    def get_stock_price_formatted(self, stock_symbol):
        stock_price = self.rh_client.get_price_for_symbol(stock_symbol)
        return round(float(stock_price[0]), 2)

    def get_stock_prices_formatted(self, stock_symbols):
        stock_prices = {}
        for stock_symbol in stock_symbols:
            stock_name = self.rh_client.get_name_for_symbol(stock_symbol)
            stock_prices[stock_name] = self.get_stock_price_formatted(stock_symbol)
        return stock_prices

    def announce_stock_prices(self, stock_symbols=None):
        if stock_symbols is None or not isinstance(stock_symbols, list):
            raise ValueError("Stock symbols must be a list of stock symbols")
        stock_prices_string = self.get_stock_prices_formatted(stock_symbols)
        os.system("say {}".format(stock_prices_string))
