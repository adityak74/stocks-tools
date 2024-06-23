"""Robinhood client"""
import robin_stocks.robinhood as rh
from stock_tools.utils.config import Config


class RobinhoodClient:
    """Robinhood client"""
    def __init__(self):
        self.current_user = None
        self.rh = rh

    def login(self):
        """Login to Robinhood"""
        cfg = Config().get_config()
        username = cfg['ROBINHOOD_USERNAME']
        password = cfg['ROBINHOOD_PASSWORD']
        self.current_user = rh.login(username, password)

    def get_price_for_symbol(self, symbol):
        """Get price at Robinhood"""
        return self.rh.get_latest_price(symbol)

    def get_name_for_symbol(self, symbol):
        """Get name for symbol"""
        return self.rh.get_name_by_symbol(symbol)
