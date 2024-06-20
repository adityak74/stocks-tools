"""Robinhood client"""
import robin_stocks.robinhood as rh


class RobinhoodClient:
    """Robinhood client"""
    def __init__(self):
        self.current_user = None
        self.rh = rh

    def login(self, username, password):
        """Login to Robinhood"""
        self.current_user = rh.login(username, password)

    def get_price_for_symbol(self, symbol):
        """Get price at Robinhood"""
        return self.rh.get_latest_price(symbol)
