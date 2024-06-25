"""Robinhood client"""
import robin_stocks.robinhood as rh
from stocks_tools.utils.config import Config
import logging

logging.basicConfig(level=logging.INFO)


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

    def get_price_for_symbol(self, symbol, **kwargs):
        """Get price at Robinhood"""
        return self.rh.get_latest_price(symbol, **kwargs)

    def get_name_for_symbol(self, symbol):
        """Get name for symbol"""
        return self.rh.get_name_by_symbol(symbol)

    def buy_stock(self, symbol, amount):
        # Get the current price of the stock
        quote = self.get_price_for_symbol(symbol, includeExtendedHours=True)
        try:
            price = float(quote[0])
            # Calculate the number of shares to buy
            if amount > 0:
                # Place a market buy order for the calculated number of shares
                buy_order = rh.orders.order_buy_fractional_by_price(symbol, amount)
                logging.info("Buy Order Response", buy_order)
                logging.info(f"Bought {amount} amount of {symbol} at ${price} each, totaling {amount / price} shares")
                return amount / price
            else:
                logging.warning(f"Not enough funds to buy any shares of {symbol} at ${price} per share")
                return None
        except Exception as e:
            logging.exception("Exception buying stocks")
