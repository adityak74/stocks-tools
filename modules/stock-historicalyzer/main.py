import json

import robin_stocks.robinhood as rh
from datetime import datetime
import signal
import sys
from dotenv import dotenv_values
import logging

logging.basicConfig(level=logging.INFO)
cfg = dotenv_values(".env")


def signal_handler(sig, frame):
    print("Exiting gracefully...")
    rh.logout()  # Log out from Robinhood
    sys.exit(0)


def get_stock_price_historicals(stock_symbol):
    return rh.get_stock_historicals(stock_symbol, interval="hour", span="3month")


def store_historicals_to_json(symbol, data):
    with open(f"{symbol}_{datetime.now().strftime('%Y%m%d')}.json", "w") as file:
        json.dump(data, file, indent=4)


def main():
    username = cfg["ROBINHOOD_USERNAME"]
    password = cfg["ROBINHOOD_PASSWORD"]
    rh.login(username, password)

    stock_symbols = cfg["ROBINHOOD_SYMBOLS"].split(",")
    for stock_symbol in stock_symbols:
        stock_price_historicals = get_stock_price_historicals(stock_symbol)
        store_historicals_to_json(stock_symbol, stock_price_historicals)


if __name__ == "__main__":
    # Set up the signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    main()
