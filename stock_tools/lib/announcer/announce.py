import robin_stocks.robinhood as rh
import os
import time
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


def get_stock_price(stock_symbol):
    return rh.get_latest_price(stock_symbol)


def get_stock_price_formatted(stock_symbol):
    stock_price = get_stock_price(stock_symbol)
    return round(float(stock_price[0]), 2)


def get_stock_prices_formatted(stock_symbols):
    stock_prices = {}
    for stock_symbol in stock_symbols:
        stock_name = rh.get_name_by_symbol(stock_symbol)
        stock_prices[stock_name] = get_stock_price_formatted(stock_symbol)
    return stock_prices


def get_stock_prices_formatted_string(stock_prices):
    stock_prices_string = ""
    for stock_name, stock_price in stock_prices.items():
        stock_prices_string += "{} {}, ".format(stock_name, stock_price)
    return stock_prices_string.strip(", ")


def say_stock_prices(stock_prices):
    stock_prices_string = get_stock_prices_formatted_string(stock_prices)
    print(stock_prices_string)
    os.system("say {}".format(stock_prices_string))


def main(sleep_time_seconds):
    stock_symbols = cfg["ROBINHOOD_SYMBOLS"].split(",")
    previous_prices = {}

    while True:
        time.sleep(sleep_time_seconds)
        current_prices = get_stock_prices_formatted(stock_symbols)

        # Check if any price has changed
        prices_changed = False
        for stock_name, current_price in current_prices.items():
            previous_price = previous_prices.get(stock_name)
            if previous_price is None or previous_price != current_price:
                prices_changed = True
                break

        # Announce prices if they have changed
        if prices_changed:
            say_stock_prices(current_prices)
            previous_prices = current_prices


if __name__ == "__main__":
    # Set up the signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    sleep_time = int(cfg.get("SLEEP_TIME", 10))
    sleep_time = max(5, sleep_time)  # Minimum sleep time of 5 seconds
    logging.info("Sleep time: {}".format(sleep_time))
    main(sleep_time)
