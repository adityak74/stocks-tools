import robin_stocks.robinhood as rh
import os
import time
import argparse
import signal
import sys
from dotenv import dotenv_values

cfg = dotenv_values(".env")


def signal_handler(sig, frame):
    print('Exiting gracefully...')
    rh.logout()  # Log out from Robinhood
    sys.exit(0)


def main(sleep_time):
    rh.login(username=cfg["ROBINHOOD_USERNAME"], password=cfg["ROBINHOOD_PASSWORD"])

    while True:
        time.sleep(sleep_time)
        price = rh.get_latest_price("RIVN")
        formatted_price = round(float(price[0]), 2)
        text_to_say = 'Rivian price is {} dollars'.format(formatted_price)
        print(text_to_say)
        os.system("say {}".format(text_to_say))


if __name__ == "__main__":
    # Set up the signal handler for graceful exit
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Rivian stock price notifier.')
    parser.add_argument('sleep_time', type=int, help='Time to sleep between price checks (in seconds).')
    args = parser.parse_args()

    main(args.sleep_time)
