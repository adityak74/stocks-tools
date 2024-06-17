import robin_stocks.robinhood as rh
from dotenv import dotenv_values
import logging

logging.basicConfig(level=logging.INFO)
cfg = dotenv_values(".env")


def login_robinhood():
    username = cfg["ROBINHOOD_USERNAME"]
    password = cfg["ROBINHOOD_PASSWORD"]
    rh.login(username, password)


def buy_stock(symbol, amount):
    # Get the current price of the stock
    quote = rh.stocks.get_latest_price(symbol, includeExtendedHours=True)
    price = float(quote[0])

    # Calculate the number of shares to buy
    if amount > 0:
        # Place a market buy order for the calculated number of shares
        # buy_order = rh.orders.order_buy_fractional_by_price(symbol, amount)
        # print(buy_order)
        logging.info(f"Bought {amount} amount of {symbol} at ${price} each, totaling {amount / price} shares")
        return amount / price
    else:
        logging.warning(f"Not enough funds to buy any shares of {symbol} at ${price} per share")
        return None


def adjust_etf_config(etf_config):
    # Step 1: Round each percentage to the nearest integer
    rounded_values = {key: round(value) for key, value in etf_config.items()}

    # Step 2: Calculate the sum of the rounded values
    total = sum(rounded_values.values())

    # Step 3: Adjust the values to sum to 100
    difference = 100 - total

    # Sort the keys based on the fractional part of the original values (descending)
    keys_sorted = sorted(etf_config.keys(), key=lambda x: etf_config[x] - rounded_values[x], reverse=True)

    # Adjust the values
    for i in range(abs(difference)):
        if difference > 0:
            rounded_values[keys_sorted[i]] += 1
        elif difference < 0:
            rounded_values[keys_sorted[-(i + 1)]] -= 1

    return rounded_values


def main():
    # Login to Robinhood
    login_robinhood()

    # Define the ETF configuration (symbols and percentages)
    etf_config = {'MSFT': 12, 'NVDA': 33, 'META': 0, 'AMZN': 1, 'QCOM': 6, 'GOOG': 5, 'GOOGL': 7, 'AVGO': 21, 'COST': 10, 'AAPL': 5}

    # Adjust ETF configuration to ensure it sums to 100%
    adjusted_etf_config = adjust_etf_config(etf_config)

    # Total amount to invest
    total_investment = 250

    total_invested = 0

    # Calculate and buy stocks
    for symbol, percentage in adjusted_etf_config.items():
        amount_to_invest = round(total_investment * (percentage / 100), 2)
        logging.info(f"Investing ${amount_to_invest} in {symbol}")
        buy_stock(symbol, amount_to_invest)
        total_invested += amount_to_invest

    logging.info(f"Total invested: ${total_invested}")


if __name__ == "__main__":
    main()
