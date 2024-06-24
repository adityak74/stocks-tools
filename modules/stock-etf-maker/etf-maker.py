import sys
import robin_stocks.robinhood as rh
from dotenv import dotenv_values
import logging
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
cfg = dotenv_values(".env")

DEBUG = False


def print_log(*args, **kwargs):
    if DEBUG or kwargs.get('should_print', False):
        kwargs.pop("should_print", None)
        print(*args, **kwargs)


def generate_integer_weights(n, total=100, min_thresholds=None, max_thresholds=None):
    min_thresholds = np.zeros(n) if min_thresholds is None else list(min_thresholds)
    max_thresholds = np.ones(n) * total if max_thresholds is None else list(max_thresholds)

    weights = np.random.rand(n)
    weights = (weights / weights.sum()) * total
    rounded_weights = np.round(weights).astype(int)

    difference = total - rounded_weights.sum()
    for _ in range(abs(difference)):
        index = np.random.randint(0, n)
        if difference > 0 and rounded_weights[index] < max_thresholds[index]:
            rounded_weights[index] += 1
        elif difference < 0 and rounded_weights[index] > min_thresholds[index]:
            rounded_weights[index] -= 1

    return np.clip(rounded_weights, min_thresholds, max_thresholds).astype(int)


def calculate_fitness(weights, etf_config, historical_prices_map):
    total_investment = 1000
    etf_prices = np.zeros(len(list(historical_prices_map.values())[0]))
    investment_values = np.zeros_like(etf_prices)

    for i, (symbol, _) in enumerate(etf_config.items()):
        prices_np_array = historical_prices_map[symbol]
        etf_prices += prices_np_array
        amount_invested = total_investment * (weights[i] / 100)
        num_shares = amount_invested / prices_np_array[0]
        investment_values += num_shares * prices_np_array

    profit = investment_values[-1] - total_investment
    return profit


def mutate_weights_and_maintain_sum_constraint(weights, min_thresholds, max_thresholds, total=100):
    n = len(weights)
    min_thresholds = list(min_thresholds)
    max_thresholds = list(max_thresholds)
    index = np.random.randint(0, n)
    value = np.random.choice([-1, 1])
    new_weights = weights.copy()
    new_weights[index] += value
    difference = total - new_weights.sum()
    new_weights[index] += difference

    return np.clip(new_weights, min_thresholds, max_thresholds).astype(int)


def maximize_etf_function(etf_config, historical_prices_map, min_thresholds, max_thresholds):
    n_iters = 500
    generations = 500
    fitness_values = []
    n = len(etf_config)
    weights = generate_integer_weights(n, min_thresholds=min_thresholds.values(),
                                       max_thresholds=max_thresholds.values())
    best_weights = weights.copy()

    for i in range(n_iters):
        current_fitness = calculate_fitness(weights, etf_config, historical_prices_map)
        best_fitness = calculate_fitness(best_weights, etf_config, historical_prices_map)

        if current_fitness > best_fitness:
            best_weights = weights.copy()

        if i % 25 == 0:
            weights = mutate_weights_and_maintain_sum_constraint(weights, min_thresholds.values(),
                                                                 max_thresholds.values())

        for j in range(generations):
            fitness = calculate_fitness(weights, etf_config, historical_prices_map)
            fitness_values.append(fitness)
            new_weights = generate_integer_weights(n, min_thresholds=min_thresholds.values(),
                                                   max_thresholds=max_thresholds.values())
            new_fitness = calculate_fitness(new_weights, etf_config, historical_prices_map)

            if new_fitness > fitness:
                weights = new_weights

    optimized_etf_config = {symbol: weight for symbol, weight in zip(etf_config.keys(), best_weights)}
    print_log("Optimized ETF Config:", optimized_etf_config, should_print=True)
    optimized_profit = calculate_fitness(best_weights, etf_config, historical_prices_map)
    print_log("Optimized ETF Price:", optimized_profit, should_print=True)

    plt.figure(figsize=(14, 7))
    plt.plot(fitness_values)
    plt.title("Fitness Values Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness Value")
    plt.show()


def login_robinhood():
    rh.login(cfg["ROBINHOOD_USERNAME"], cfg["ROBINHOOD_PASSWORD"])


@lru_cache(maxsize=128)
def get_stock_price(symbol):
    print_log(f"Fetching historical prices for {symbol}")
    return rh.stocks.get_stock_historicals(symbol, span="3month", bounds="regular")


def extract_close_prices(historical_prices):
    return [float(price["close_price"]) for price in historical_prices]


def plot_etf_prices(etf_prices, investment_values):
    plt.figure(figsize=(14, 7))
    plt.subplot(2, 1, 1)
    plt.title("Custom ETF Chart (last 3 months closing price)")
    (line1,) = plt.plot(etf_prices, label="Average ETF Prices")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title("Investment Value Over Time")
    (line2,) = plt.plot(investment_values, label="Investment Value")
    plt.legend()

    cursor1 = mplcursors.cursor(line1, hover=True)
    cursor1.connect("add", lambda sel: sel.annotation.set_text(f"Price: {etf_prices[int(sel.index)]:.2f}"))

    cursor2 = mplcursors.cursor(line2, hover=True)
    cursor2.connect("add", lambda sel: sel.annotation.set_text(f"Value: ${investment_values[int(sel.index)]:.2f}"))

    plt.tight_layout()
    plt.show()


def main():
    login_robinhood()
    etf_config = {
        "MSFT": 30,
        "NVDA": 25,
        "META": 10,
        "AMZN": 5,
        "QCOM": 5,
        "GOOG": 5,
        "GOOGL": 5,
        "AVGO": 5,
        "COST": 5,
        "AAPL": 5
    }

    min_thresholds = {
        "MSFT": 10,
        "NVDA": 30,
        "META": 2,
        "AMZN": 2,
        "QCOM": 2,
        "GOOG": 2,
        "GOOGL": 2,
        "AVGO": 2,
        "COST": 2,
        "AAPL": 2
    }

    max_thresholds = {
        "MSFT": 30,
        "NVDA": 50,
        "META": 10,
        "AMZN": 10,
        "QCOM": 10,
        "GOOG": 10,
        "GOOGL": 10,
        "AVGO": 10,
        "COST": 10,
        "AAPL": 10
    }

    total_investment = 200
    etf_prices = None
    investment_values = np.zeros(len(get_stock_price("AAPL")))
    historical_prices_map = {}
    investment_values_array = []

    for symbol, percentage in etf_config.items():
        symbol_historical_price = get_stock_price(symbol)
        prices_np_array = np.array(extract_close_prices(symbol_historical_price))
        historical_prices_map[symbol] = prices_np_array
        etf_prices = prices_np_array if etf_prices is None else etf_prices + prices_np_array
        amount_invested = total_investment * (percentage / 100)
        num_shares = amount_invested / prices_np_array[0]
        investment_values += num_shares * prices_np_array
        investment_values_array.append(investment_values)

    etf_prices /= len(etf_config)
    profit = investment_values[-1] - total_investment

    print_log(f"Total investment: ${total_investment}", should_print=True)
    print_log(f"Total profit: ${profit:.2f}", should_print=True)

    maximize_etf_function(etf_config, historical_prices_map, min_thresholds, max_thresholds)


if __name__ == "__main__":
    main()
