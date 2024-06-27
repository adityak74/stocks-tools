import robin_stocks.robinhood as rh
from dotenv import dotenv_values
import logging
import numpy as np
import matplotlib.pyplot as plt
import mplcursors
from functools import lru_cache
import pygad

logging.basicConfig(level=logging.INFO)
cfg = dotenv_values(".env")

DEBUG = False
TOTAL_INVESTMENT = 100

def print_log(*args, **kwargs):
    if DEBUG or kwargs.get('should_print', False):
        kwargs.pop("should_print", None)
        print(*args, **kwargs)


def normalize_to_100(arr):
    total_sum = sum(arr)
    normalized = [(x / total_sum) * 100 for x in arr]
    rounded_normalized = [round(num, 2) for num in normalized]

    # Adjust for rounding error to ensure sum is exactly 100
    rounded_normalized[-1] += 100 - sum(rounded_normalized)

    return rounded_normalized


def calculate_fitness(solution, ga_instance, etf_config, historical_prices_map):
    solution = normalize_to_100(solution)
    etf_prices = np.zeros(len(list(historical_prices_map.values())[0]))
    investment_values = np.zeros_like(etf_prices)

    for i, (symbol, _) in enumerate(etf_config.items()):
        prices_np_array = historical_prices_map[symbol]
        etf_prices += prices_np_array
        amount_invested = TOTAL_INVESTMENT * (solution[i] / 100)
        num_shares = amount_invested / prices_np_array[0]
        investment_values += num_shares * prices_np_array

    profit = investment_values[-1] - TOTAL_INVESTMENT
    return profit


def on_generation(ga_instance):
    print_log(f"Generation: {ga_instance.generations_completed}", should_print=True)
    print_log(f"Fitness of the best solution: {ga_instance.best_solution()[1]:.2f}", should_print=True)


def login_robinhood():
    rh.login(cfg["ROBINHOOD_USERNAME"], cfg["ROBINHOOD_PASSWORD"])


@lru_cache(maxsize=128)
def get_stock_price(symbol):
    print_log(f"Fetching historical prices for {symbol}")
    return rh.stocks.get_stock_historicals(symbol, span="3month")


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
        "MSFT": 5,
        "NVDA": 5,
        "META": 5,
        "AMZN": 5,
        "QCOM": 0,
        "GOOG": 0,
        "GOOGL": 0,
        "AVGO": 0,
        "COST": 0,
        "AAPL": 0
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

    etf_prices = None
    investment_values = np.zeros(len(get_stock_price("AAPL")))
    historical_prices_map = {}
    investment_values_array = []

    for symbol, percentage in etf_config.items():
        symbol_historical_price = get_stock_price(symbol)
        prices_np_array = np.array(extract_close_prices(symbol_historical_price))
        historical_prices_map[symbol] = prices_np_array
        etf_prices = prices_np_array if etf_prices is None else etf_prices + prices_np_array
        amount_invested = TOTAL_INVESTMENT * (percentage / 100)
        num_shares = amount_invested / prices_np_array[0]
        investment_values += num_shares * prices_np_array
        investment_values_array.append(investment_values)

    etf_prices /= len(etf_config)
    profit = investment_values[-1] - TOTAL_INVESTMENT

    print_log(f"Total investment: ${TOTAL_INVESTMENT}", should_print=True)
    print_log(f"Total profit: ${profit:.2f}", should_print=True)

    # Prepare for pygad optimization
    def fitness_func(ga_instance, solution, solution_idx):
        return calculate_fitness(solution, ga_instance, etf_config, historical_prices_map)

    # Custom initialization to ensure weights sum to 100%
    def on_start(ga_instance):
        for i in range(ga_instance.population.shape[0]):
            ga_instance.population[i] = ga_instance.population[i] / np.sum(ga_instance.population[i]) * 100

    ga_instance = pygad.GA(
        num_generations=30,
        num_parents_mating=1000,
        fitness_func=fitness_func,
        sol_per_pop=2500,
        num_genes=len(etf_config),
        on_generation=on_generation,
        on_start=on_start,
        gene_space=[{"low": min_thresholds[symbol], "high": max_thresholds[symbol]} for symbol in etf_config.keys()],
        parent_selection_type="sss",
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=5,
        mutation_by_replacement=True,
        keep_elitism=250,
    )

    ga_instance.run()

    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    best_solution = normalize_to_100(best_solution)
    optimized_etf_config = {symbol: float(weight) for symbol, weight in zip(etf_config.keys(), best_solution)}
    print_log("Optimized ETF Config:", optimized_etf_config, should_print=True)
    print_log("Optimized ETF Profit:", best_solution_fitness, should_print=True)

    optimized_and_adjusted_etf_config = adjust_etf_config(optimized_etf_config)
    print("Adjusted ETF Config:", optimized_and_adjusted_etf_config)

    plt.figure(figsize=(14, 7))
    plt.plot(ga_instance.best_solutions_fitness)
    plt.title("Fitness Values Across Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    plt.show()


if __name__ == "__main__":
    main()
