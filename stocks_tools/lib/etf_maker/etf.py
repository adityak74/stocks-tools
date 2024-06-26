"""ETF Maker"""

import logging

import yfinance as yf
from typing import List


class ETF:
    """ETF Class"""

    def __init__(self, name="My ETF"):
        """Initialize ETF Class"""
        self.name = name
        self.symbol_percents = []
        self.total_percentage = 0.0

    def add_symbol_and_percent(self, symbol, percent):
        """Add symbol and percent to ETF"""
        if not self.check_if_valid_symbol(symbol):
            raise ValueError("Invalid symbol")
        if not (isinstance(percent, int | float) and percent > 0):
            raise ValueError("Invalid percent value")
        self.symbol_percents.append(
            {"symbol": symbol, "percent": percent, "ticker": yf.Ticker(symbol)}
        )
        self.total_percentage += percent

    def get_symbol_percents(self):
        """Get Symbol Percentages"""
        return self.symbol_percents

    def get_total_percentage(self):
        """Get Total Percentage"""
        return self.total_percentage

    @staticmethod
    def check_if_valid_symbol(symbol):
        """Check whether symbol is valid"""
        try:
            _ = yf.Ticker(symbol).info
            return True
        except Exception as e:
            logging.exception(e)
            return False


class ETFMaker:
    """ETF Maker Class"""

    def __init__(self):
        """Initialize ETF Maker"""
        self.etf_config = {}
        self.etfs = []

    def validate_etf_config(self, etf_config: dict):
        """Validate ETF config"""
        etf = ETF("Custom ETF")
        for symbol, percent in etf_config.items():
            etf.add_symbol_and_percent(symbol, percent)
        if etf.total_percentage != 100.0:
            raise ValueError("ETF config should add up to 100%")
        self.etfs.append(etf)
        return self.etfs

    def generate_custom_etf(self, etf) -> List[ETF]:
        """Generate custom ETF Maker"""
        if not isinstance(etf, dict):
            raise ValueError("Etf must be dict with symbols and percentage values")
        etfs = self.validate_etf_config(etf)
        return etfs
