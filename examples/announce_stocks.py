"""Announce Stocks"""
import sys
import os

# Add the directory containing your module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stocks_tools import Announcer

announcer = Announcer()
announcer.announce_stock_prices(["NVDA"])
