"""ETF Maker"""
import sys
import os

# Add the directory containing your module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stocks_tools import ETFMaker

etf_maker = ETFMaker()
custom_etf = etf_maker.generate_custom_etf({
    'NVDA': 90,
    'MSFT': 10
})
print(custom_etf[0].get_symbol_percents())
