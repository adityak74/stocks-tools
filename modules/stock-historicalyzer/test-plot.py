import matplotlib.pyplot as plt
import json

# Load the JSON data from the context file
with open('RIVN_20240605.json') as f:
    data = json.load(f)

# Extract the open prices for the stock "RIVN"
open_prices = [d['open_price'] for d in data if d['symbol'] == 'RIVN']

# Plot the graph of all the open prices over time
plt.plot(open_prices)
plt.xlabel('Time')
plt.ylabel('Open Price')
plt.title('Graph of Open Prices for Stock RIVN Over Time')
plt.show()