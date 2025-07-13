import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os

# set settings
stock_symbol = 'AAPL'
start_date = '2018-01-01'
end_date = '2024-12-31'
data_path = 'data/aapl.csv'

# download stock data
print(f"Downloading {stock_symbol} data from {start_date} to {end_date}...")
data = yf.download(stock_symbol, start=start_date, end=end_date) # this allow us to see all apple stock data

# keep revelent columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# save data to csv file
os.makedirs("data", exist_ok=True)
data.to_csv(data_path)
print(f"Saved to {data_path}")

# plot closing price
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price')
plt.title(f"{stock_symbol} Stock Price")
plt.xlabel("Data")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

