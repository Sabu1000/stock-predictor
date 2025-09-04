from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import ta
import os
import yfinance as yf
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

app = Flask(__name__)

def download_stock_data(ticker):
    """Download stock data using yfinance if CSV doesn't exist"""
    csv_path = f'../data/{ticker.lower()}.csv'
    
    if os.path.exists(csv_path):
        return csv_path  # File already exists
    
    try:
        print(f"Downloading {ticker} data...")
        start_date = '2018-01-01'
        end_date = '2024-12-31'
        
        # Download data using yfinance
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Keep relevant columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Create data directory if it doesn't exist
        os.makedirs('../data', exist_ok=True)
        
        # Save to CSV
        data.to_csv(csv_path)
        print(f"Successfully downloaded and saved {ticker} data to {csv_path}")
        
        return csv_path
        
    except Exception as e:
        raise ValueError(f"Failed to download data for {ticker}: {str(e)}")

def create_sequences(data, seq_length):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Predict Close price
    return np.array(X), np.array(y)

def train_model_if_needed(ticker, df):
    """Train model and create data files if they don't exist"""
    ticker_lower = ticker.lower()
    
    # Check if files exist
    files_needed = [
        f'../data/X_test_{ticker_lower}.npy',
        f'../data/y_test_{ticker_lower}.npy',
        f'../data/scaler_{ticker_lower}.pkl',
        f'../models/lstm_model_{ticker_lower}.h5'
    ]
    
    if all(os.path.exists(f) for f in files_needed):
        return  # All files exist, no need to train
    
    print(f"Training model for {ticker}...")
    
    # Prepare data for LSTM
    features = ['Close', 'SMA_20', 'RSI', 'MACD', 'Close_lag1', 'Close_lag2', 'Close_lag3']
    data = df[features].values
    
    # Scale the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences
    seq_length = 60
    X, y = create_sequences(scaled_data, seq_length)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, len(features))),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train model
    model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
    
    # Save everything
    os.makedirs('../data', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    np.save(f'../data/X_train_{ticker_lower}.npy', X_train)
    np.save(f'../data/y_train_{ticker_lower}.npy', y_train)
    np.save(f'../data/X_test_{ticker_lower}.npy', X_test)
    np.save(f'../data/y_test_{ticker_lower}.npy', y_test)
    
    joblib.dump(scaler, f'../data/scaler_{ticker_lower}.pkl')
    model.save(f'../models/lstm_model_{ticker_lower}.h5')
    
    print(f"Model training completed for {ticker}")

def predict_future_price(model, scaler, last_sequence, days_ahead):
    """Predict stock price for a future date"""
    current_sequence = last_sequence.copy()
    
    for _ in range(days_ahead):
        # Reshape for prediction
        pred_input = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        
        # Predict next value
        next_pred = model.predict(pred_input, verbose=0)
        
        # Create new sequence by rolling forward
        new_row = current_sequence[-1].copy()
        new_row[0] = next_pred[0][0]  # Update Close price
        
        # Update other features (simple approach - you could make this more sophisticated)
        # For now, we'll just shift the lagged values
        new_row[4] = new_row[0]  # Close_lag1 = current Close
        new_row[5] = current_sequence[-1, 4]  # Close_lag2 = previous Close_lag1
        new_row[6] = current_sequence[-1, 5]  # Close_lag3 = previous Close_lag2
        
        # Roll the sequence forward
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = new_row
    
    # Get the final prediction
    final_pred = current_sequence[-1, 0]
    
    # Convert back to original scale
    dummy_array = np.zeros((1, 7))
    dummy_array[0, 0] = final_pred
    future_price = scaler.inverse_transform(dummy_array)[0, 0]
    
    return future_price

@app.route("/", methods=['GET', 'POST'])
def index():
    prediction_made = False
    ticker = ''
    error_message = ''
    future_price = None
    target_date = None

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        target_date_str = request.form.get('target_date', '').strip()
        
        try:
            # Download stock data if CSV doesn't exist
            csv_path = download_stock_data(ticker)
            
            # load data and make columns 
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            # Clean the data - remove metadata rows and convert to numeric
            df = df.drop(['Ticker', 'Date'], errors='ignore')  # Remove metadata rows
            
            # Convert all columns to numeric, coercing errors to NaN
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows with NaN values (from conversion)
            df = df.dropna()
            
            # Convert index to datetime if it's not already
            try:
                df.index = pd.to_datetime(df.index)
            except:
                pass
            
            print("Data cleaned successfully!")
            print("Columns:", df.columns.tolist())
            print("Data types:", df.dtypes.tolist())
            print("Shape:", df.shape)
            print("First 5 rows:")
            print(df.head())
            
            df['SMA_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
            df['MACD'] = ta.trend.MACD(df['Close']).macd()

            df['Close_lag1'] = df['Close'].shift(1)
            df['Close_lag2'] = df['Close'].shift(2)
            df['Close_lag3'] = df['Close'].shift(3)
            df = df[['Close', 'SMA_20', 'RSI', 'MACD', 'Close_lag1', 'Close_lag2', 'Close_lag3']].dropna()

            # Train model if needed (this will create the .npy files)
            train_model_if_needed(ticker, df)

            # load test data and model
            X_test = np.load(f'../data/X_test_{ticker.lower()}.npy')
            y_test = np.load(f'../data/y_test_{ticker.lower()}.npy')
            scaler = joblib.load(f'../data/scaler_{ticker.lower()}.pkl')  # Fixed: pk1 -> pkl
            model = load_model(f'../models/lstm_model_{ticker.lower()}.h5')  # Fixed: missing slash and underscore

            # predict 
            y_pred = model.predict(X_test)

            pad = np.zeros((len(y_pred), 6))
            pred_full = np.concatenate((y_pred, pad), axis=1)
            actual_full = np.concatenate((y_test.reshape(-1, 1), pad), axis=1)

            predicted = scaler.inverse_transform(pred_full)[:, 0]
            actual = scaler.inverse_transform(actual_full)[:, 0]

            # Handle future date prediction if provided
            if target_date_str:
                try:
                    target_date = datetime.strptime(target_date_str, '%m/%d/%y')
                    last_date = df.index[-1]
                    
                    if isinstance(last_date, str):
                        last_date = pd.to_datetime(last_date)
                    
                    days_ahead = (target_date - last_date).days
                    
                    if days_ahead > 0:
                        # Get the last sequence from the data
                        features = ['Close', 'SMA_20', 'RSI', 'MACD', 'Close_lag1', 'Close_lag2', 'Close_lag3']
                        last_data = df[features].tail(60).values  # Get last 60 days
                        scaled_last_data = scaler.transform(last_data)
                        
                        future_price = predict_future_price(model, scaler, scaled_last_data, days_ahead)
                        target_date = target_date_str
                    else:
                        error_message = "Target date must be in the future"
                except ValueError:
                    error_message = "Invalid date format. Use MM/DD/YY (e.g., 07/21/25)"

            # Plot final
            plt.figure(figsize=(12,6))
            plt.plot(actual, label='Actual')
            plt.plot(predicted, label='Predicted')
            plt.title(f'{ticker} Price Prediction (2024)')
            plt.xlabel('Days')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            os.makedirs('static', exist_ok=True)
            plt.savefig('static/prediction.png')
            plt.close()

            prediction_made = True
        except Exception as e:
            error_message = str(e)  

        return render_template('index.html', rediction=prediction_made, ticker=ticker, error=error_message, future_price=future_price, target_date=target_date)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)