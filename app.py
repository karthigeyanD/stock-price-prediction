from flask import Flask, request, render_template, jsonify
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

def prepare_stock_data(symbol, start_date, end_date):
    try:
        # Convert string dates to datetime objects
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Validate dates
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        if end_date > datetime.now():
            raise ValueError("End date cannot be in the future")
        
        # Handle Indian stocks by appending .NS for NSE stocks if not already present
        if symbol.isalpha() and not any(ext in symbol for ext in ['.NS', '.BO']):
            try:
                df = yf.download(f"{symbol}.NS", start=start_date, end=end_date, interval="1h")
                if not df.empty:
                    symbol = f"{symbol}.NS"
            except:
                df = yf.download(symbol, start=start_date, end=end_date, interval="1h")
        else:
            df = yf.download(symbol, start=start_date, end=end_date, interval="1h")
        
        if df.empty:
            raise ValueError(f"No data found for {symbol} in the specified date range")
        
        # Create technical indicators
        df['Returns'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['RSI'] = calculate_rsi(df['Close'])
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        
        # Create targets for different time periods
        df['Target_1h'] = df['Close'].shift(-1)
        df['Target_4h'] = df['Close'].shift(-4)
        df['Target_1d'] = df['Close'].shift(-24)  # Assuming 24 hours in a day
        
        # Store historical data for chart
        historical_data = {
            'dates': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            'prices': df['Close'].tolist()
        }
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 24:  # Require at least 24 hours of data
            raise ValueError("Insufficient data for prediction. Please select a longer date range")
        
        # Prepare features
        feature_columns = ['Close', 'Returns', 'MA5', 'MA20', 'Volatility', 'RSI', 'Volume_MA5']
        X = df[feature_columns].values
        y_1h = df['Target_1h'].values
        y_4h = df['Target_4h'].values
        y_1d = df['Target_1d'].values
        
        last_price = float(df['Close'].iloc[-1])
        
        return X, y_1h, y_4h, y_1d, last_price, symbol, historical_data
        
    except Exception as e:
        raise ValueError(f"Error preparing data: {str(e)}")

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock_symbol = request.form['symbol']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        
        # Prepare data and train model
        X, y_1h, y_4h, y_1d, last_price, symbol, historical_data = prepare_stock_data(stock_symbol, start_date, end_date)
        
        # Train models for different time periods
        model_1h = LinearRegression()
        model_4h = LinearRegression()
        model_1d = LinearRegression()
        
        # Fit models
        model_1h.fit(X[:-1], y_1h[:-1])
        model_4h.fit(X[:-4], y_4h[:-4])
        model_1d.fit(X[:-24], y_1d[:-24])
        
        # Make predictions
        pred_1h = float(model_1h.predict(X[-1:])[0])
        pred_4h = float(model_4h.predict(X[-1:])[0])
        pred_1d = float(model_1d.predict(X[-1:])[0])
        
        # Calculate predicted changes
        change_1h = ((pred_1h - last_price) / last_price * 100)
        change_4h = ((pred_4h - last_price) / last_price * 100)
        change_1d = ((pred_1d - last_price) / last_price * 100)
        
        return jsonify({
            "symbol": symbol,
            "current_price": round(last_price, 2),
            "predictions": {
                "1_hour": {
                    "price": round(pred_1h, 2),
                    "change": round(change_1h, 2)
                },
                "4_hours": {
                    "price": round(pred_4h, 2),
                    "change": round(change_4h, 2)
                },
                "1_day": {
                    "price": round(pred_1d, 2),
                    "change": round(change_1d, 2)
                }
            },
            "historical_data": historical_data,
            "start_date": start_date,
            "end_date": end_date
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
