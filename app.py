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
            # Try NSE first, if it fails, it will raise an error
            try:
                df = yf.download(f"{symbol}.NS", start=start_date, end=end_date)
                if not df.empty:
                    symbol = f"{symbol}.NS"
            except:
                # If NSE fails, try without extension (for international stocks)
                df = yf.download(symbol, start=start_date, end=end_date)
        else:
            df = yf.download(symbol, start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol} in the specified date range")
        
        # Create more advanced features
        df['Returns'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['RSI'] = calculate_rsi(df['Close'])
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        
        # Create targets for different time periods
        df['Target_1d'] = df['Close'].shift(-1)
        df['Target_1w'] = df['Close'].shift(-5)
        df['Target_1m'] = df['Close'].shift(-21)
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 50:  # Require at least 50 days of data for better predictions
            raise ValueError("Insufficient data for prediction. Please select a longer date range")
        
        # Prepare features
        feature_columns = ['Close', 'Returns', 'MA5', 'MA20', 'MA50', 'Volatility', 'RSI', 'Volume_MA5']
        X = df[feature_columns].values
        y_1d = df['Target_1d'].values
        y_1w = df['Target_1w'].values
        y_1m = df['Target_1m'].values
        
        last_price = float(df['Close'].iloc[-1])
        
        return X, y_1d, y_1w, y_1m, last_price, symbol
        
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
        X, y_1d, y_1w, y_1m, last_price, symbol = prepare_stock_data(stock_symbol, start_date, end_date)
        
        # Train models for different time periods
        model_1d = LinearRegression()
        model_1w = LinearRegression()
        model_1m = LinearRegression()
        
        # Fit models
        model_1d.fit(X[:-1], y_1d[:-1])
        model_1w.fit(X[:-5], y_1w[:-5])
        model_1m.fit(X[:-21], y_1m[:-21])
        
        # Make predictions
        pred_1d = float(model_1d.predict(X[-1:])[0])
        pred_1w = float(model_1w.predict(X[-1:])[0])
        pred_1m = float(model_1m.predict(X[-1:])[0])
        
        # Calculate predicted changes
        change_1d = ((pred_1d - last_price) / last_price * 100)
        change_1w = ((pred_1w - last_price) / last_price * 100)
        change_1m = ((pred_1m - last_price) / last_price * 100)
        
        return jsonify({
            "symbol": symbol,
            "current_price": round(last_price, 2),
            "predictions": {
                "1_day": {
                    "price": round(pred_1d, 2),
                    "change": round(change_1d, 2)
                },
                "1_week": {
                    "price": round(pred_1w, 2),
                    "change": round(change_1w, 2)
                },
                "1_month": {
                    "price": round(pred_1m, 2),
                    "change": round(change_1m, 2)
                }
            },
            "start_date": start_date,
            "end_date": end_date
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
