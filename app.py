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
        
        # Download stock data
        df = yf.download(symbol, start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol} in the specified date range")
        
        # Create features
        df['Returns'] = df['Close'].pct_change()
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Create target (next day's price)
        df['Target'] = df['Close'].shift(-1)
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 30:  # Require at least 30 days of data
            raise ValueError("Insufficient data for prediction. Please select a longer date range")
        
        # Prepare features
        X = df[['Close', 'Returns', 'MA5', 'MA20']].values
        y = df['Target'].values
        
        # Convert last_price to native Python float
        last_price = float(df['Close'].iloc[-1])
        
        return X, y, last_price
        
    except Exception as e:
        raise ValueError(f"Error preparing data: {str(e)}")

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
        X, y, last_price = prepare_stock_data(stock_symbol, start_date, end_date)
        
        # Train model on all available data
        model = LinearRegression()
        model.fit(X[:-1], y[:-1])
        
        # Make prediction using the last day's data
        prediction = float(model.predict(X[-1:])[0])  # Convert numpy float to Python float
        
        # Calculate predicted change
        predicted_change = ((prediction - last_price) / last_price * 100)
        
        return jsonify({
            "symbol": stock_symbol,
            "current_price": round(last_price, 2),
            "predicted_price": round(prediction, 2),
            "predicted_change": round(predicted_change, 2),
            "start_date": start_date,
            "end_date": end_date
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
