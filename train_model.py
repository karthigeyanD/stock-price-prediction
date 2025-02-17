import yfinance as yf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(stock_symbol="AAPL"):
    data = yf.download(stock_symbol, start="2020-01-01", end="2024-01-01")
    data["Prediction"] = data["Close"].shift(-1)
    data.dropna(inplace=True)

    X = np.array(data[["Close"]])
    y = np.array(data["Prediction"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open("stock_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("✅ Model trained and saved as stock_model.pkl")

if __name__ == "__main__":
    train_model()


