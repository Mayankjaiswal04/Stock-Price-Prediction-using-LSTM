#Set-ExecutionPolicy Unrestricted -Scope Process
#python -m venv venv
#venv\Scripts\activate
#pip install Flask pandas numpy yfinance scikit-learn tensorflow keras
#python app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def prepare_data(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size, 3])  # Close price at index 3
    return np.array(X), np.array(y)

def dataprocess(df, ticker, start_date, end_date):
    data = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA50']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    window_size = 60
    X, y = prepare_data(scaled_data, window_size)

    if len(X) == 0 or len(y) == 0:
        raise ValueError("Not enough data to train the model. Try increasing the date range.")

    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)


    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(np.hstack((np.zeros((predictions.shape[0], 5)), predictions.reshape(-1, 1))))[:, -1]

    actual_prices = df['Close'][train_size + window_size:].values
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mean_actual = np.mean(actual_prices)
    accuracy = max(0, 100 - (rmse / mean_actual) * 100)

    # Plotting
    plt.figure(figsize=(20, 10))
    plt.plot(df.index[train_size + window_size:], actual_prices, label='Actual Stock Price', color='blue', linewidth=2)
    plt.plot(df.index[train_size + window_size:], predicted_prices, label='Predicted Stock Price', linestyle='--', color='green', linewidth=2)

    plt.annotate(f"Start: {predicted_prices[0]:.2f}", xy=(df.index[train_size + window_size], predicted_prices[0]),
                 xytext=(-50, 30), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'), fontsize=12)

    plt.annotate(f"End: {predicted_prices[-1]:.2f}", xy=(df.index[-1], predicted_prices[-1]),
                 xytext=(-50, -30), textcoords='offset points', arrowprops=dict(arrowstyle='->', color='black'), fontsize=12)

    plt.title(f'{ticker} Stock Price Prediction', fontsize=18, fontweight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Stock Price', fontsize=14)
    plt.legend(fontsize=12, shadow=True, loc='upper left')
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    return predicted_prices[-1], accuracy
