from flask import Flask, request, jsonify, render_template
import os
import yfinance as yf
import matplotlib.pyplot as plt
from predict import dataprocess

app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('Index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        ticker, start, end = data['ticker'], data['start_date'], data['end_date']

        df = yf.download(ticker, start=start, end=end)
        if df.empty:
            return jsonify({'error': 'No data found for the given ticker and date range.'})

        df.dropna(inplace=True)
        df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()

        predicted_price, accuracy = dataprocess(df, ticker, start, end)

        # Save plot
        plt.savefig('static/prediction_plot.png')
        plt.clf()  # Clear the plot

        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'accuracy': round(accuracy, 2),
            'plot_url': '/static/prediction_plot.png'
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
