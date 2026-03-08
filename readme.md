# Stock Analysis Tool
A command-line application that fetches historical stock data and runs predictive analysis using two strategies: an LSTM neural network and technical indicator regression.

## How It Works

The user enters a stock ticker symbol (validated via the Finnhub API), and the app pulls six years of historical price data using yfinance, then the user picks one of two strategies to analyze the stock:

**Strategy 1  LSTM Price Forecast:** Trains a Long Short-Term Memory recurrent neural network on the stock's closing prices. The data is scaled to [0, 1], split 80/20 into train and test sets, and fed into the network using a 40-day sliding window. After training for 100 epochs, the best model (by validation loss) is loaded and used to generate a 30-day forward price forecast. Results are plotted as actual vs. predicted using Plotly.

**Strategy 2 Technical Indicator:** Uses technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator, ATR) to predict the next day's closing price via rolling OLS regression. Each indicator's predictive accuracy is measured with MAE and MSE. Results are plotted using Plotly.
### Prerequisites
- A [Finnhub](https://finnhub.io/) API key (free)
### Installation

```bash
git clone <StcokAnalysisTool>
cd <StcokAnalysisTool>
pip install -r requirements.txt
```

Create a .env file in the /src directory:
```
API_KEY=FinnhubAPIkey
```

Alternatively, you can paste your API key directly in main.py:
```python
finnhub_client = finnhub.Client("FinnhubAPIkey")
```
  
### Running
```
python main.py
```

Follow the prompts to enter a stock symbol and choose a strategy.
## Disclaimer
This tool is for educational and experimental purposes only. It is not financial advice. Stock predictions from these models should not be used to make real trading decisions.
