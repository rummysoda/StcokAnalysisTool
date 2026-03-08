import numpy as np
import pandas_ta as ta
import pandas as pd
import pathlib
import plotly.graph_objs as go
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm

def run_TI(symbol):
    print("This model is currently training, please wait.\n"
          "estimated waiting time: ~20s")
    df = pd.read_csv(pathlib.Path(__file__).parent.parent/"fetchedData"/f'{symbol}.csv')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    df['Previous_Close'] = df['Close'].shift(1)
    df['Close_shifted'] = df['Close'].shift(1)
    df['Open_shifted'] = df['Open'].shift(1)
    df['High_shifted'] = df['High'].shift(1)
    df['Low_shifted'] = df['Low'].shift(1)

    # Calculate technical indicators on the shifted data

    # Simple Moving Average (SMA): Average price over the last 50 periods
    df['SMA_50'] = ta.sma(df['Close_shifted'], length=50)

    # Exponential Moving Average (EMA): Weighted average that reacts faster to recent price changes, using 50 periods
    df['EMA_50'] = ta.ema(df['Close_shifted'], length=50)

    # Relative Strength Index (RSI): Momentum indicator that measures the magnitude of recent price changes to evaluate overbought/oversold conditions, using a 14-period lookback
    df['RSI'] = ta.rsi(df['Close_shifted'], length=14)

    # Moving Average Convergence Divergence (MACD): Trend-following momentum indicator, using 12 and 26 periods for the fast and slow EMAs and a 9-period signal line
    macd = ta.macd(df['Close_shifted'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['Signal_Line'] = macd['MACDs_12_26_9']

    # Bollinger Bands: Volatility indicator using a 20-period moving average and 2 standard deviations
    bollinger = ta.bbands(df['Close_shifted'], length=20, std=2)
    
    df['BB_Upper'] = bollinger['BBU_20_2.0_2.0']  # Upper Bollinger Band
    df['BB_Middle'] = bollinger['BBM_20_2.0_2.0'] # Middle Band (20-period SMA)
    df['BB_Lower'] = bollinger['BBL_20_2.0_2.0']  # Lower Bollinger Band

    # Stochastic Oscillator: Momentum indicator comparing closing prices to price ranges over 14 periods with a 3-period %D moving average
    stoch = ta.stoch(df['High_shifted'], df['Low_shifted'], df['Close_shifted'], k=14, d=3)
    df['%K'] = stoch['STOCHk_14_3_3'] # %K line (main line)
    df['%D'] = stoch['STOCHd_14_3_3'] # %D line (3-period moving average of %K)

    # Average True Range (ATR): Volatility indicator measuring the average range of price movement over the last 14 periods
    df['ATR'] = ta.atr(df['High_shifted'], df['Low_shifted'], df['Close_shifted'], length=14)

    # get rid of days with missing data
    df.dropna(inplace=True)

    # Parameters
    window_size = 40

    # List of indicators to test, including Previous_Close
    indicators = ['SMA_50', 'EMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_Upper', 'BB_Middle', 'BB_Lower', '%K', '%D', 'ATR', 'Close_shifted', 'Previous_Close']

    results = {indicator: {'predictions': [], 'actual': [], 'daily_mae': []} for indicator in indicators}

    for i in range(window_size, len(df) - 1):
        train_df = df.iloc[i - window_size:i]  # Training window 
        test_index = i + 1  # Index of next day's prediction
        actual_close_price = df['Close'].iloc[test_index]  # Next day's actual closing price

        # Individual indicators as predictors (plus Previous_Close)
        for indicator in indicators[:-1]:  # Exclude Previous_Close from standalone tests
            X_train = train_df[[indicator, 'Previous_Close']]
            y_train = train_df['Close']
            X_train = sm.add_constant(X_train)  # Add constant for intercept

            model = sm.OLS(y_train, X_train).fit()
            X_test = pd.DataFrame({indicator: [df[indicator].iloc[test_index]], 'Previous_Close': [df['Previous_Close'].iloc[test_index]]})
            X_test = sm.add_constant(X_test, has_constant='add')  # Add constant for prediction

            prediction = model.predict(X_test)[0]
            results[indicator]['predictions'].append(prediction)
            results[indicator]['actual'].append(actual_close_price)
            
            daily_mae = mean_absolute_error([actual_close_price], [prediction])
            results[indicator]['daily_mae'].append(daily_mae)
    
    # Calculate accuracy metrics
    accuracy_data = {'Indicator': [],'MAE': [],'MSE': []}

    for indicator in indicators[:-1]:  # Exclude Previous_Close from standalone tests in accuracy table
        if results[indicator]['actual']:  # Check if there are results for this indicator
            mae = mean_absolute_error(results[indicator]['actual'], results[indicator]['predictions'])
            mse = mean_squared_error(results[indicator]['actual'], results[indicator]['predictions'])
            accuracy_data['Indicator'].append(indicator)
            accuracy_data['MAE'].append(mae)
            accuracy_data['MSE'].append(mse)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price',line=dict(color='#636EFA', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50',line=dict(color='#FFD700', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50',line=dict(color='#FF8C00', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='BB Upper',line=dict(color='rgba(100,149,237,0.5)', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='BB Lower',line=dict(color='rgba(100,149,237,0.5)', width=2, dash='dot'),fill='tonexty', fillcolor='rgba(100,149,237,0.1)'))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Middle'], mode='lines', name='BB Middle',line=dict(color='#6495ED', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='cyan', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name='Signal Line', line=dict(color='purple', width=2)))

    fig.update_layout(title=f"Overlay of Technical Indicators on {symbol} Close Price",xaxis_title="Days",yaxis_title="Price")
    fig.show()

    csvs = pathlib.Path(__file__).parent.parent / 'fetchedData'
    for file in csvs.glob('*.csv'):
        file.unlink()

if __name__ == "__main__":
    print("please run main.py")
    exit()