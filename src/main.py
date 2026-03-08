import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import dataFetch
from strategies import Technical_indicator
from strategies import LSTM_forecasting
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path
from dotenv import dotenv_values
import finnhub

if __name__ == "__main__":

    config = dotenv_values(Path(__file__).parent / ".env")
    finnhub_client = finnhub.Client(api_key=config["API_KEY"])

    def runStrategy(stock):
        print("valid stock symbol, please choose a strategy by typing its number\n" \
            "1: LSTM Price Forecast\n" \
            "2: Best Technical Indicator")
        i = int(input())
        while True:
            match i:
                case 1:
                    LSTM_forecasting.run_analysis(stock)
                    return
                case 2:
                    Technical_indicator.run_TI(stock)
                    return
                case _:
                    print("please pick a valid strategy\n" \
                    "1: LSTM Price Forecast\n" \
                    "2: Best Technical Indicator")
                    i = int(input())            

    def check_symbol():
        print("Put the stock symbol that you want to use for this test:")
        stock = input().upper()
        while True:
            result = finnhub_client.symbol_lookup(stock)
            if result["count"] > 0 and result["result"][0]["symbol"] == stock:
                dataFetch.fetch_data(stock)
                return runStrategy(stock)

            print("Invalid stock symbol, please retry.")
            stock = input().upper()

    check_symbol()