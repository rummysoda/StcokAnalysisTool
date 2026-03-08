from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pathlib

def fetch_data(symbol):
    now = datetime.today()
    then = datetime.today() + relativedelta(years=-6)

    start_date = then.strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    stock = yf.Ticker(symbol)
    historical_data = stock.history(start=start_date, end=end_date)
    historical_data.index = historical_data.index.tz_localize(None)
    savePath = pathlib.Path(__file__).parent / "fetchedData"
    historical_data.to_csv(savePath/ f'{symbol}.csv')

if __name__ == "__main__":
    print("please run main.py")
    exit()