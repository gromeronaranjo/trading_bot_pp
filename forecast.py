from mcts.mcts_algorithm import predict
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import torch

end = datetime.now()
start = end - timedelta(days=180)

stock_list = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "BAC", "GS", "XOM", "CVX", "JNJ", "LLY", "UNH", "WMT", "COST", "HD", "CAT", "GE", "NEE", "DUK", "KO", "PEP"]


def fetch_TE(dates):
    dates = pd.Series(pd.to_datetime(dates))

    day_of_week = torch.tensor(dates.dt.dayofweek.to_numpy(), dtype=torch.long)
    time_of_day = torch.zeros(len(dates), dtype=torch.long)

    TE = torch.stack([day_of_week, time_of_day], dim=-1)
    TE = TE.unsqueeze(0)

    return TE


def fetch_daily_with_vwap(ticker):
    data = yf.download(ticker, start=start, end=end, interval="1h", auto_adjust=False, prepost=False, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data["typical_price"] = (data["High"] + data["Low"] + data["Close"]) / 3
    data["price_volume"] = data["typical_price"] * data["Volume"]
    data["date"] = data.index.date

    daily = data.groupby("date").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
        price_volume=("price_volume", "sum")
    )

    daily["VWAP"] = daily["price_volume"] / daily["Volume"]
    daily = daily.drop(columns=["price_volume"])

    return daily


def extract_362_features(data):
    data = data.copy()

    data["return"] = data["Close"].pct_change()
    data["trend"] = (data["return"] > 0).astype(float)

    features = []
    dates = []

    for i in range(59, len(data)):
        row = []

        row.append(data["return"].iloc[i])
        row.append(data["trend"].iloc[i])

        current_close = data["Close"].iloc[i]
        current_volume = data["Volume"].iloc[i]

        for j in range(i, i - 60, -1):
            row.append(data["Close"].iloc[j] / current_close)

        for j in range(i, i - 60, -1):
            row.append(data["Open"].iloc[j] / current_close)

        for j in range(i, i - 60, -1):
            row.append(data["High"].iloc[j] / current_close)

        for j in range(i, i - 60, -1):
            row.append(data["Low"].iloc[j] / current_close)

        for j in range(i, i - 60, -1):
            row.append(data["VWAP"].iloc[j] / current_close)

        for j in range(i, i - 60, -1):
            row.append(data["Volume"].iloc[j] / current_volume)

        features.append(row)
        dates.append(data.index[i])

    return pd.DataFrame(features, index=dates)


stocks = {}

for ticker in stock_list:
    daily = fetch_daily_with_vwap(ticker)
    features = extract_362_features(daily)
    stocks[ticker] = features

tensors = []

for ticker in stock_list:
    df = stocks[ticker].tail(20)
    tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)
    tensors.append(tensor)

X = torch.stack(tensors)
X = X.permute(1, 0, 2)
X = X.unsqueeze(0)

dates = stocks[stock_list[0]].tail(20).index
TE = fetch_TE(dates)

predict(X, TE, 1000, [25, 50, 75], 10000, 40)