import pandas as pd
import torch

df = pd.read_parquet("data/stockformer_raw.parquet")

tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "BAC", "GS", "XOM", "CVX", "JNJ", "LLY", "UNH", "WMT", "COST", "HD", "CAT", "GE", "NEE", "DUK", "KO", "PEP"]

sequence_length = 20
prediction_length = 2

dfs = {}

for ticker in tickers:
    indiv_df = pd.DataFrame()

    for index, row in df.iterrows():
        if row["symbol"] == ticker:
            indiv_df = pd.concat(
                [indiv_df, row.to_frame().T],
                ignore_index=True
            )

    indiv_df = indiv_df.sort_values("datetime").reset_index(drop=True)

    indiv_df["return"] = indiv_df["close"].pct_change()
    indiv_df["trend"] = (indiv_df["return"] > 0).astype("float32")

    for i in range(60):
        indiv_df["CLOSE" + str(i)] = indiv_df["close"].shift(i) / indiv_df["close"]
        indiv_df["OPEN" + str(i)] = indiv_df["open"].shift(i) / indiv_df["close"]
        indiv_df["HIGH" + str(i)] = indiv_df["high"].shift(i) / indiv_df["close"]
        indiv_df["LOW" + str(i)] = indiv_df["low"].shift(i) / indiv_df["close"]
        indiv_df["VWAP" + str(i)] = indiv_df["vwap"].shift(i) / indiv_df["close"]
        indiv_df["VOLUME" + str(i)] = indiv_df["volume"].shift(i) / indiv_df["volume"]

    dfs[ticker] = indiv_df

feature_cols = ["return", "trend"]

for i in range(60):
    feature_cols.append("CLOSE" + str(i))

for i in range(60):
    feature_cols.append("OPEN" + str(i))

for i in range(60):
    feature_cols.append("HIGH" + str(i))

for i in range(60):
    feature_cols.append("LOW" + str(i))

for i in range(60):
    feature_cols.append("VWAP" + str(i))

for i in range(60):
    feature_cols.append("VOLUME" + str(i))

for ticker in tickers:
    dfs[ticker] = dfs[ticker][
        ["symbol", "datetime"] + feature_cols
    ]

    dfs[ticker] = dfs[ticker].dropna()

common_dates = set(dfs[tickers[0]]["datetime"])

for ticker in tickers:
    common_dates = common_dates.intersection(
        set(dfs[ticker]["datetime"])
    )

common_dates = sorted(common_dates)

for ticker in tickers:
    indiv_df = pd.DataFrame()

    for index, row in dfs[ticker].iterrows():
        if row["datetime"] in common_dates:
            indiv_df = pd.concat(
                [indiv_df, row.to_frame().T],
                ignore_index=True
            )

    indiv_df = indiv_df.sort_values("datetime").reset_index(drop=True)

    dfs[ticker] = indiv_df

X_stocks = {}
Y_regression_stocks = {}
Y_classifier_stocks = {}

for ticker in tickers:
    stock_df = dfs[ticker]

    stock_features = stock_df.drop(
        columns=["symbol", "datetime"]
    )

    tensor = torch.tensor(
        stock_features.to_numpy(dtype="float32"),
        dtype=torch.float32
    )

    returns = torch.tensor(
        stock_df["return"].to_numpy(dtype="float32"),
        dtype=torch.float32
    )

    trends = torch.tensor(
        stock_df["trend"].to_numpy(dtype="int64"),
        dtype=torch.long
    )

    X = []
    Y_regression = []
    Y_classifier = []

    n_sequences = (
        len(tensor)
        - sequence_length
        - prediction_length
        + 1
    )

    for i in range(n_sequences):
        X.append(
            tensor[
                i:
                i + sequence_length
            ]
        )

        Y_regression.append(
            returns[
                i + sequence_length:
                i + sequence_length + prediction_length
            ]
        )

        Y_classifier.append(
            trends[
                i + sequence_length:
                i + sequence_length + prediction_length
            ]
        )

    X = torch.stack(X)
    Y_regression = torch.stack(Y_regression)
    Y_classifier = torch.stack(Y_classifier)

    X_stocks[ticker] = X
    Y_regression_stocks[ticker] = Y_regression
    Y_classifier_stocks[ticker] = Y_classifier

X_list = []
Y_regression_list = []
Y_classifier_list = []

for ticker in tickers:
    X_list.append(X_stocks[ticker])
    Y_regression_list.append(Y_regression_stocks[ticker])
    Y_classifier_list.append(Y_classifier_stocks[ticker])

X = torch.stack(
    X_list,
    dim=2
)

Y_regression = torch.stack(
    Y_regression_list,
    dim=2
)

Y_classifier = torch.stack(
    Y_classifier_list,
    dim=2
)

torch.save(X, "/Users/gromeronaranjo/Desktop/personal_project/data/X.pt")
torch.save(Y_regression, "/Users/gromeronaranjo/Desktop/personal_project/data/Y_regression.pt")
torch.save(Y_classifier, "/Users/gromeronaranjo/Desktop/personal_project/data/Y_classifier.pt")
print("done")