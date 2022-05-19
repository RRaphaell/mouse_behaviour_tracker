import pandas as pd


def df_to_dict(df, col):
    data = []
    for index, row in df.iterrows():
        data.append({"segment key": row["segment key"],
                     row["segment key"]: row[col],
                     "value": round(row[col], 3)})

    return data
