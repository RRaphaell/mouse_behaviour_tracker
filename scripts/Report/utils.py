import base64
import webbrowser


def df_to_dict(df, col):
    data = []
    for index, row in df.iterrows():
        data.append({"segment key": row["segment key"],
                     row["segment key"]: row[col],
                     "value": round(row[col], 3)})

    return data


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'data:file/csv;base64,{b64}'
    print(href)
    return href


def download_file(df):
    url = get_table_download_link(df)
    webbrowser.open(url, new=0, autoraise=False)
