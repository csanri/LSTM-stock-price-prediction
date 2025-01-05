from pyti.relative_strength_index import relative_strength_index as rsi

# Adding more indicators for better predictions
def make_indicators(stock_data):
    df = stock_data.copy()

    df["SMA_30"] = df["Close"].rolling(window=30).mean()
    df["SMA_14"] = df["Close"].rolling(window=14).mean()

    df["STD_30"] = df["Close"].rolling(window=30).std()
    df["STD_14"] = df["Close"].rolling(window=14).std()

    df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()

    # df["RSI_30"] = rsi(df["Close"], period=30) 

    return df

