import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def get_stock_data(ticker, period="2y", interval="1d"):
    """
    Fetch historical stock data using yfinance.
    
    Args:
        ticker (str): Stock symbol (e.g., 'NVDA', '2330.TW')
        period (str): Data period (e.g., '1y', '2y', '5y', 'max')
        interval (str): Data interval (e.g., '1d', '1wk', '1mo')
        
    Returns:
        pd.DataFrame: Historical stock data
    """
    print(f"--- Fetching data for {ticker} (Period: {period}, Interval: {interval}) ---")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    
    if df.empty:
        print(f"❌ No data found for {ticker}")
        return None
        
    print(f"✅ Successfully fetched {len(df)} rows of data.")
    return df

def prepare_for_timesfm(df, column='Close'):
    """
    Prepare the dataframe for TimesFM input.
    TimesFM expects a list of 1D numpy arrays.
    """
    if df is None or column not in df.columns:
        return None
    
    # Extract the target column as a numpy array
    data_array = df[column].values.astype(np.float32)
    return data_array

def get_latest_stock_data(ticker, interval="1m"):
    """
    Fetch the most recent stock data.
    
    Args:
        ticker (str): Stock symbol
        interval (str): Data interval for real-time (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h)
        
    Returns:
        pd.DataFrame: Latest stock data
    """
    print(f"--- Fetching latest data for {ticker} (Interval: {interval}) ---")
    stock = yf.Ticker(ticker)
    # For 1m interval, max period is 7d
    period = "1d" if interval == "1m" else "5d"
    df = stock.history(period=period, interval=interval)
    
    if df.empty:
        print(f"❌ No real-time data found for {ticker}")
        return None
        
    latest_price = df['Close'].iloc[-1]
    latest_volume = df['Volume'].iloc[-1]
    latest_time = df.index[-1]
    
    print(f"✅ Latest Data ({latest_time}): Price: {latest_price:.2f}, Volume: {latest_volume}")
    return df

import requests

def search_stock_tickers(query):
    """
    Search for stock tickers and names using Yahoo Finance search API.
    """
    if not query or len(query) < 2:
        return []
    
    url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        suggestions = []
        for quote in data.get('quotes', []):
            symbol = quote.get('symbol')
            shortname = quote.get('shortname', quote.get('longname', ''))
            exchange = quote.get('exchange', '')
            if symbol:
                suggestions.append({
                    "symbol": symbol,
                    "name": shortname,
                    "exchange": exchange,
                    "display": f"{symbol} - {shortname} ({exchange})"
                })
        return suggestions
    except Exception as e:
        print(f"Error searching tickers: {e}")
        return []

if __name__ == "__main__":
    # Test with NVDA (NVIDIA)
    ticker = "NVDA"
    
    # 1. Fetch Historical (Daily)
    hist_df = get_stock_data(ticker, period="1mo")
    
    # 2. Fetch Latest (Intraday)
    latest_df = get_latest_stock_data(ticker, interval="1m")
    
    if latest_df is not None:
        print("\nLast 5 minutes of intraday data:")
        print(latest_df[['Close', 'Volume']].tail())
