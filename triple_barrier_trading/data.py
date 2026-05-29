import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from twelvedata import TDClient

class FinancialDataset(Dataset):
    def __init__(self, df, window_length=168, horizon=24):
        """
        df: DataFrame containing [Open, High, Low, Close, Volume]
        window_length: Historical lookback window
        horizon: Triple barrier future window
        """
        self.window_length = window_length
        self.horizon = horizon
        
        # 1. Feature Engineering
        # Log Returns - Add safety for log(0)
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1).replace(0, np.nan))
        
        # Intraday Volatility - Ensure High > Low and no zero Low
        df['volatility'] = np.log((df['High'] / df['Low'].replace(0, np.nan)).clip(lower=1e-9))
        
        # Open Relative Position
        df['open_pos'] = np.log(df['Open'] / df['Close'].shift(1).replace(0, np.nan))
        
        # Relative Volume (RVOL) - Handle missing Volume data
        if 'Volume' in df.columns and df['Volume'].notna().any() and (df['Volume'] > 0).any():
            df['sma_v20'] = df['Volume'].rolling(20).mean()
            # Avoid division by zero in RVOL
            df['rvol'] = df['Volume'] / df['sma_v20'].replace(0, np.nan)
            # Fill NaNs in RVOL with 1.0 (neutral)
            df['rvol'] = df['rvol'].fillna(1.0)
        else:
            # If Volume is missing, use 1.0 as a neutral placeholder
            df['rvol'] = 1.0
        
        # Replace any remaining Inf with 0 or NaN to be dropped
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Filter and handle NaNs
        self.features_df = df[['log_ret', 'volatility', 'open_pos', 'rvol']].dropna()
        self.prices = df['Close'].loc[self.features_df.index]
        
        # We need enough data for both window and horizon
        self.data_indices = np.arange(len(self.features_df) - window_length - horizon)

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        start_idx = self.data_indices[idx]
        end_idx = start_idx + self.window_length
        
        # Features [Window, Feature_Dim]
        # Use .copy() to ensure the array is writable before converting to tensor
        x = self.features_df.iloc[start_idx:end_idx].values.copy()
        
        # Future prices for Triple Barrier calculation
        future_prices = self.prices.iloc[end_idx : end_idx + self.horizon].values.copy()
        current_price = self.prices.iloc[end_idx - 1]
        
        return {
            'x': torch.FloatTensor(x),
            'current_price': torch.FloatTensor([current_price]),
            'future_prices': torch.FloatTensor(future_prices)
        }

class TwelveDataLoader:
    def __init__(self, api_key):
        self.td = TDClient(apikey=api_key)

    def fetch_data(self, symbol, interval='1h', outputsize=5000):
        """
        Fetches data from Twelve Data and formats it for the dataset.
        """
        print(f"Fetching {symbol} ({interval}) from Twelve Data...")
        ts = self.td.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            order='ASC'
        )
        df = ts.as_pandas()
        
        # Standardize column names
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume'
        })
        
        return df
