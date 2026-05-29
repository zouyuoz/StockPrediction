import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from twelvedata import TDClient

class FinancialDataset(Dataset):
    def __init__(self, df, window_length=168, horizon=24, hurdle=0.05, oversample_rate=10):
        """
        df: DataFrame containing [Open, High, Low, Close, Volume]
        window_length: Historical lookback window
        horizon: Triple barrier future window
        hurdle: 認定為暴利行情的門檻 (5%)
        oversample_rate: 暴利行情在 Dataset 中的複製倍數
        """
        self.window_length = window_length
        self.horizon = horizon
        
        # 1. Feature Engineering
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1).replace(0, np.nan))
        df['volatility'] = np.log((df['High'] / df['Low'].replace(0, np.nan)).clip(lower=1e-9))
        df['open_pos'] = np.log(df['Open'] / df['Close'].shift(1).replace(0, np.nan))
        
        if 'Volume' in df.columns and df['Volume'].notna().any() and (df['Volume'] > 0).any():
            df['sma_v20'] = df['Volume'].rolling(20).mean()
            df['rvol'] = df['Volume'] / df['sma_v20'].replace(0, np.nan)
            df['rvol'] = df['rvol'].fillna(1.0)
        else:
            df['rvol'] = 1.0
        
        df = df.replace([np.inf, -np.inf], np.nan)
        
        self.features_df = df[['log_ret', 'volatility', 'open_pos', 'rvol']].dropna()
        self.prices = df['Close'].loc[self.features_df.index]
        
        base_indices = np.arange(len(self.features_df) - window_length - horizon)
        
        # 2. 狙擊手模式：極端事件過採樣 (Extreme Events Oversampling)
        # 為了打破 100% Hold，我們預先掃描一次未來的 K 線，找出大行情
        normal_indices = []
        extreme_indices = []
        
        prices_array = self.prices.values
        for idx in base_indices:
            end_idx = idx + window_length
            current_p = prices_array[end_idx - 1]
            future_p = prices_array[end_idx : end_idx + horizon]
            
            # 計算未來的最大可能獲利空間 (不論做多做空)
            max_up = (np.max(future_p) - current_p) / current_p
            max_down = (current_p - np.min(future_p)) / current_p
            
            if max_up >= hurdle or max_down >= hurdle:
                extreme_indices.append(idx)
            else:
                normal_indices.append(idx)
                
        # 將極端行情複製放大，強迫模型面對獵物
        if extreme_indices:
            self.data_indices = normal_indices + (extreme_indices * oversample_rate)
        else:
            self.data_indices = normal_indices
            
        # 由於有重複的 index，最好預先洗牌確保 DataLoader 取 Batch 時均勻混合
        np.random.shuffle(self.data_indices)

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, idx):
        start_idx = self.data_indices[idx]
        end_idx = start_idx + self.window_length
        
        x = self.features_df.iloc[start_idx:end_idx].values.copy()
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
        print(f"Fetching {symbol} ({interval}) from Twelve Data...")
        ts = self.td.time_series(
            symbol=symbol,
            interval=interval,
            outputsize=outputsize,
            order='ASC'
        )
        df = ts.as_pandas()
        
        df = df.rename(columns={
            'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'volume': 'Volume'
        })
        
        return df