import os
import requests
import pandas as pd
import numpy as np
import torch
import timesfm
from stock_data_loader import get_stock_data, prepare_for_timesfm
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm
import time

# GPU memory management
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"

import os
import requests
import pandas as pd
import numpy as np
import torch
import timesfm
from stock_data_loader import get_stock_data, prepare_for_timesfm
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm
import time
import io

# GPU memory management - use dynamic allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# Leave some room just in case
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".80"

def get_sp500_tickers():
    """Fetch all S&P 500 tickers from Wikipedia with User-Agent."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        # Use StringIO to wrap the HTML string for read_html
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]
        return df['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching S&P 500: {e}")
        # Fallback list if Wiki fails
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BRK.B", "JPM", "V"]

def get_crypto_top_n(n=20):
    """Fetch top N crypto tickers."""
    # Common crypto tickers on Yahoo Finance
    base_crypto = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "TRX-USD", 
                   "LINK-USD", "MATIC-USD", "LTC-USD", "BCH-USD", "SHIB-USD", "ATOM-USD", "XLM-USD", "UNI-USD", "HBAR-USD", "NEAR-USD"]
    return base_crypto[:n]

def get_taiwan_top_tickers():
    """Fetch a larger set of Taiwan tickers (Top market cap)."""
    # 0050 constituents
    tw_list = [f"{i:04d}.TW" for i in [
        2330, 2317, 2454, 2308, 2303, 2881, 2882, 3008, 2412, 1301,
        1303, 1326, 2886, 2002, 2912, 2382, 2357, 5880, 2891, 2892,
        2880, 2884, 2885, 3045, 2408, 2327, 2395, 3711, 2603, 2609
    ]]
    return tw_list

def run_sector_evaluation(ticker_list, sector_name, test_days=10, limit=30):
    """
    Evaluate TimesFM across a whole sector.
    """
    print(f"\n--- Evaluating Sector: {sector_name} (Sample Limit: {limit}) ---")
    
    # Load model once
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    
    results = []
    # Limit the list for speed
    tickers_to_test = ticker_list[:limit]
    
    for ticker in tqdm(tickers_to_test, desc=f"Testing {sector_name}"):
        try:
            df = get_stock_data(ticker, period="1y", interval="1d")
            if df is None or len(df) <= test_days:
                continue
            
            context_data = prepare_for_timesfm(df.iloc[:-test_days])
            actual_prices = df.iloc[-test_days:]['Close'].values
            
            model.compile(timesfm.ForecastConfig(
                max_context=1024, max_horizon=test_days+5, normalize_inputs=True
            ))
            
            point_forecast, _ = model.forecast(horizon=test_days, inputs=[context_data])
            predicted_prices = point_forecast[0]
            
            mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
            results.append(mape)
            
        except Exception as e:
            # print(f"Error {ticker}: {e}")
            continue
            
    if not results:
        return None
        
    return {
        "Sector": sector_name,
        "Mean MAPE": np.mean(results),
        "Median MAPE": np.median(results),
        "Std Dev": np.std(results),
        "Sample Size": len(results)
    }

if __name__ == "__main__":
    # 1. S&P 500
    sp500_list = get_sp500_tickers()
    sp500_res = run_sector_evaluation(sp500_list, "S&P 500", limit=30)
    
    # 2. Taiwan Stock (TW Top 50 sample)
    tw_list = get_taiwan_top_tickers()
    tw_res = run_sector_evaluation(tw_list, "Taiwan Top Cap", limit=10)
    
    # 3. Crypto
    crypto_list = get_crypto_top_n(20)
    crypto_res = run_sector_evaluation(crypto_list, "Crypto (Top 10)", limit=10)
    
    # Final Summary Table
    all_results = [r for r in [sp500_res, tw_res, crypto_res] if r is not None]
    summary_df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print("SECTOR-WIDE ACCURACY SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
