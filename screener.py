import os
import requests
import pandas as pd
import numpy as np
import torch
import timesfm
import yfinance as yf
from stock_data_loader import get_stock_data, prepare_for_timesfm
from tqdm import tqdm
from datetime import datetime
import io
import concurrent.futures

# GPU memory management
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"

def get_all_us_tickers():
    """Fetch a large list of US tickers using a more reliable method."""
    print("Fetching US Ticker list...")
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        tickers = [t.strip() for t in response.text.split('\n') if t.strip()]
        # Filter for standard equity tickers (usually 1-4 chars)
        tickers = [t for t in tickers if t.isalpha() and 1 <= len(t) <= 4]
        return sorted(list(set(tickers)))
    except Exception as e:
        print(f"Error fetching US tickers: {e}")
        return ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]

def get_all_taiwan_tickers():
    """Fetch all listed Taiwan stock tickers from TWSE/TPEx official lists."""
    print("Fetching Taiwan Ticker list from TWSE...")
    tickers = []
    try:
        # TWSE (Listed)
        url_twse = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
        response = requests.get(url_twse, timeout=10)
        response.encoding = 'big5'
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]
        for val in df[0]:
            if '　' in str(val):
                symbol = str(val).split('　')[0].strip()
                if symbol.isdigit() and len(symbol) == 4:
                    tickers.append(f"{symbol}.TW")
        
        # TPEx (OTC)
        url_tpex = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
        response = requests.get(url_tpex, timeout=10)
        response.encoding = 'big5'
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]
        for val in df[0]:
            if '　' in str(val):
                symbol = str(val).split('　')[0].strip()
                if symbol.isdigit() and len(symbol) == 4:
                    tickers.append(f"{symbol}.TWO")
                    
        return sorted(list(set(tickers)))
    except Exception as e:
        print(f"Error fetching Taiwan tickers: {e}")
        return ["2330.TW", "2317.TW", "2454.TW", "2308.TW"]

def get_comprehensive_ticker_list():
    """Aggregate all sources."""
    us = get_all_us_tickers()
    tw = get_all_taiwan_tickers()
    crypto = [f"{c}-USD" for c in ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "AVAX", "DOT", "TRX", "LINK", "MATIC"]]
    return us + tw + crypto

def screen_high_confidence_growth(tickers, forecast_horizon=14, growth_threshold=0.10):
    """
    Screen for tickers with predicted growth > threshold and high confidence.
    Optimized for massive scanning.
    """
    print(f"Initializing TimesFM for Massive Screening...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    
    print("Compiling model for current hardware...")
    model.compile(timesfm.ForecastConfig(
        max_context=1024, 
        max_horizon=forecast_horizon+5, 
        normalize_inputs=True,
        use_continuous_quantile_head=True
    ))
    
    candidates = []
    output_csv = f"full_market_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"Targeting >{growth_threshold*100}% growth. Saving results to {output_csv}")
    
    for ticker in tqdm(tickers, desc="Screening"):
        try:
            # 1. Quick liquidity & data check
            df = get_stock_data(ticker, period="1mo", interval="1d")
            if df is None or len(df) < 15:
                continue
                
            # Liquidity check
            avg_vol = df['Volume'].iloc[-5:].mean()
            avg_price = df['Close'].iloc[-5:].mean()
            turnover = avg_vol * avg_price
            
            threshold = 1_000_000 if not ticker.endswith(('.TW', '.TWO')) else 30_000_000
            if turnover < threshold:
                continue
            
            # 2. Get full context for inference
            df_full = get_stock_data(ticker, period="1y", interval="1d")
            if df_full is None: continue
            
            latest_price = df_full['Close'].iloc[-1]
            context_data = prepare_for_timesfm(df_full)
            
            # 3. Model Inference
            point_forecast, quantile_forecast = model.forecast(horizon=forecast_horizon, inputs=[context_data])
            
            predicted_end_price = point_forecast[0][-1]
            predicted_growth = (predicted_end_price - latest_price) / latest_price
            
            if predicted_growth >= growth_threshold:
                q_low = quantile_forecast[0][-1, 1]
                q_high = quantile_forecast[0][-1, 9]
                interval_width_pct = (q_high - q_low) / latest_price
                
                res = {
                    "Ticker": ticker,
                    "Current Price": round(float(latest_price), 2),
                    "Predicted Price": round(float(predicted_end_price), 2),
                    "Growth %": round(float(predicted_growth * 100), 2),
                    "Confidence Score": round(float(100 / (1 + interval_width_pct)), 2),
                    "Safety Check": "✅" if q_low > latest_price else "⚠️",
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                candidates.append(res)
                print(f"\n🔥 Found Candidate: {ticker} (+{res['Growth %']}%)")
                
                # Incremental Save
                pd.DataFrame(candidates).to_csv(output_csv, index=False)
                
        except Exception:
            continue

    return pd.DataFrame(candidates)

if __name__ == "__main__":
    full_list = get_comprehensive_ticker_list()
    final_candidates = screen_high_confidence_growth(full_list, growth_threshold=0.10)
    
    print("\n" + "="*80)
    print("FULL MARKET SCAN COMPLETE")
    print("="*80)
    if not final_candidates.empty:
        print(f"Found {len(final_candidates)} candidates.")
        print(final_candidates.to_string(index=False))
    else:
        print("No candidates found matching the criteria.")
