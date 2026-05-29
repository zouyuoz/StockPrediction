import pandas as pd
import numpy as np
import yfinance as yf
from stock_data_loader import get_stock_data
from screener import get_all_us_tickers
import os
from datetime import datetime

def calculate_modified_z_score(data):
    D = np.array(data)
    if len(D) < 2:
        return np.zeros_like(D)
    median = np.median(D)
    mad = np.median(np.abs(D - median))
    if mad == 0:
        return np.zeros_like(D)
    modified_z_scores = 0.6745 * (D - median) / mad
    return modified_z_scores

def main():
    tickers = []
    scan_file = "full_market_scan_20260524_082928.csv"
    if os.path.exists(scan_file):
        df_scan = pd.read_csv(scan_file)
        tickers = df_scan['Ticker'].tolist()
    
    if len(tickers) < 10:
        more_tickers = get_all_us_tickers()
        tickers.extend([t for t in more_tickers if t not in tickers])

    results = []
    found_count = 0
    target_count = 5
    threshold = 3.5
    output_file = f"volume_zscore_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    print(f"Scanning for {target_count} stocks with Vol Z-Score >= {threshold}")

    for ticker in tickers:
        try:
            df = get_stock_data(ticker, period="1y", interval="1d")
            if df is None or len(df) < 30:
                continue
            
            volumes = df['Volume'].values
            z_scores = calculate_modified_z_score(volumes)
            latest_z_score = z_scores[-1]
            
            res = {
                "Ticker": ticker,
                "Latest Volume": int(volumes[-1]),
                "Average Volume (1y)": int(np.mean(volumes)),
                "Modified Z-Score": round(float(latest_z_score), 4)
            }
            results.append(res)
            print(f"Ticker: {ticker:6} | Z-Score: {latest_z_score:8.4f} {'🔥' if latest_z_score >= threshold else ''}")
            
            if latest_z_score >= threshold:
                found_count += 1
                # Save results every time we find a hit
                pd.DataFrame(results).to_csv(output_file, index=False)
                if found_count >= target_count:
                    break
        except Exception as e:
            continue

    if not pd.DataFrame(results).empty:
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Final results saved to {output_file}")

if __name__ == "__main__":
    main()
