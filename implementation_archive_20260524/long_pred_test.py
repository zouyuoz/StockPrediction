import os
import pandas as pd
import numpy as np
import torch
import timesfm
import yfinance as yf
from stock_data_loader import get_stock_data, prepare_for_timesfm
from screener import get_all_us_tickers
from tqdm import tqdm
from datetime import datetime, timedelta

# GPU memory management
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"

def calculate_modified_z_score(data):
    D = np.array(data)
    if len(D) < 2: return 0.0
    median = np.median(D)
    mad = np.median(np.abs(D - median))
    if mad == 0: return 0.0
    return 0.6745 * (D[-1] - median) / mad

def check_liquidity(df_window):
    if len(df_window) < 5: return 0.0
    avg_vol = df_window["Volume"].iloc[-5:].mean()
    avg_price = df_window["Close"].iloc[-5:].mean()
    return avg_vol * avg_price

def main():
    print("Initializing TimesFM for Backtesting...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    
    forecast_horizon = 10
    model.compile(timesfm.ForecastConfig(
        max_context=1024, 
        max_horizon=forecast_horizon, 
        normalize_inputs=True,
        use_continuous_quantile_head=True
    ))

    # Priority tickers from the scan
    scan_file = "full_market_scan_20260524_082928.csv"
    priority_tickers = []
    if os.path.exists(scan_file):
        priority_tickers = pd.read_csv(scan_file)["Ticker"].tolist()
    
    all_tickers = get_all_us_tickers()
    tickers = priority_tickers + [t for t in all_tickers if t not in priority_tickers]
    
    results = []
    success_count = 0
    target_success = 100
    stride = 5 # Smaller stride for more granular testing
    output_csv = "long_pred_test_result.csv"
    
    pbar = tqdm(total=target_success, desc="Successes Found")

    for ticker in tickers:
        if success_count >= target_success:
            break
            
        try:
            # Fetch 2 years of data
            df = get_stock_data(ticker, period="2y", interval="1d")
            if df is None or len(df) < 252 + forecast_horizon:
                continue
            
            liq_threshold = 1_000_000 if not ticker.endswith((".TW", ".TWO")) else 30_000_000
            window_size = 252
            
            # Sub-loop for windows
            for i in range(0, len(df) - window_size - forecast_horizon, stride):
                df_window = df.iloc[i : i + window_size]
                df_future = df.iloc[i + window_size : i + window_size + forecast_horizon]
                
                # Check liquidity
                if check_liquidity(df_window) < liq_threshold:
                    continue
                
                # Volume Z-Score Check
                vol_z = calculate_modified_z_score(df_window["Volume"].values)
                if vol_z > 20:
                    continue
                
                # Model Inference
                context_data = prepare_for_timesfm(df_window)
                point_forecast, quantile_forecast = model.forecast(horizon=forecast_horizon, inputs=[context_data])
                
                latest_price = df_window["Close"].iloc[-1]
                predicted_end_price = point_forecast[0][-1]
                predicted_growth = (predicted_end_price - latest_price) / latest_price
                
                # Trigger Conditions
                if predicted_growth >= 0.10:
                    q_low = quantile_forecast[0][-1, 1]
                    q_high = quantile_forecast[0][-1, 9]
                    interval_width_pct = (q_high - q_low) / latest_price
                    confidence_score = 100 / (1 + interval_width_pct)
                    
                    if confidence_score >= 50:
                        # TRIGGERED!
                        actual_prices = df_future["Close"].values
                        actual_max_price = np.max(actual_prices)
                        actual_growth_max = (actual_max_price - latest_price) / latest_price
                        
                        success_threshold = max(0.10, 0.8 * predicted_growth)
                        is_success = actual_growth_max >= success_threshold
                        
                        actual_end_price = actual_prices[-1]
                        actual_growth_end = (actual_end_price - latest_price) / latest_price
                        
                        res = {
                            "Ticker": ticker,
                            "Date": df_window.index[-1].strftime("%Y-%m-%d"),
                            "Current Price": round(float(latest_price), 2),
                            "Predicted Growth %": round(float(predicted_growth * 100), 2),
                            "Confidence Score": round(float(confidence_score), 2),
                            "Volume Z-Score": round(float(vol_z), 4),
                            "Actual Max Growth %": round(float(actual_growth_max * 100), 2),
                            "Actual End Growth %": round(float(actual_growth_end * 100), 2),
                            "Success Threshold %": round(float(success_threshold * 100), 2),
                            "Result": "Success" if is_success else "Fail"
                        }
                        
                        results.append(res)
                        pd.DataFrame(results).to_csv(output_csv, index=False)
                        
                        if is_success:
                            success_count += 1
                            pbar.update(1)
                            if success_count >= target_success:
                                break
                                
        except Exception:
            continue
    
    pbar.close()
    print(f"\nBacktesting complete. Results saved to {output_csv}")

if __name__ == "__main__":
    main()
