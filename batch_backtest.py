import os
# Prevent JAX from pre-allocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"

import torch
import numpy as np
import pandas as pd
import timesfm
from stock_data_loader import get_stock_data, prepare_for_timesfm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import datetime

# Set precision
torch.set_float32_matmul_precision("high")

def batch_backtest(tickers, test_days=14, history_period="1y"):
    """
    Run backtests for a list of tickers and aggregate results.
    """
    print(f"Initializing TimesFM 2.5 Torch model for batch processing...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    
    results = []
    
    for ticker in tickers:
        print(f"\n>>>> Testing {ticker} <<<<")
        try:
            # 1. Get data
            df = get_stock_data(ticker, period=history_period, interval="1d")
            if df is None or len(df) <= test_days:
                print(f"Skipping {ticker}: Not enough data.")
                continue
            
            # 2. Split
            context_df = df.iloc[:-test_days]
            actual_df = df.iloc[-test_days:]
            actual_prices = actual_df['Close'].values
            
            # 3. Predict
            context_data = prepare_for_timesfm(context_df, column='Close')
            
            model.compile(
                timesfm.ForecastConfig(
                    max_context=1024,
                    max_horizon=test_days + 10,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            
            point_forecast, _ = model.forecast(
                horizon=test_days,
                inputs=[context_data],
            )
            predicted_prices = point_forecast[0]
            
            # 4. Metrics
            mae = mean_absolute_error(actual_prices, predicted_prices)
            rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
            mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
            
            results.append({
                "Ticker": ticker,
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
                "Status": "✅ Success"
            })
            print(f"Result for {ticker}: MAPE = {mape:.2%}")
            
        except Exception as e:
            print(f"❌ Error testing {ticker}: {e}")
            results.append({"Ticker": ticker, "Status": f"❌ Error: {str(e)}"})
            
    # Aggregate results
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == "__main__":
    # Define a diverse list of tickers
    # Tech giants, ETFs, High volatility, and International (Taiwan)
    test_list = [
        "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", # US Tech
        "SPY", "QQQ",                                   # ETFs
        "2330.TW", "2454.TW", "2317.TW",                # Taiwan Tech (TSMC, MTK, Foxconn)
        "BTC-USD", "ETH-USD"                            # Crypto (as extreme cases)
    ]
    
    summary_df = batch_backtest(test_list, test_days=10)
    
    print("\n" + "="*50)
    print("BATCH BACKTEST SUMMARY")
    print("="*50)
    print(summary_df.to_string(index=False))
    
    # Save summary
    output_csv = f"batch_backtest_{datetime.now().strftime('%Y%m%d')}.csv"
    summary_df.to_csv(output_csv, index=False)
    print(f"\nDetailed summary saved to {output_csv}")
