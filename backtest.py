import os
# Prevent JAX from pre-allocating all GPU memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".40"

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timesfm
from stock_data_loader import get_stock_data, prepare_for_timesfm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from datetime import datetime

# Set precision
torch.set_float32_matmul_precision("high")

def run_backtest(ticker, test_days=14, history_period="1y"):
    """
    Perform a backtest by withholding the last `test_days` and comparing prediction to actual.
    """
    print(f"\n==== Backtesting TimesFM for {ticker} (Last {test_days} days) ====")
    
    # 1. Get historical data
    df = get_stock_data(ticker, period=history_period, interval="1d")
    if df is None or len(df) <= test_days:
        print("❌ Not enough data for backtest.")
        return
    
    # 2. Split into context and ground truth
    context_df = df.iloc[:-test_days]
    actual_df = df.iloc[-test_days:]
    
    actual_prices = actual_df['Close'].values
    
    # 3. Prepare for TimesFM
    context_data = prepare_for_timesfm(context_df, column='Close')
    
    # 4. Load and run model
    print("Running TimesFM prediction...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    
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
    
    point_forecast, quantile_forecast = model.forecast(
        horizon=test_days,
        inputs=[context_data],
    )
    
    predicted_prices = point_forecast[0]
    
    # 5. Calculate Metrics
    mae = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mape = mean_absolute_percentage_error(actual_prices, predicted_prices)
    
    print("\n--- Backtest Metrics ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2%} (Mean Absolute Percentage Error)")
    
    # 6. Visualization
    plt.figure(figsize=(15, 7))
    
    # Last 30 days of context
    hist_to_show = 30
    plt.plot(context_df.index[-hist_to_show:], context_df['Close'].iloc[-hist_to_show:], 
             label='Historical Context', color='blue')
    
    # Actual future (ground truth)
    plt.plot(actual_df.index, actual_prices, label='Actual Price (Ground Truth)', color='green', marker='o')
    
    # Predicted future
    plt.plot(actual_df.index, predicted_prices, label='Predicted Price', color='red', linestyle='--', marker='x')
    
    # Quantiles (80% confidence interval)
    quantiles = quantile_forecast[0]
    plt.fill_between(actual_df.index, quantiles[:, 1], quantiles[:, 9], 
                     color='red', alpha=0.1, label='80% Confidence Interval')
    
    plt.title(f"TimesFM Backtest: {ticker} (Last {test_days} Days)", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = f"{ticker}_backtest_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(output_file)
    print(f"\n✅ Backtest visualization saved to {output_file}")
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "actual": actual_prices,
        "predicted": predicted_prices
    }

if __name__ == "__main__":
    run_backtest("NVDA", test_days=14)
