import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timesfm
from stock_data_loader import get_stock_data, prepare_for_timesfm
import os

# Set precision
torch.set_float32_matmul_precision("high")

def run_stock_forecast(ticker, forecast_horizon=7):
    """
    Fetch stock data and run TimesFM forecast for the next N days.
    """
    print(f"\n==== Starting Forecast for {ticker} ====")
    
    # 1. Get historical data (daily)
    # We take 1 year of data to provide enough context
    df = get_stock_data(ticker, period="1y", interval="1d")
    if df is None:
        return
    
    # 2. Prepare context for TimesFM
    # TimesFM works best with a good amount of history
    context_data = prepare_for_timesfm(df, column='Close')
    
    # 3. Initialize and Load Model
    print("Initializing TimesFM 2.5 Torch model...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    
    model.compile(
        timesfm.ForecastConfig(
            max_context=1024,
            max_horizon=forecast_horizon * 2, # Buffer for horizon
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )
    
    # 4. Run Forecast
    print(f"Predicting next {forecast_horizon} days...")
    # inputs expects a list of arrays
    point_forecast, quantile_forecast = model.forecast(
        horizon=forecast_horizon,
        inputs=[context_data],
    )
    
    # Reshape results (since we only passed one input)
    points = point_forecast[0] # (horizon,)
    quantiles = quantile_forecast[0] # (horizon, 10)
    
    # 5. Prepare Visualization
    print("Generating visualization...")
    plt.figure(figsize=(15, 7))
    
    # Plot last 60 days of history for context
    history_to_show = 60
    hist_subset = df.iloc[-history_to_show:]
    plt.plot(hist_subset.index, hist_subset['Close'], label='Historical Close', color='blue', linewidth=2)
    
    # Prepare forecast dates
    last_date = df.index[-1]
    forecast_dates = [last_date + pd.Timedelta(days=i+1) for i in range(forecast_horizon)]
    
    # Plot point forecast
    plt.plot(forecast_dates, points, label='TimesFM Forecast', color='red', linestyle='--', marker='o')
    
    # Plot confidence interval (Quantiles 1 and 9 represent roughly 80-90% interval)
    # Index 0 is mean, 1-9 are quantiles. Let's use 10th and 90th.
    low_bound = quantiles[:, 1]
    high_bound = quantiles[:, 9]
    plt.fill_between(forecast_dates, low_bound, high_bound, color='red', alpha=0.2, label='80% Confidence Interval')
    
    plt.title(f"{ticker} Stock Price Forecast - Next {forecast_horizon} Days", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Save the plot
    output_plot = f"{ticker}_forecast_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(output_plot)
    print(f"✅ Forecast plot saved to {output_plot}")
    
    # Print numerical forecast
    print("\nForecasted Prices:")
    for i, (date, price) in enumerate(zip(forecast_dates, points)):
        print(f"Day {i+1} ({date.strftime('%Y-%m-%d')}): {price:.2f}")

if __name__ == "__main__":
    from datetime import datetime
    # You can change the ticker here
    target_stock = "NVDA"
    run_stock_forecast(target_stock, forecast_horizon=10)
