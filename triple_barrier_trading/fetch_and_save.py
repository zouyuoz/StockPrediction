import argparse
import os
import pandas as pd
from data import TwelveDataLoader

def main():
    parser = argparse.ArgumentParser(description='Fetch data from Twelve Data and save to CSV')
    parser.add_argument('--api_key', type=str, required=True, help='Twelve Data API Key')
    parser.add_argument('--symbol', type=str, default='BTC/USD', help='Ticker symbol')
    parser.add_argument('--interval', type=str, default='1h', help='Time interval (1min, 5min, 1h, 1day)')
    parser.add_argument('--outputsize', type=int, default=5000, help='Number of data points (max 5000 for free plan)')
    parser.add_argument('--output_dir', type=str, default='data_cache', help='Directory to save CSV files')

    args = parser.parse_args()

    # 1. Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Initialize Loader
    loader = TwelveDataLoader(args.api_key)

    try:
        # 3. Fetch Data
        df = loader.fetch_data(args.symbol, interval=args.interval, outputsize=args.outputsize)
        
        # 4. Generate Filename
        # Clean symbol name for filename (e.g., BTC/USD -> BTC_USD)
        clean_symbol = args.symbol.replace('/', '_').replace(':', '_')
        file_path = os.path.join(args.output_dir, f"{clean_symbol}_{args.interval}.csv")
        
        # 5. Save to CSV
        df.to_csv(file_path)
        print(f"Successfully saved data to: {file_path}")
        print(f"Total rows: {len(df)}")
        print(f"Time range: {df.index[0]} to {df.index[-1]}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
