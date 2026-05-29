import argparse
import os
import time
import json
import pandas as pd
from twelvedata import TDClient
from datetime import datetime

class BulkDownloader:
    def __init__(self, api_key, output_dir='data_cache', rate_limit=8):
        self.td = TDClient(apikey=api_key)
        self.output_dir = output_dir
        self.rate_limit = rate_limit
        self.calls_this_minute = 0
        self.start_time = time.time()
        os.makedirs(output_dir, exist_ok=True)
        
        # Path for ignore list (symbols that have no volume or no data)
        self.ignore_path = os.path.join(os.path.dirname(__file__), 'ignore_list.txt')
        self.ignore_list = self._load_ignore_list()

    def _load_ignore_list(self):
        if os.path.exists(self.ignore_path):
            with open(self.ignore_path, 'r') as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def _add_to_ignore_list(self, symbol):
        if symbol not in self.ignore_list:
            self.ignore_list.add(symbol)
            with open(self.ignore_path, 'a') as f:
                f.write(f"{symbol}\n")
            print(f"Added {symbol} to ignore list.")

    def _wait_for_rate_limit(self):
        """Throttles requests to stay within the 8 calls/minute limit."""
        self.calls_this_minute += 1
        if self.calls_this_minute >= self.rate_limit:
            elapsed = time.time() - self.start_time
            if elapsed < 60:
                sleep_time = 61 - elapsed
                print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            self.calls_this_minute = 0
            self.start_time = time.time()

    def fetch_full_history(self, symbol, interval='1h'):
        """Fetches all available historical data for a symbol by paging backwards."""
        if symbol in self.ignore_list:
            print(f"Skipping {symbol}: In ignore list.")
            return True

        clean_symbol = symbol.replace('/', '_').replace(':', '_')
        file_path = os.path.join(self.output_dir, f"{clean_symbol}_{interval}_full.csv")
        
        # Check if file already exists
        if os.path.exists(file_path):
            print(f"Skipping {symbol}: {file_path} already exists.")
            return True
        
        all_data = []
        last_timestamp = None
        
        print(f"\n--- Starting Full Fetch for {symbol} ---")
        
        while True:
            self._wait_for_rate_limit()
            
            try:
                params = {
                    "symbol": symbol,
                    "interval": interval,
                    "outputsize": 5000,
                    "order": "DESC"
                }
                if last_timestamp:
                    params["end_date"] = last_timestamp
                
                ts = self.td.time_series(**params)
                df = ts.as_pandas()
                
                if df is None or df.empty:
                    print(f"No more data found for {symbol}.")
                    break
                
                current_last_timestamp = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                if current_last_timestamp == last_timestamp:
                    break
                
                all_data.append(df)
                last_timestamp = current_last_timestamp
                
                print(f"  Downloaded {len(df)} rows. Last date: {last_timestamp}")
                
                if len(df) < 4900:
                    break
                
            except Exception as e:
                err_msg = str(e)
                print(f"Error fetching {symbol}: {err_msg}")
                if "rate limit" in err_msg.lower():
                    time.sleep(60)
                    continue
                elif "not found" in err_msg.lower() or "not available" in err_msg.lower():
                    self._add_to_ignore_list(symbol)
                    return False
                else:
                    break

        if all_data:
            final_df = pd.concat(all_data).sort_index()
            final_df = final_df[~final_df.index.duplicated(keep='first')]
            
            # Check for volume
            if 'volume' not in final_df.columns or final_df['volume'].sum() == 0:
                print(f"Skipping {symbol}: No volume data found.")
                self._add_to_ignore_list(symbol)
                return False
            
            final_df.to_csv(file_path)
            print(f"Successfully saved {len(final_df)} rows to {file_path}")
            return True
        else:
            # If we went through the loop and got nothing, it's a dead symbol
            self._add_to_ignore_list(symbol)
            return False

def main():
    parser = argparse.ArgumentParser(description='Automated bulk data downloader for Twelve Data')
    parser.add_argument('--api_key', type=str, required=True, help='Twelve Data API Key')
    parser.add_argument('--interval', type=str, default='1h', help='Time interval')
    
    args = parser.parse_args()
    
    # 1. Load existing symbols from symbols.json
    existing_symbols = set()
    symbols_json_path = os.path.join(os.path.dirname(__file__), 'symbols.json')
    if os.path.exists(symbols_json_path):
        with open(symbols_json_path, 'r') as f:
            symbols_config = json.load(f)
            for category in symbols_config:
                existing_symbols.update(symbols_config[category])
        print(f"Loaded {len(existing_symbols)} symbols from symbols.json")
    
    # 2. Read symbols from companiesmarketcap.csv and filter for simple symbols
    csv_path = os.path.join(os.path.dirname(__file__), 'companiesmarketcap.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df_csv = pd.read_csv(csv_path)
    
    # -------------------------------------------------------------------------------------
    # Filter out symbols with special characters (like '.', '-') which often indicate 
    # international stocks or complex share classes that may cause API errors.
    df_csv['Company Code'] = df_csv['Company Code'].astype(str)
    simple_symbols_df = df_csv[~df_csv['Company Code'].str.contains(r'[\.\-]', regex=True)]
    csv_symbols = simple_symbols_df['Company Code'].tolist()
    
    print(f"Filtered for simple symbols: {len(simple_symbols_df)} out of {len(df_csv)} companies.")
    # -------------------------------------------------------------------------------------
    
    # 3. Filter list
    downloader = BulkDownloader(args.api_key)
    fetch_list = [s for s in csv_symbols if s not in existing_symbols and s not in downloader.ignore_list]
    
    print(f"Total symbols in CSV: {len(csv_symbols)}")
    print(f"Already in symbols.json: {len(existing_symbols)}")
    print(f"Already in ignore list: {len(downloader.ignore_list)}")
    print(f"Remaining to check: {len(fetch_list)}")

    for symbol in fetch_list:
        success = downloader.fetch_full_history(symbol, args.interval)
        if not success:
            # Errors or No Volume handled inside fetch_full_history
            pass
            
    print("\nBulk download task completed.")

if __name__ == "__main__":
    main()
