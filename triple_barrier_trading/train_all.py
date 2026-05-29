import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import pandas as pd
import argparse
import os
import glob
from tqdm import tqdm
import random

from models import TradingPolicyNetwork
from data import FinancialDataset
from engine import triple_barrier_loss

def load_subset_datasets(file_paths):
    datasets = []
    for f in file_paths:
        try:
            symbol = os.path.basename(f).split('_')[0]
            df = pd.read_csv(f, index_col=0, parse_dates=True)
            df = df.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 
                'close': 'Close', 'volume': 'Volume',
                'Open': 'Open', 'High': 'High', 'Low': 'Low', 
                'Close': 'Close', 'Volume': 'Volume'
            })
            asset_ds = FinancialDataset(df, window_length=168, horizon=24)
            if len(asset_ds) > 0:
                datasets.append(asset_ds)
        except Exception as e:
            print(f"  Error loading {f}: {e}")
    return datasets

def train_all():
    parser = argparse.ArgumentParser(description='Universal Multi-Asset Training (Circular Loader)')
    parser.add_argument('--data_dir', type=str, default='data_cache', help='Directory containing *_full.csv files')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='universal_model.pth', help='Output model name')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--files_per_epoch', type=int, default=80, help='Total CSV files to use per epoch')
    
    args = parser.parse_args()

    csv_files = glob.glob(os.path.join(args.data_dir, "*_full.csv"))
    if not csv_files:
        print(f"No data files found in {args.data_dir}")
        return
    
    random.seed(42)
    random.shuffle(csv_files)
    num_total_files = len(csv_files)
    print(f"Found {num_total_files} asset files total.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TradingPolicyNetwork(feature_dim=4, hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_reward = -float('inf')
    reward_history = []
    current_file_ptr = 0 

    os.makedirs('checkpoints', exist_ok=True)

    if args.resume and os.path.exists('checkpoints/latest_checkpoint.pth'):
        print("Resuming from latest checkpoint...")
        checkpoint = torch.load('checkpoints/latest_checkpoint.pth', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_reward = checkpoint.get('best_val_reward', -float('inf'))
        reward_history = checkpoint.get('reward_history', [])
        current_file_ptr = (start_epoch * args.files_per_epoch) % num_total_files
        print(f"  Resumed from epoch {start_epoch}, File Pointer: {current_file_ptr}")

    print(f"Starting Sniper Mode Training on {device}...")

    for epoch in range(start_epoch, args.epochs):
        selected_files = []
        for i in range(args.files_per_epoch):
            selected_files.append(csv_files[(current_file_ptr + i) % num_total_files])
        
        current_file_ptr = (current_file_ptr + args.files_per_epoch) % num_total_files
        
        train_len = int(args.files_per_epoch * 0.9)
        train_files = selected_files[:train_len]
        val_files = selected_files[train_len:]
        
        print(f"\n[Epoch {epoch+1}] Loading {len(train_files)} train + {len(val_files)} val CSVs...")
        train_datasets = load_subset_datasets(train_files)
        val_datasets = load_subset_datasets(val_files)
        
        train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=args.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False)
        
        model.train()
        train_rewards = []
        train_losses = []
        hold_counts = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in pbar:
            x = batch['x'].to(device)
            current_price = batch['current_price'].to(device)
            future_prices = batch['future_prices'].to(device)

            optimizer.zero_grad()
            probs, tp_ratio, sl_ratio = model(x)
            
            with torch.no_grad():
                actions = torch.distributions.Categorical(probs).sample()
                hold_counts += (actions == 2).sum().item()
                total_samples += actions.size(0)

            # [修改點]: 移除 lmbda 參數傳遞，完全依靠不對稱 Reward
            loss, avg_reward = triple_barrier_loss(
                probs, tp_ratio, sl_ratio, 
                {'current_price': current_price, 'future_prices': future_prices}
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_rewards.append(avg_reward.item())
            
            # [修改點]: 介面不再顯示 lmbda
            pbar.set_postfix({'loss': f"{loss:.2e}", 'reward': f"{avg_reward.item():.2e}"})

        epoch_hold_rate = hold_counts / total_samples
        avg_train_reward = np.mean(train_rewards)
        reward_history.append(float(avg_train_reward))

        # [修改點]: 徹底移除 Dynamic Lambda Adjustment 區塊

        model.eval()
        val_rewards = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(device)
                current_price = batch['current_price'].to(device)
                future_prices = batch['future_prices'].to(device)
                probs, tp_ratio, sl_ratio = model(x)
                
                # Validation 也不傳遞 lmbda
                _, avg_reward = triple_barrier_loss(probs, tp_ratio, sl_ratio, 
                                                  {'current_price': current_price, 'future_prices': future_prices})
                val_rewards.append(avg_reward.item())

        avg_val_reward = np.mean(val_rewards)
        print(f"Summary | Loss: {np.mean(train_losses):.2e} | Reward: {avg_train_reward:.2e} | ValReward: {avg_val_reward:.2e} | HoldRate: {epoch_hold_rate:.2%}")

        # [修改點]: 儲存 Checkpoint 時不再包含 lmbda
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_reward': best_val_reward,
            'reward_history': reward_history
        }, 'checkpoints/latest_checkpoint.pth')
        
        if avg_val_reward > best_val_reward:
            best_val_reward = avg_val_reward
            torch.save(model.state_dict(), f'checkpoints/best_{args.model_name}')
            print(f"  *** New Best Validation Reward: {best_val_reward:.2e}. Model saved. ***")

    torch.save(model.state_dict(), f'checkpoints/{args.model_name}')
    print(f"Universal training complete. Model saved as {args.model_name}")

if __name__ == "__main__":
    train_all()