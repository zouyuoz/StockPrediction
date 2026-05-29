import torch
import torch.nn.functional as F

class StrategyEngine:
    @staticmethod
    def compute_reward(action, tp, sl, current_price, future_prices, cost=0.001, hurdle=0.01):
        """
        Inputs:
        - action: [Batch] (0: Long, 1: Short, 2: Hold)
        - tp, sl: [Batch, 1] 
        - current_price: [Batch] 或 [Batch, 1]
        - future_prices: [Batch, Horizon]
        """
        device = current_price.device
        batch_size, horizon = future_prices.shape
        
        # 確保形狀對齊為 [Batch, 1]
        tp = tp.view(-1, 1)
        sl = sl.view(-1, 1)
        current_price = current_price.view(-1, 1)

        # 定義邊界價格
        long_tp_price = current_price * (1 + tp)
        long_sl_price = current_price * (1 - sl)
        short_tp_price = current_price * (1 - tp) # 做空的 TP 在下方
        short_sl_price = current_price * (1 + sl) # 做空的 SL 在上方

        # 建立觸碰矩陣 (Boolean)
        hit_long_tp = future_prices >= long_tp_price
        hit_long_sl = future_prices <= long_sl_price
        hit_short_tp = future_prices <= short_tp_price
        hit_short_sl = future_prices >= short_sl_price

        max_time = horizon + 1 # 定義一個超越 Horizon 的時間代表未觸發

        def get_first_hit_time(condition_matrix):
            # 取第一根滿足條件的 index，若完全沒碰到則給予 max_time
            hit_indices = torch.argmax(condition_matrix.int(), dim=1)
            any_hit = torch.any(condition_matrix, dim=1)
            return torch.where(any_hit, hit_indices, torch.tensor(max_time, device=device))

        time_long_tp = get_first_hit_time(hit_long_tp)
        time_long_sl = get_first_hit_time(hit_long_sl)
        time_short_tp = get_first_hit_time(hit_short_tp)
        time_short_sl = get_first_hit_time(hit_short_sl)

        # --- 計算 Long 的結果 ---
        long_exit_time = torch.min(time_long_tp, time_long_sl)
        long_timeout = long_exit_time == max_time
        # 若 timeout，以最後一根 K 線平倉
        long_safe_idx = torch.where(long_timeout, torch.tensor(horizon - 1, device=device), long_exit_time)
        long_exit_price = future_prices.gather(1, long_safe_idx.unsqueeze(1)).squeeze(1)
        
        # 保守原則：若同一根 K 線同時觸碰 TP 與 SL，視為先掃到 SL
        long_is_sl = (time_long_sl <= time_long_tp) & (~long_timeout)
        
        long_final_price = torch.where(long_is_sl, long_sl_price.squeeze(1),
                           torch.where(long_timeout, long_exit_price, long_tp_price.squeeze(1)))
        long_return = (long_final_price - current_price.squeeze(1)) / current_price.squeeze(1) - cost

        # --- 計算 Short 的結果 ---
        short_exit_time = torch.min(time_short_tp, time_short_sl)
        short_timeout = short_exit_time == max_time
        short_safe_idx = torch.where(short_timeout, torch.tensor(horizon - 1, device=device), short_exit_time)
        short_exit_price = future_prices.gather(1, short_safe_idx.unsqueeze(1)).squeeze(1)
        
        short_is_sl = (time_short_sl <= time_short_tp) & (~short_timeout)
        
        short_final_price = torch.where(short_is_sl, short_sl_price.squeeze(1),
                            torch.where(short_timeout, short_exit_price, short_tp_price.squeeze(1)))
        short_return = (current_price.squeeze(1) - short_final_price) / current_price.squeeze(1) - cost

        # --- Hurdle 過濾 (Alpha-hunting) ---
        # 如果利潤沒有跨越 hurdle，將其歸零 (或加入微小懲罰) 以避免模型頻繁交易於雜訊中
        long_return = torch.where((long_return > 0) & (long_return < hurdle), torch.tensor(0.0, device=device), long_return)
        short_return = torch.where((short_return > 0) & (short_return < hurdle), torch.tensor(0.0, device=device), short_return)

        # --- 整合最終 Reward ---
        reward = torch.zeros_like(long_return)
        reward = torch.where(action == 0, long_return, reward)
        reward = torch.where(action == 1, short_return, reward)
        # action == 2 (Hold) 維持為 0

        return reward


def triple_barrier_loss(probs, tp, sl, data_batch, lmbda=0.02, entropy_coef=0.01):
    """
    Inputs:
    - probs: [Batch, 3] 
    - tp, sl: [Batch, 1]
    - data_batch: dict 包含 'current_price' 與 'future_prices'
    - lmbda: Hold 懲罰係數
    """
    current_price = data_batch['current_price']
    future_prices = data_batch['future_prices']
    
    # 1. 動作採樣與 Policy Gradient
    dist = torch.distributions.Categorical(probs)
    action = dist.sample() # [Batch]
    
    # 環境結算不可微，必須使用 no_grad
    with torch.no_grad():
        reward = StrategyEngine.compute_reward(action, tp, sl, current_price, future_prices)
        # Advantage Normalization: 將 Reward 標準化，穩定梯度下降方向
        advantage = (reward - reward.mean()) / (reward.std() + 1e-8)

    # Policy Gradient Loss: - E[log(P(A)) * Advantage]
    log_probs = dist.log_prob(action)
    pg_loss = - (log_probs * advantage).mean()

    # 2. Entropy Bonus: 懲罰過度自信，防止模型迅速陷入 100% 只做 Long 或 Hold
    entropy = dist.entropy().mean()

    # 3. Hold Penalty: 直接對 Hold (index 2) 的輸出機率施壓
    # 數值越大，模型越不敢輸出 Hold
    hold_penalty = lmbda * probs[:, 2].mean()
    
    # ------------------------------------------------------------------
    # 4. TP / SL 輔助損失 (Crucial Fix for Non-Differentiability)
    # ------------------------------------------------------------------
    # 計算未來窗口內的最大上漲幅度與最大下跌幅度作為 "Label"
    with torch.no_grad():
        max_price = future_prices.max(dim=1)[0].unsqueeze(1)
        min_price = future_prices.min(dim=1)[0].unsqueeze(1)
        
        # 真實的波動極值 (百分比)
        actual_max_up = (max_price - current_price) / current_price
        actual_max_down = (current_price - min_price) / current_price
        
        # 將極值限制在網路輸出的合理範圍內 (例如 tp 最多 0.15, sl 最多 0.10)
        optimal_tp = torch.clamp(actual_max_up, min=0.0, max=0.15)
        optimal_sl = torch.clamp(actual_max_down, min=0.0, max=0.10)

    # 使用 MSE 迫使 tp 和 sl 的神經網路頭去預測真實的波動區間
    # 這樣 tp 和 sl 才能產生實質的梯度進行權重更新
    tp_loss = F.mse_loss(tp, optimal_tp)
    sl_loss = F.mse_loss(sl, optimal_sl)
    auxiliary_loss = tp_loss + sl_loss

    # 總 Loss 整合
    total_loss = pg_loss - (entropy_coef * entropy) + hold_penalty + auxiliary_loss

    # 回傳 loss 用於反向傳播，回傳 reward_mean 用於監控訓練狀態
    return total_loss, reward.mean()