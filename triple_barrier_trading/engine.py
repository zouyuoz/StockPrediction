import torch
import torch.nn.functional as F

class StrategyEngine:
    @staticmethod
    def compute_reward(action, tp, sl, current_price, future_prices, cost=0.0015, hurdle=0.05):
        """
        Sniper Mode Reward System
        - action: [Batch] (0: Long, 1: Short, 2: Hold)
        - tp, sl: [Batch, 1] 
        - current_price: [Batch] 或 [Batch, 1]
        - future_prices: [Batch, Horizon]
        - cost: 0.0015 (考慮手動交易潛在較大的 slippage 與手續費)
        - hurdle: 0.05 (過濾掉小於 5% 的平庸波動，非大波段不交易)
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
        raw_long_return = (long_final_price - current_price.squeeze(1)) / current_price.squeeze(1) - cost

        # --- 計算 Short 的結果 ---
        short_exit_time = torch.min(time_short_tp, time_short_sl)
        short_timeout = short_exit_time == max_time
        short_safe_idx = torch.where(short_timeout, torch.tensor(horizon - 1, device=device), short_exit_time)
        short_exit_price = future_prices.gather(1, short_safe_idx.unsqueeze(1)).squeeze(1)
        
        short_is_sl = (time_short_sl <= time_short_tp) & (~short_timeout)
        
        short_final_price = torch.where(short_is_sl, short_sl_price.squeeze(1),
                            torch.where(short_timeout, short_exit_price, short_tp_price.squeeze(1)))
        raw_short_return = (current_price.squeeze(1) - short_final_price) / current_price.squeeze(1) - cost

        # ==========================================
        # 關鍵改造：狙擊手不對稱 Reward 塑形
        # ==========================================
        def shape_reward(returns):
            # 1. 虧損放大懲罰 (Risk Aversion，痛苦係數 x3)
            r = torch.where(returns < 0, returns * 3.0, returns)
            # 2. 獲利未達 Hurdle，視為無效交易，給予微小懲罰 (-0.02)
            r = torch.where((r >= 0) & (r < hurdle), torch.tensor(-0.02, device=device), r)
            # 3. 暴利非線性放大 (超過 Hurdle 的部分給予指數級獎勵)
            r = torch.where(r >= hurdle, (r * 10.0) ** 1.5, r)
            return r

        long_reward = shape_reward(raw_long_return)
        short_reward = shape_reward(raw_short_return)

        # --- 整合最終 Reward ---
        reward = torch.zeros_like(long_reward)
        reward = torch.where(action == 0, long_reward, reward)
        reward = torch.where(action == 1, short_reward, reward)
        # action == 2 (Hold) 維持為絕對安全的 0

        return reward


def triple_barrier_loss(probs, tp, sl, data_batch, entropy_coef=0.001):
    """
    Sniper Mode Policy Gradient Loss
    - 移除了 lmbda (Hold Penalty)，模型現在可以心安理得地 100% Hold。
    - entropy_coef 調降至 0.001，僅保留極微弱的探索。
    """
    current_price = data_batch['current_price']
    future_prices = data_batch['future_prices']
    
    # 1. 動作採樣與 Policy Gradient
    dist = torch.distributions.Categorical(probs)
    action = dist.sample() # [Batch]
    
    # 環境結算不可微，必須使用 no_grad
    with torch.no_grad():
        reward = StrategyEngine.compute_reward(action, tp, sl, current_price, future_prices)
        # Advantage Normalization: 加入微小 epsilon 防止全 Hold 時 reward 為 0 導致 NaN
        advantage = (reward - reward.mean()) / (reward.std() + 1e-5)

    # Policy Gradient Loss: - E[log(P(A)) * Advantage]
    log_probs = dist.log_prob(action)
    pg_loss = - (log_probs * advantage).mean()

    # 2. Entropy Bonus
    entropy = dist.entropy().mean()
    
    # ------------------------------------------------------------------
    # 3. TP / SL Auxiliary Loss (輔助損失)
    # ------------------------------------------------------------------
    # 確保即使模型選擇 Hold，負責輸出 tp/sl 的神經網路層仍會去學習真實波動
    with torch.no_grad():
        max_price = future_prices.max(dim=1)[0].unsqueeze(1)
        min_price = future_prices.min(dim=1)[0].unsqueeze(1)
        
        actual_max_up = (max_price - current_price) / current_price
        actual_max_down = (current_price - min_price) / current_price
        
        # 你的 TP_head 上限是 0.15，SL_head 上限是 0.10 (基於 models.py 的設計)
        optimal_tp = torch.clamp(actual_max_up, min=0.0, max=0.15)
        optimal_sl = torch.clamp(actual_max_down, min=0.0, max=0.10)

    tp_loss = F.mse_loss(tp, optimal_tp)
    sl_loss = F.mse_loss(sl, optimal_sl)
    auxiliary_loss = tp_loss + sl_loss

    # 總 Loss 整合 (不再包含 hold_penalty)
    total_loss = pg_loss - (entropy_coef * entropy) + auxiliary_loss

    return total_loss, reward.mean()