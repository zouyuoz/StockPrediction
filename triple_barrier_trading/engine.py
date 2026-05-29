import torch

class StrategyEngine:
    @staticmethod
    def compute_reward(action, tp, sl, current_price, future_prices, cost=0.001, hurdle=0.01):
        """
        Refined Reward: Balanced for exploration and Alpha-hunting.
        # HERE #
        """
        batch_size = action.shape[0]
        returns = torch.zeros(batch_size, device=action.device)
        
        for i in range(batch_size):
            act = action[i].item()
            if act == 2: # Hold
                returns[i] = 0.0
                continue
            
            p0 = current_price[i].item()
            p_future = future_prices[i]
            
            tp_dist = tp[i].item()
            sl_dist = sl[i].item()
            
            tp_price = p0 * (1 + tp_dist) if act == 0 else p0 * (1 - tp_dist)
            sl_price = p0 * (1 - sl_dist) if act == 0 else p0 * (1 + sl_dist)
            
            realized_ret = 0.0
            exited = False
            for t in range(len(p_future)):
                pt = p_future[t].item()
                if act == 0: # Long
                    if pt >= tp_price:
                        realized_ret = (pt / p0) - 1 - cost
                        exited = True; break
                    elif pt <= sl_price:
                        realized_ret = (pt / p0) - 1 - cost
                        exited = True; break
                else: # Short
                    if pt <= tp_price:
                        realized_ret = (p0 / pt) - 1 - cost
                        exited = True; break
                    elif pt >= sl_price:
                        realized_ret = (p0 / pt) - 1 - cost
                        exited = True; break
            
            if not exited:
                p_last = p_future[-1].item()
                realized_ret = (p_last / p0 - 1 - cost) if act == 0 else (p0 / p_last - 1 - cost)

            # --- BALANCED NONLINEAR TRANSFORMATION ---
            if realized_ret > hurdle:
                # Big Win: Moderate exponential boost
                returns[i] = realized_ret * 50.0 
            elif realized_ret > 0:
                # Small Win: Linear positive
                returns[i] = realized_ret * 10.0 
            else:
                # Loss: Negative penalty (symmetrical to Big Win)
                returns[i] = realized_ret * 50.0 
                
        return returns

def triple_barrier_loss(probs, tp, sl, data_batch, lmbda=0.02):
    """
    Advanced Policy Gradient with Advantage Normalization and Entropy Bonus.
    This structure prevents 100% Hold collapse.
    """
    # 1. Action Sampling & Entropy
    m = torch.distributions.Categorical(probs)
    actions = m.sample()
    log_probs = m.log_prob(actions)
    entropy = m.entropy().mean() # Encourage exploration
    
    # 2. Environment Simulation
    with torch.no_grad():
        rewards = StrategyEngine.compute_reward(
            actions, tp, sl, 
            data_batch['current_price'], 
            data_batch['future_prices']
        )
        
        # Treat lmbda as cost of inaction
        adjusted_rewards = rewards.clone()
        adjusted_rewards[actions == 2] = -lmbda
        
        # --- CRITICAL: ADVANTAGE NORMALIZATION ---
        # Normalize within batch. Even if all rewards are -0.02, 
        # the 'best' actions become positive and 'worst' become negative.
        if len(adjusted_rewards) > 1:
            adv = (adjusted_rewards - adjusted_rewards.mean()) / (adjusted_rewards.std() + 1e-8)
        else:
            adv = adjusted_rewards

    # 3. Total Loss
    # We maximize (log_prob * advantage) + entropy
    policy_loss = -(log_probs * adv).mean()
    
    # entropy_beta=0.01: Forces the model to NOT pick a single action 100% of the time.
    total_loss = policy_loss - (0.01 * entropy)
    
    return total_loss, rewards.mean()

# -----------------------

class StrategyEngine:
    @staticmethod
    def compute_reward(action, tp, sl, current_price, future_prices, cost=0.001, hurdle=0.01):
        """
        Refined Reward: Balanced for exploration and Alpha-hunting.
        Inputs:
        - action: [Batch] 模型選擇的動作 (0: Long, 1: Short, 2: Hold)
        - tp: [Batch, 1] 模型預測的動態止盈百分比 (0 ~ 0.15)
        - sl: [Batch, 1] 模型預測的動態止損百分比 (0 ~ 0.10)
        - current_price: [Batch] 進場價
        - future_prices: [Batch, batch_size] 未來 24h 價格序列
        - cost: 單次交易成本 (0.001 = 0.1%)
        - hurdle: 小利潤門檻，超過此值才
        """
        # TODO

def triple_barrier_loss(probs, tp, sl, data_batch, lmbda=0.02):
    """
    Advanced Policy Gradient with Advantage Normalization and Entropy Bonus.
    This structure prevents 100% Hold collapse.

    Inputs:
    - probs: [Batch, 3] 模型輸出的動作機率 (Long, Short, Hold)
    - tp: [Batch, 1] 模型預測的動態止盈百分比 (0 ~ 0.15)
    - sl: [Batch, 1] 模型預測的動態止損百分比 (0 ~ 0.10)
    - data_batch: 包含 'current_price' (進場價) 與 'future_prices' (未來 24h 價格序列) 的字典
    - lmbda: 空手 (Hold) 的懲罰代價，數值越大模型越傾向於交易
    """
    # TODO