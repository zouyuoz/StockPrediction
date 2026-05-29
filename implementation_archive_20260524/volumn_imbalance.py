import numpy as np

def is_imbalanced(data, threshold=3.0):
    """
    判定數據集是否不平衡
    :param data: list 或 np.array 的數值資料
    :param threshold: 敏感度閥值 (建議設為 2.0 - 3.0)
    :return: True (不平衡), False (平衡)
    """
    D = np.array(data)
    if len(D) < 2:
        return False
    
    # 1. 計算 Median Absolute Deviation (MAD)
    # 使用中位數而非平均值，避免 10M 這種極端值影響基準線
    median = np.median(D)
    mad = np.median(np.abs(D - median))
    
    # 避免 mad 為 0 的邊緣情況
    if mad == 0:
        return False
    
    # 2. 計算 Modified Z-score
    # 衡量最新數據或整體分佈對中位數的偏移程度
    # 0.6745 是常態分佈下 MAD 與標準差的轉換常數
    modified_z_scores = 0.6745 * (D - median) / mad
    
    # 3. 判定邏輯
    # 如果數據集中有任何一點的修正 Z 分數超過閥值，視為不平衡
    # 對於交易量爆衝的情況，這能有效捕捉結構性突變
    max_score = np.max(np.abs(modified_z_scores))
    
    return max_score > threshold

# --- 測試案例 ---
normal_data = [20000, 22000, 21000, 25000, 23000] # 平衡
burst_data = [20000, 22000, 21000, 10000000]      # 不平衡

print(f"Normal data: {is_imbalanced(normal_data)}") # False
print(f"Burst data: {is_imbalanced(burst_data)}")   # True