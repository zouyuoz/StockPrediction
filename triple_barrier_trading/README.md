你是一位頂級的量化金融機器學習專家。我希望實作一個端到端（End-to-End）的深度學習交易策略模型，將傳統的「價格預測問題」轉化為「離散動作與動態邊界決策問題」。請為我設計並實作完整的 PyTorch 代碼。

以下是該系統的詳細規格設計：

### 1. 數據輸入 (Input Specification)
* **原始數據：** 歷史 1 年的日 K 線數據，包含 (Open, High, Low, Close, Volume)。
* **特徵工程 (Feature Engineering)：** 為了消除數據的非平穩性（Non-stationarity）並保持尺度一致，請將原始數據轉換為以下特徵：
    * 價格特徵：計算對數收益率 Log Returns ($r_t = \ln(Close_t) - \ln(Close_{t-1})$)。
    * 日內波動特徵：$\ln(High_t / Low_t)$。
    * 開盤相對位置：$\ln(Open_t / Close_{t-1})$。
    * 成交量特徵：相對成交量 RVOL ($Volume_t / \text{SMA}(Volume, 20)_t$)。
* **輸入張量形狀 (Tensor Shape)：** $[Batch\_Size, Window\_Length, Feature\_Dim]$，其中 $Window\_Length = 252$（約一年交易日），$Feature\_Dim = 4$。

### 2. 模型架構 (Model Architecture)
* 採用雙分支或多模態網路（如 1D-CNN + Transformer Encoder），專注於捕捉價量的局部微觀 Pattern 與長程依賴關係。
* **輸出頭 (Output Heads)：** 模型共有三個獨立的輸出預測頭：
    1.  **策略動作頭 (Action Head):** 分類任務（Classification），經由 Softmax 輸出三個離散動作的機率分布：$[P(\text{Long}), P(\text{Short}), P(\text{Hold})]$。
    2.  **止盈邊界頭 (Take-Profit Head):** 迴歸任務（Regression），經由 Sigmoid 縮放，輸出相對於當前收盤價的止盈百分比 $\Delta_{tp} \in (0, 0.15)$。
    3.  **止損邊界頭 (Stop-Loss Head):** 迴歸任務（Regression），經由 Sigmoid 縮放，輸出相對於當前收盤價的止損百分比 $\Delta_{sl} \in (0, 0.10)$。

### 3. 策略執行與標籤模擬 (Strategy Execution Engine)
模型給出決策後（假設在 $t$ 時刻），環境會模擬未來的價格走勢，直到觸發以下三重柵欄法（Triple Barrier）條件之一：
* **做多 (Long)：** 若未來價格先觸及 $Close_t \times (1 + \Delta_{tp})$，則獲利出場；若先觸及 $Close_t \times (1 - \Delta_{sl})$，則止損出場；若超過最大持有期限（例如 20 天）皆未觸及，則以第 20 天的收盤價強制平倉。
* **做空 (Short)：** 若未來價格先觸及 $Close_t \times (1 - \Delta_{tp})$，則獲利出場；若先觸及 $Close_t \times (1 + \Delta_{sl})$，則止損出場；若超過最大持有期限則強制平倉。
* **空手 (Hold)：** 不建立任何倉位。

### 4. 自定義損失函數設計 (Custom Differentiable Loss Function)
為了讓模型能夠進行梯度反向傳播，請設計一個結合「期望收益」與「延宕懲罰」的 Differentiable Loss 或使用 Policy Gradient (PPO/REINFORCE) 的 Reward 函數：

$$\text{Loss} = - \mathbb{E}[\text{Return}] + \lambda \cdot \text{Penalty}_{\text{Hold}}$$

* **收益部分 ($\text{Return}$):** 當 Action 為 Long 或 Short 時，根據實際平倉時的真實盈虧百分比計算（扣除 0.1% 的交易摩擦成本）。
* **延宕懲罰部分 ($\text{Penalty}_{\text{Hold}}$):** 當 Action 為 Hold 時，該步收益為 0，但引入一個時間延宕懲罰（Time Delay Penalty），公式為 $\text{Penalty} = \alpha \cdot \ln(\text{Consecutive\_Hold\_Days} + 1)$，以防止模型為了絕對安全而陷入永遠不交易的死循環。

---

請基於以上規格，為我生成：
1.  **PyTorch 模型類別 (`TradingPolicyNetwork`)**，包含 Action、Take-Profit、Stop-Loss 三個 Output Heads。
2.  **自定義的 Loss Function 實作**（若使用 Reinforcement Learning，請提供環境環境 `gym.Env` 與 PPO Reward 計算邏輯）。
3.  **數據預處理自訂 Dataset 類別**。

<!-- 
python triple_barrier_trading/bulk_fetch.py \
    --api_key 340e56a5899a4379aa5ddac92c44ff98 \
    --config triple_barrier_trading/symbols.json \
    --interval 1h 
-->

回到 train_all.py。我現在已經蒐集了非常多的資料，這樣會導致於一個 epoch 要跑非常久。
有沒有辦法，將 data loader 的方式，改成 "每個 epoch 只抓 80 筆 csv (64 train + 16 valid)" 來訓練，
並且 circular 輪流取資料
假設我們有 300 筆 .csv file。取資料流程為:
1 epoch: 000~079
2 epoch: 080~159
3 epoch: 160~239
4 epoch: 240~299 + 000~019
5 epoch: 020~099
...
並且回答，這樣訓練會有甚麼隱憂嗎?