`screener.py` 是預測未來，現在，請你寫另一個程式 (`long_pred_test.py`)，對於過去股價進行預測。
也算是一個測試，看他的預估有多準。
一樣，迴圈所有股票，window 設為一年，判斷:
- 未來10天的漲幅大於等於 10%，
- 且 Confidence Score 大於等於 50%，
- 且這一年的 volume Z-Score 小於等於20 (這是篩選條件)
- 篩掉成交量過低的股票區間，不考慮
你可以自己定義 window 的 stride 要多大，但未來要有10天的股價。stride 建議在 5~20 之間，以 5 為倍數
---
每當一個 prediction 被 trigger，就紀錄一次 (紀錄在新的 `long_pred_test_result.csv`):
- 預測漲幅 與 實際漲幅
- 預測成功定義：在這 10 天內，漲幅有一度超過 $\max (10\%, 預測漲幅的 0.8 倍)$。
- 要記錄 prediction success/fail
- 可以記錄其他指標，來判斷一個判斷有多準/多不準
---
迴圈停止條件: 有 100 筆 success，則停止