import pandas as pd
import numpy as np

# 讀取 XGBoost 與 TabNet 預測結果
xgb_df = pd.read_csv("/Users/chenbaiyan/Desktop/something/resultstomerge/submission_xg_0.78.csv")  # 格式: [id, prediction]
tabnet_df = pd.read_csv("/Users/chenbaiyan/Desktop/something/resultstomerge/submission_0.789_tabnet.csv")  # 格式: [id, prediction]

# 確保 encounter_id 順序相同
assert (xgb_df["encounter_id"] == tabnet_df["encounter_id"]).all(), "❌ encounter_id 順序不匹配"
# 設定 XGBoost 和 TabNet 的權重
w_xgb = 0.6  # XGBoost 權重（假設它效果更好）
w_tabnet = 0.4  # TabNet 權重

# 計算加權平均機率
ensemble_proba = (w_xgb * xgb_df["hospital_death"] + w_tabnet * tabnet_df["hospital_death"]) / (w_xgb + w_tabnet)

# 轉換為 0/1
ensemble_binary = (ensemble_proba > 0.4).astype(int)

# 儲存結果
ensemble_df = pd.DataFrame({
    "encounter_id": xgb_df["encounter_id"], 
    "hospital_death": ensemble_binary
})

ensemble_df.to_csv("submission_ensemble_weighted.csv", index=False)

print("✅ 加權融合完成，結果已存入 submission_ensemble_weighted.csv")