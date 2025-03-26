import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
SELECTED_FEATURES = [
    'apache_4a_hospital_death_prob',
    'd1_mbp_min',
    'h1_resprate_min',
    'ventilated_apache',
    'd1_temp_max',
    'd1_sysbp_noninvasive_max',
    'd1_potassium_min','hospital_death'
]
# 讀取數據
df = pd.read_csv("datasets/train.csv")  # 請替換為你的數據路徑
df = df[SELECTED_FEATURES]
# 指定要繪製的變數
features = [
    'apache_4a_hospital_death_prob',
    'd1_mbp_min',
    'h1_resprate_min',
    'ventilated_apache',
    'd1_temp_max',
    'd1_sysbp_noninvasive_max',
    'd1_potassium_min'
]
# 篩選出 apache_4a_hospital_death_prob = -1 的樣本
subset = df[df['apache_4a_hospital_death_prob'] == -1]

# 計算 hospital_death = 0 的樣本數
count_hospital_death_0 = (subset['hospital_death'] == 0).sum()

# 計算總數
total_count = len(subset)

# 計算 hospital_death = 0 的機率
probability = count_hospital_death_0 / total_count if total_count > 0 else 0

print(f"當 apache_4a_hospital_death_prob = -1 時，hospital_death = 0 的機率為: {probability:.2%}")
# 設置畫布大小
plt.figure(figsize=(14, 10))

# 遍歷特徵，繪製直方圖 + KDE 曲線
for i, feature in enumerate(features, 1):
    plt.subplot(4, 2, i)  # 4 行 2 列的子圖布局
    sns.histplot(df[feature], bins=50, kde=True, color='blue')
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

# 調整子圖布局，避免重疊
plt.tight_layout()
plt.show()