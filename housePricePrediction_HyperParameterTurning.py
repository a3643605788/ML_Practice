#房價預測-調整參數，改良預測結果
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #線性回歸模型
from sklearn.metrics import mean_squared_error, r2_score #評估模型結果好壞
import numpy as np;
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform



df = pd.read_csv("dataset/housePricePredition.csv")

import datetime
current_year = datetime.datetime.now().year
df["house_age"] = current_year - df["yr_built"] #建立「屋齡」
df["renovated"] = (df["yr_renovated"] > 0).astype(int) #是否翻修過
df["living_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1)  #居住面積比例(避免除以0)
df["basement_ratio"] = df["sqft_basement"] / (df["sqft_living"] + 1) #地下室佔比
df["is_multi_floor"] = (df["floors"] > 2).astype(int) #總樓層是否大於 2
df["sqft_lot_log"] = np.log1p(df["sqft_lot"]) #高度偏態變數做對數轉換

df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)
df["living_per_floor"] = df["sqft_living"] / (df["floors"] + 1)
df["lot_per_floor"] = df["sqft_lot"] / (df["floors"] + 1)
skew_cols = ['sqft_living', 'sqft_above', 'sqft_basement']
for col in skew_cols:
    df[col + "_log"] = np.log1p(df[col])
# 從 statezip 萃取郵遞區號
df["zipcode"] = df["statezip"].str.extract(r'(\d{5})').astype(float)
# 依郵遞區號加上平均房價特徵
df["zip_mean_price"] = df.groupby("zipcode")["price"].transform("mean")
# 城市平均價
df["city_mean_price"] = df.groupby("city")["price"].transform("mean")
# 做 one-hot，保留 dummy 欄位，同時保留原始 zipcode 以便之後需要（不 drop_first 才不會少一類）
zip_dummies = pd.get_dummies(df["zipcode"], prefix="zip", dtype=int)
df = pd.concat([df, zip_dummies], axis=1)

# 更新 feature：用 one-hot 欄位取代原本的 'zipcode'
zip_dummy_cols = zip_dummies.columns.tolist()

feature = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated',
    'house_age', 'renovated', 'living_ratio', 'basement_ratio',
    'is_multi_floor', 'sqft_lot_log',
    'bed_bath_ratio', 'living_per_floor', 'lot_per_floor',
    'sqft_living_log', 'sqft_above_log', 'sqft_basement_log',
    'zip_mean_price', 'city_mean_price'
] + zip_dummy_cols


# 修正異常值
# 刪除price偏離平均值超過3倍標準差的資料
from scipy import stats
# stats.zscore(df["price"])):計算每個數值高出或低了平均值有幾個標準差
# abs:絕對值
df = df[(abs(stats.zscore(df["price"]))) < 3]



# 特徵與標籤
X = df[feature] #輸入的資料(特徵)(自變數)
y = df["price"] #要預測的值(標籤)(應變數)

# 特徵處理
# scaler =StandardScaler()
# X_scaled = scaler.fit_transform(X) #把dataset的各特徵做特徵處理(用標準化的方式處理)
y_log = np.log1p(y) #y做log轉換


X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42) #用特徵處理過的dataset，以及log處理過的y_log，分割出訓練用和測試用
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42) #用特徵處理過的dataset，以及log處理過的y_log，分割出訓練用和測試用

# ---------- 線性回歸（需要縮放） ----------
# lin_pipeline = Pipeline([
#     ("scaler", StandardScaler()),
#     ("lin", LinearRegression())
# ])
# lin_pipeline.fit(X_train, y_train)
# y_pred_log = lin_pipeline.predict(X_test)
# y_pred = np.expm1(y_pred_log)

# ---------- 隨機森林（不縮放） + RandomizedSearchCV（快速） ----------
rf = RandomForestRegressor(random_state=42)

# param_dist = {
#     "n_estimators": randint(120, 320),        # 搜尋期先少一點樹
#     "max_depth": randint(8, 22),              # 限制深度
#     "min_samples_split": randint(2, 12),
#     "max_features": ["sqrt", "log2", None]
# }

param_dist = {
    "n_estimators": randint(200, 600),      #森林中樹的數量
    "max_depth": randint(8, 40),            #每顆樹的深度
    "min_samples_split": randint(2, 15),    #節點再分裂的最少樣本數
    "min_samples_leaf": randint(1, 6),      #葉子節點的最少樣本數
    "max_features": ["sqrt", "log2", None]  #每次分裂時考慮的特徵數量
}


rf_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=24,                  # 先來 24 組，通常很快就有感
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

# 用最佳參數「加大樹的數量」重訓一次（提升最終精度）
best_rf.set_params(n_estimators=1000, n_jobs=-1)  # 最終模型多放樹
best_rf.fit(X_train, y_train)

# 評估
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

y_test_expm1 = np.expm1(y_test)
y_rf_log = best_rf.predict(X_test)
y_rf = np.expm1(y_rf_log)

# RMSE = np.sqrt(mean_squared_error(y_test_expm1, y_pred))
# R2 = r2_score(y_test_expm1, y_pred)

RMSE_RF = np.sqrt(mean_squared_error(y_test_expm1, y_rf))
R2_RF = r2_score(y_test_expm1, y_rf)

# print("Linear RMSE:", RMSE)
# print("Linear R2:", R2)
# print("Linear 相對誤差:", RMSE / y.mean())
print("RF best params (search期):", rf_search.best_params_)
print("RF RMSE (final 1000 trees):", RMSE_RF)
print("RF R2 (final 1000 trees):", R2_RF)
print("RF 相對誤差:", RMSE_RF / y.mean())
print("price mean:", y.mean())





# =====================
# 匯出模型
# =====================
from joblib import dump
import os

# 確保 model 資料夾存在
os.makedirs("model", exist_ok=True)

model_path = "model/house_price_model.joblib"
dump(best_rf, model_path)

print(f"✓ 模型匯出完成：{model_path}")




# -----隨機森林調整參數簡介-----
# n_estimators	    森林中樹的數量      randint(200, 500)	
# 隨機從 200～499 間選一個整數。
# 樹越多 → 模型越穩定但越慢。
# 後續你已額外把最佳參數再擴成 1000 棵。

# max_depth         每棵樹的最大深度    randint(6, 30)	
# 控制每棵樹的成長深度。
# 太小 → 欠擬合，太深 → 過擬合。
# 這裡給 6~30 屬於合理範圍。

# min_samples_split 節點再分裂的最少樣本數  randint(2, 15)	
# 決定何時繼續分裂節點。
# 數值越大 → 分裂越保守。

# min_samples_leaf	葉節點的最少樣本數  randint(1, 6)	
# 防止樹生成太多只有1筆資料的葉節點（過擬合）。
# 範圍 1~5 算常見設定。

# max_features      每次分裂時考慮的特徵數量    ["sqrt", "log2", None]	
# 控制特徵隨機性：
# • "sqrt" → 根號(n_features)
# • "log2" → log₂(n_features)
# • None → 用全部特徵。
# 不同取樣策略會影響模型多樣性。