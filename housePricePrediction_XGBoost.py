#房價預測-調整參數，改良預測結果。XGBoost是梯度提升樹(GBDT)強化版
import sys
import inspect
import pandas as pd
import numpy as np
import datetime
from scipy import stats
from scipy.stats import randint, uniform

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
import xgboost as xgb

# --- 環境檢查（除錯用；不影響訓練） ---
print("PYTHON   :", sys.executable)
print("XGBOOST  :", xgb.__version__)
try:
    import sklearn
    print("SKLEARN  :", sklearn.__version__)
    from xgboost.sklearn import XGBRegressor as _XGBR
    print("XGB.fit signature:", inspect.signature(_XGBR.fit))
except Exception as _:
    pass

# -----------------------------
# 1) 讀檔與基礎前處理
# -----------------------------
df = pd.read_csv("dataset/housePricePredition.csv")

current_year = datetime.datetime.now().year
df["house_age"] = current_year - df["yr_built"]                  # 屋齡
df["renovated"] = (df["yr_renovated"] > 0).astype(int)           # 是否翻修
df["living_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1)    # 居住面積比例
df["basement_ratio"] = df["sqft_basement"] / (df["sqft_living"] + 1)  # 地下室佔比
df["is_multi_floor"] = (df["floors"] > 2).astype(int)            # 樓層 > 2
df["sqft_lot_log"] = np.log1p(df["sqft_lot"])                    # 偏態矯正

df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)
df["living_per_floor"] = df["sqft_living"] / (df["floors"] + 1)
df["lot_per_floor"] = df["sqft_lot"] / (df["floors"] + 1)

# 對偏態特徵做 log1p
for col in ['sqft_living', 'sqft_above', 'sqft_basement']:
    df[col + "_log"] = np.log1p(df[col])

# 從 statezip 萃取 zipcode
df["zipcode"] = df["statezip"].str.extract(r'(\d{5})').astype(float)

# One-hot zipcode（不 drop_first）
zip_dummies = pd.get_dummies(df["zipcode"], prefix="zip", dtype=int)
df = pd.concat([df, zip_dummies], axis=1)
zip_dummy_cols = zip_dummies.columns.tolist()

# 特徵欄位（先不含 zip_mean_price / city_mean_price，之後補進 X_train / X_test）
feature = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated',
    'house_age', 'renovated', 'living_ratio', 'basement_ratio',
    'is_multi_floor', 'sqft_lot_log',
    'bed_bath_ratio', 'living_per_floor', 'lot_per_floor',
    'sqft_living_log', 'sqft_above_log', 'sqft_basement_log'
] + zip_dummy_cols

# 移除 price 的極端值（>3σ）
df = df[(abs(stats.zscore(df["price"]))) < 3]

# 特徵與標籤
X_all = df[feature]
y_all = df["price"]
y_log_all = np.log1p(y_all)

# -----------------------------
# 2) Train/Test 切分
# -----------------------------
X_train, X_test, y_train_log, y_test_log = train_test_split(
    X_all, y_log_all, test_size=0.2, random_state=42
)

# 為了建立「不洩漏」的 zip/city 平均價，需要用到對應 index 的原欄位
train_zip = df.loc[X_train.index, "zipcode"]
test_zip  = df.loc[X_test.index,  "zipcode"]

# 若資料真的有 city 欄位就使用；如果沒有，這裡保底用 zipcode 當 city 群組（避免 KeyError）
if "city" in df.columns:
    train_city = df.loc[X_train.index, "city"]
    test_city  = df.loc[X_test.index,  "city"]
else:
    train_city = train_zip.copy()
    test_city  = test_zip.copy()

# 用 train 的「原尺度」價格來算平均價
y_train_raw = np.expm1(y_train_log)

zip_mean_map  = y_train_raw.groupby(train_zip).mean()
city_mean_map = y_train_raw.groupby(train_city).mean()
global_mean   = y_train_raw.mean()

# 將均值特徵加入到 X_train / X_test（用 train 的 map，test 未出現者以全域均值補）
X_train = X_train.copy()
X_test  = X_test.copy()
X_train["zip_mean_price"]  = train_zip.map(zip_mean_map).fillna(global_mean)
X_test["zip_mean_price"]   = test_zip.map(zip_mean_map).fillna(global_mean)
X_train["city_mean_price"] = train_city.map(city_mean_map).fillna(global_mean)
X_test["city_mean_price"]  = test_city.map(city_mean_map).fillna(global_mean)

# -----------------------------
# 3) XGBoost + RandomizedSearchCV
#   注意：不要用 xgb 當變數名，避免覆蓋模組
# -----------------------------
xgb_reg = XGBRegressor(
    random_state=42,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1
)

param_dist = {
    "n_estimators": randint(300, 3000),         # 交給 early stopping 決定最佳迭代
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.15),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 12),
    "gamma": uniform(0.0, 0.5),
    "reg_alpha": uniform(0.0, 0.1),
    "reg_lambda": uniform(0.7, 0.5)
}

search = RandomizedSearchCV(
    xgb_reg,
    param_distributions=param_dist,
    n_iter=32,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    random_state=42,
    verbose=1
)

search.fit(X_train, y_train_log)
best_xgb = search.best_estimator_

# -----------------------------
# 4) 早停訓練 + 評估（xgboost.train 版，適用 3.x）
# -----------------------------
# 1) 取出 RandomizedSearchCV 找到的最佳參數，轉為原生 train 參數
best_params = best_xgb.get_xgb_params()
best_params.update({
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
})

# 2) 準備 DMatrix（此處 y 用的是對數空間）
dtrain = xgb.DMatrix(X_train, label=y_train_log)
dvalid = xgb.DMatrix(X_test,  label=y_test_log)

# 3) 設定較大的上限，交給 EarlyStopping 停
num_boost_round = 5000

bst = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=[(dvalid, "valid")],
    callbacks=[EarlyStopping(rounds=100, save_best=True)]
)

best_iter = bst.best_iteration

# 4) 預測與評估
y_test_true = np.expm1(y_test_log)
y_pred_log = bst.predict(dvalid)     # 直接用 Booster 預測（已自動使用最佳迭代）
y_pred = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_test_true, y_pred))
r2 = r2_score(y_test_true, y_pred)

# print("XGB best params (search期):", search.best_params_)
# print("best_iteration:", best_iter)
print("XGB RMSE (early stopping):", rmse)
print("XGB R2 (early stopping):", r2)
print("XGB 相對誤差:", rmse / y_all.mean())
print("price mean:", y_all.mean())

# （可選）若後續想沿用 sklearn 包裝器的 predict 介面：
# 讓包裝器掛上這顆 Booster 並同步最佳樹數
best_xgb._Booster = bst
best_xgb.n_estimators = best_iter + 1



# XGBoost(梯度下降) 為例：
# 模型一開始亂猜，誤差很大（Loss 高）。
# 計算「梯度」→ 告訴我們往哪個方向能讓 Loss 變小。
# 建立一棵新樹，去修正這個誤差（負梯度）。
# 把這棵樹加進整體模型 → 模型變得更準。
# 重複步驟 2–4，每次都讓誤差更小。
