#房價預測-調整參數，改良預測結果
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression #線性回歸模型
from sklearn.metrics import mean_squared_error, r2_score #評估模型結果好壞
import numpy as np;
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor



df = pd.read_csv("dataset/housePricePredition.csv")

# 2025/09/18 先加上來而已，要理解特徵具體怎麼建立出來的

import datetime
current_year = datetime.datetime.now().year
df["house_age"] = current_year - df["yr_built"] #建立「屋齡」
df["renovated"] = (df["yr_renovated"] > 0).astype(int) #是否翻修過
df["living_ratio"] = df["sqft_living"] / (df["sqft_lot"] + 1)  #居住面積比例(避免除以0)
df["basement_ratio"] = df["sqft_basement"] / (df["sqft_living"] + 1) #地下室佔比
df["is_multi_floor"] = (df["floors"] > 2).astype(int) #總樓層是否大於 2
df["sqft_lot_log"] = np.log1p(df["sqft_lot"]) #高度偏態變數做對數轉換

feature = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
    'yr_built', 'yr_renovated',
    'house_age', 'renovated', 'living_ratio', 'basement_ratio',
    'is_multi_floor', 'sqft_lot_log'
]

# 2025/09/18 先加上來而已，要理解特徵具體怎麼建立出來的





# 要輸入的特徵(加入我們認為會影響房價的特徵)
# feature = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
#             'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
#             'yr_built', 'yr_renovated']

# 資料品質檢查
# print("shape:", df.shape)
# print("\nDTypes:\n", df.dtypes)
# print("\nHead:\n", df.head(3))
# print("\nDescribe (數值):\n", df[feature + ["price"]].describe(percentiles=[.01,.05,.5,.95,.99]))

# 修正異常值
# 刪除price偏離平均值超過3倍標準差的資料
from scipy import stats
# stats.zscore(df["price"])):計算每個數值高出或低了平均值有幾個標準差
# abs:絕對值
df = df[(abs(stats.zscore(df["price"]))) < 3]

# 缺失值補齊
# 刪除缺失值
# print(df.isnull().sum()) #每個欄位缺失的比數
# df = df.dropna(subset=["price"])


# 特徵與標籤
X = df[feature] #輸入的資料(特徵)(自變數)
y = df["price"] #要預測的值(標籤)(應變數)

# 特徵處理
scaler =StandardScaler()
X_scaled = scaler.fit_transform(X) #把dataset的各特徵做特徵處理(用標準化的方式處理)
y_log = np.log1p(y) #y做log轉換


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_log, test_size=0.2, random_state=42) #用特徵處理過的dataset，以及log處理過的y_log，分割出訓練用和測試用
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) #用特徵處理過的dataset，分割出訓練用和測試用

# 線性回歸模型訓練
model = LinearRegression()
model.fit(X_train, y_train) #用x訓練集資料和y訓練集資料，丟給線性回歸模型做訓練
y_pred_log = model.predict(X_test) #訓練完後拿x測試集資料產出預測結果
# y_pred = model.predict(X_test) #訓練完後拿x測試集資料產出預測結果
y_pred = np.expm1(y_pred_log) #還原到原始房價

# Baseline: 只猜平均值
# dummy = DummyRegressor(strategy="mean")
# dummy.fit(X_train, y_train)
# y_dummy = dummy.predict(X_test)

# 隨機叢林模型訓練
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_rf_log = rf.predict(X_test)
# y_rf = rf.predict(X_test)
y_rf = np.expm1(y_rf_log)

# 隨機叢林模型訓練-GridSearchCV(參數搜尋方法)
# 2025/09/24 先加上來而已，要理解怎麼建立出來的
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'max_features': [None, 'sqrt', 'log2'],  # 改掉 'auto'
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,  # 三折交叉驗證
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# 2025/09/24 先加上來而已，要理解怎麼建立出來的

# RMSE_Dummy = np.sqrt(mean_squared_error(y_test, y_dummy))
# RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
# RMSE_RF = np.sqrt(mean_squared_error(y_test, y_rf))

y_test_expm1 = np.expm1(y_test) #還原到原始房價

RMSE = np.sqrt(mean_squared_error(y_test_expm1, y_pred))
RMSE_RF = np.sqrt(mean_squared_error(y_test_expm1, y_rf))


# print("y's head: ", y.head())
# print("y's type: ", type(y.iloc[0]))

# print("Dummy's RMSE: ", RMSE_Dummy)
# print("Dummy's R^2 score: ", r2_score(y_test, y_dummy))

print("RMSE: ", RMSE)
print("R^2 score: ", r2_score(y_test_expm1, y_pred))
print("RF's RMSE", RMSE_RF)
print("RF's R^2 score: ", r2_score(y_test_expm1, y_rf))
print("最佳參數:", grid_search.best_params_)
print("最佳RMSE:", np.sqrt(-grid_search.best_score_))
print("price's mean: ", y.mean()) #平均房價
print("相對誤差: ", RMSE/y.mean())