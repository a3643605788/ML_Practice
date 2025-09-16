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

# 要輸入的特徵(加入我們認為會影響房價的特徵)
feature = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated']

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
print(df.isnull().sum()) #每個欄位缺失的比數
df = df.dropna(subset=["price"])


# 特徵與標籤
X = df[feature] #輸入的資料(特徵)(自變數)
y = df["price"] #要預測的值(標籤)(應變數)

# 特徵處理
scaler =StandardScaler()
X_scaled = scaler.fit_transform(X) #把dataset的各特徵做特徵處理(用標準化的方式處理)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) #用特徵處理過的dataset，分割出訓練用和測試用

# 線性回歸模型訓練
model = LinearRegression()
model.fit(X_train, y_train) #用x訓練集資料和y訓練集資料，丟給線性回歸模型做訓練
y_pred = model.predict(X_test) #訓練完後拿x測試集資料產出預測結果

# Baseline: 只猜平均值
# dummy = DummyRegressor(strategy="mean")
# dummy.fit(X_train, y_train)
# y_dummy = dummy.predict(X_test)

# 隨機叢林模型訓練
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)

# RMSE_Dummy = np.sqrt(mean_squared_error(y_test, y_dummy))
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
RMSE_RF = np.sqrt(mean_squared_error(y_test, y_rf))


# print("y's head: ", y.head())
# print("y's type: ", type(y.iloc[0]))

# print("Dummy's RMSE: ", RMSE_Dummy)
# print("Dummy's R^2 score: ", r2_score(y_test, y_dummy))

# print("RMSE: ", RMSE)
# print("R^2 score: ", r2_score(y_test, y_pred))
# print("RF's RMSE", RMSE_RF)
# print("RF's R^2 score: ", r2_score(y_test, y_rf))
# print("price's mean: ", y.mean()) #平均房價
# print("相對誤差: ", RMSE/y.mean())