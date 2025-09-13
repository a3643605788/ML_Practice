#房價預測-調整參數，改良預測結果
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/housePricePredition.csv")

# 要輸入的特徵(加入我們認為會影響房價的特徵)
feature = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated']

# 特徵與標籤
X = df[feature] #輸入的資料(特徵)(自變數)
y = df["price"] #要預測的值(標籤)(應變數)

# 特徵處理
scaler =StandardScaler()
X_scaled = scaler.fit_transform(X) #把dataset的各特徵做特徵處理(用標準化的方式處理)

from sklearn.linear_model import LinearRegression #線性回歸模型
from sklearn.metrics import mean_squared_error, r2_score #評估模型結果好壞

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) #用特徵處理過的dataset，分割出訓練用和測試用
model = LinearRegression()
model.fit(X_train, y_train) #用x訓練集資料和y訓練集資料，丟給線性回歸模型做訓練

y_pred = model.predict(X_test) #訓練完後拿x測試集資料產出預測結果

RMSE = mean_squared_error(y_test, y_pred)

print("RMSE: ", RMSE)
print("R^2 score: ", r2_score(y_test, y_pred))

print("price's mean: ", y.mean()) #平均房價
print("相對誤差: ", RMSE/y.mean())