# Liner Regression
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/housePricePredition.csv")

# 輸入的特徵(加入我們認為會影響房價的特徵)
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
            'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
            'yr_built', 'yr_renovated']

X = df[features] #輸入的特徵(自變數)
y = df['price'] #要預測的變數(應變數)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) #把訓練集的各特徵標準化處理(類似於讓每個特徵的單位一致)

from sklearn.linear_model import LinearRegression #線性回歸模型
from sklearn.metrics import mean_squared_error, r2_score #評估結果好壞模型

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) #餵入的資料分割成訓練和測試
model = LinearRegression()
model.fit(X_train, y_train) #x,y的資料丟進線性回歸模型做訓練，學習線性關係

y_pred = model.predict(X_test) #用x資料做預測

print("RMSE: ", mean_squared_error(y_test, y_pred))
print("R^2 score: ", r2_score(y_test, y_pred))

