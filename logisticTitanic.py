# Logistic Regression
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train = pd.read_csv("dataset/train-Titanic.csv")

# 修正欄位名稱空格
train.columns = train.columns.str.strip()

#把缺的資料補上
train.fillna({'Age': train['Age'].median()}, inplace=True)              #補中位數
train.fillna({'Fare': train['Fare'].median()}, inplace=True)            #補中位數
train.fillna({'Embarked': train['Embarked'].mode()[0]}, inplace=True)   #補眾數

#類別轉換，因為機器學習模型只能處理數值，無法處理文字
le_sex = LabelEncoder()
le_embarked = LabelEncoder()
train['Sex'] = le_sex.fit_transform(train['Sex'])                   #將文字類型轉成數字('male', 'female' -> 0,1)
train['Embarked'] = le_embarked.fit_transform(train['Embarked'])    #將文字類型轉成數字('S', 'C', 'Q' -> 0,1,2)

#擷取稱謂
train['Title'] = train['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False) #從 Name 欄位中擷取「稱謂」文字（如 Mr., Mrs., Miss. 等）。稱謂常隱含乘客的社會地位或性別，對於預測是否生存可能有用
train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare') #冷門的稱謂合併成 Rare，降低 overfitting 風險
train['Title'] = train['Title'].replace(['Mlle', 'Ms'], 'Miss')     #等價稱謂整併（Mlle、Ms → Miss），降低 overfitting 風險
train['Title'] = train['Title'].replace(['Mme'], 'Mrs')             #等價稱謂整併（Mme → Mrs），降低 overfitting 風險
train['Title'] = LabelEncoder().fit_transform(train['Title'])       #將文字類型轉成數字

# 選取特徵
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title'] #選擇和生還可能有關的特徵(欄位)
X = train[features]     #和生存可能有關的dataset
y = train['Survived']   #是否生還的dataset

# 切分資料 80%訓練/20%測試
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_val) #20%的 X_val 測試資料做測試

# 評估準確率
acc = accuracy_score(y_val, y_pred) #比對是否準確(y_val:生還答案 y_pred:生還預測結果)
print(acc)
