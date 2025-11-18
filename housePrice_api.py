from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()

# 載入模型
model = joblib.load("model/house_price_model.joblib")

FEATURES = [
    'bedrooms',
    'bathrooms',
    'sqft_living',
    'sqft_lot',
    'floors',
    'waterfront',
    'view',
    'condition',
    'sqft_above',
    'sqft_basement',
    'yr_built',
    'yr_renovated'
]

class HouseInput(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int

@app.post("/predict")
def predict_price(house: HouseInput):
    df = pd.DataFrame([house.dict()])
    df = df[FEATURES]

    y_log = model.predict(df)[0]
    price = float(np.expm1(y_log))

    return {"predicted_price": price}
