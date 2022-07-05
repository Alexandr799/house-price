import dill
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd


app = FastAPI()
with open('model.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    Id: int
    MSSubClass: int
    MSZoning: str
    LotFrontage: float
    LotArea: int
    Street: str
    Alley: str
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str
    MasVnrArea: float
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: str
    BsmtCond: str
    BsmtExposure: str
    BsmtFinType1: str
    BsmtFinSF1: float
    BsmtFinType2: str
    BsmtFinSF2: float
    BsmtUnfSF: float
    TotalBsmtSF: float
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str
    stFlrSF: int
    ndFlrSF: int
    LowQualFinSF: int
    GrLivArea: int
    BsmtFullBath: float
    BsmtHalfBath: float
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: str
    GarageType: str
    GarageYrBlt: float
    GarageFinish: str
    GarageCars: float
    GarageArea: float
    GarageQual: str
    GarageCond: str
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    SsnPorch: int
    ScreenPorch: int
    PoolArea: int
    PoolQC: str
    Fence: str
    MiscFeature: str
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str


class Prediction(BaseModel):
    id: int
    result: float


@app.get('/status')
def status():
    return 'I am OK!'


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([dict(form)])
    m = model['model']
    result = m.predict(df)
    return {
        'id': df['Id'],
        'result': round(result[0],2)
    }
