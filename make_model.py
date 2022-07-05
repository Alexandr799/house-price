from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from dill import dump
import datetime


def delete_col(df):
  df1 = df.copy()
  df1 = df1.drop(columns=[
                            'Alley',
                            'FireplaceQu',
                            'PoolQC',
                            'Fence',
                            'MiscFeature',
                            'Id'
                             ])
  return df1


def make_model():
    import pandas as pd
    data = pd.read_csv('data/train.csv')
    df = data.copy()
    x = df.drop(columns=['SalePrice'])
    y = df['SalePrice']
    gbr = GradientBoostingRegressor(random_state=42)
    pipe_num = Pipeline(steps=[
        ('delete_na', SimpleImputer(strategy='median')),
        ('std', StandardScaler())
    ])

    pipe_cat = Pipeline(steps=[
        ('delete', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))
    ])

    column_trans = ColumnTransformer([
        ('transform_cat', pipe_cat,
         make_column_selector(dtype_include=object)),
        ('transform_num', pipe_num, make_column_selector(dtype_include=['int64', 'float64']))
    ])

    head_model = Pipeline(steps=[
        ('delete_columns', FunctionTransformer(delete_col)),
        ('preprocessing', column_trans),
        ('model', gbr)
    ])

    head_model.fit(
        x,
        y
    )
    model_to_pkl = {'model': head_model,
                    'metadata': {
                        'name': 'house_price_predict',
                        'date': datetime.datetime.now(),
                        'type': 'GradientBoostingRegressor' }
                    }
    with open('model/model.pkl', 'wb') as file:
        dump(model_to_pkl, file)
    print('Готово, модель в папке model')


if __name__ == '__main__':
    make_model()
