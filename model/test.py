import dill
import pandas as pd

with open('model.pkl', 'rb') as file:
    model = dill.load(file)

df = pd.read_csv('/home/alex/PycharmProjects/house-price/data/train.csv')

m = model['model'].predict(df.drop(columns=('SalePrice')))
print(m)