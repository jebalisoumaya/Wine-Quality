import pandas as pd

df=pd.read_csv('WineQT.csv')

print(df.head(5))
print(df.info)
print(df.describe())


print(df.isnull().sum())
#there is no missing values , so i don't need to do any imputations