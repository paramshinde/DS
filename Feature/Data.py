import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler

df=pd.read_csv('sales.csv')
print("Original Dataset")
print(df,"\n")

#fillna
df2=df.fillna(value=0)
#print(df2,"\n")

#dropna
df3=df.dropna()
#print(df3)

#dummification
df_dummy=pd.get_dummies(df,columns=['Country','Purchased'])
#print(df_dummy)

#standardization
standard=StandardScaler()
df_standard=standard.fit_transform(df_dummy)
#print(pd.DataFrame(df_standard,columns=df_dummy.columns).head())

#normalization
norm = MinMaxScaler()
df_normal = norm.fit_transform(df_dummy)

print("\nNormalized Data")
print(pd.DataFrame(df_normal, columns=df_dummy.columns).head())