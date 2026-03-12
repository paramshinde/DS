#apply feature scaling technique like standardization and normalization using python on boston housing dataset
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder

cali=fetch_california_housing()

df=pd.DataFrame(cali.data,columns=cali.feature_names)

print("Original Dataset")
print(df.head())

#standardization
scaler=StandardScaler()
df_standard=scaler.fit_transform(df)
print(pd.DataFrame(df_standard,columns=cali.feature_names).head())

#Normalization
normal=MinMaxScaler()
df_normal=normal.fit_transform(df)
print(pd.DataFrame(df_normal,columns=cali.feature_names).head())

