import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler

iris = load_iris()

X = iris.data
columns = iris.feature_names

df = pd.DataFrame(X, columns=columns)

print("Original Data:")
print(df.head())

scaler = StandardScaler()

X_standardized = scaler.fit_transform(X)

df_standardized = pd.DataFrame(X_standardized, columns=columns)

print("\nStandardized Data:")
print(df_standardized.head())

normalizer = MinMaxScaler()

X_normalized = normalizer.fit_transform(X)

df_normalized = pd.DataFrame(X_normalized, columns=columns)

print("\nNormalized Data:")
print(df_normalized.head())