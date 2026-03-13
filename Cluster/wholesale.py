#!/usr/bin/env python
# same as prac7_k-means_clustering.ipynb

# # K-Means Clustering

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


df = pd.read_csv("wholesale.csv")
print("\n>>> df.head()", df.head(), sep='\n', end='\n\n')


categorical_features = ["Channel", "Region"]
continous_features = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
print("\n>>> df[continous_features].describe()", df[continous_features].describe(), sep='\n', end='\n\n')


for col in categorical_features:
    dummies = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummies], axis=1)
    df.drop(col, axis=1, inplace=True)
print("\n>>> df.head()", df.head(), sep='\n', end='\n\n')


scaler = MinMaxScaler()
scaler.fit(df)
data = scaler.transform(df)

inertias = []  # sum of squared distances
K = range(1, 15)
for k in K:
    model = KMeans(n_clusters=k)
    model = model.fit(data)
    inertias.append(model.inertia_)

print("\nValues of Inertias :-\n")
for k in K:
    print("At k = %2d, inertia = %f" %(k, inertias[k-1]))


# Visualize the data
plt.plot(K, inertias, 'bx-')
plt.xlabel("Values of K")
plt.ylabel("Inertias (Sum of Squared Distances)")
plt.title("Elbow method for optimal K")
plt.show()
