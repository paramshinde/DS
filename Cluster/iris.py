#load iris dataset apply k means
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
#print(df.head())

scaler=MinMaxScaler()
df_scaled=scaler.fit_transform(df)
#print(df_scaled)

sse=[]
for k in range(1,11):
    km=KMeans(n_clusters=k,random_state=42)
    km.fit(df_scaled)
    sse.append(km.inertia_)

plt.plot(range(1,11), sse)
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.title("Elbow Method")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

# Add cluster column
df['Cluster'] = clusters

print(df.head())

plt.scatter(df['sepal length (cm)'], df['petal width (cm)'], c=df['Cluster'])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering on Iris Dataset")
plt.show()

