from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd

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

plt.plot(range(1,11),sse)
plt.xlabel("No of cluster")
plt.ylabel("SSE")
plt.show()

kmeans=KMeans(n_clusters=4,random_state=42)
clusters=kmeans.fit_predict(df_scaled)
df['Clusters']=clusters
print(df)