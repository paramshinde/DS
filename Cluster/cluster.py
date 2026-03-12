# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Step 2: Load dataset
# Change file name or dataset source
#df = pd.read_csv("iris.csv")

# Optional: select only numeric columns
#df = df[['sepal.length','sepal.width','petal.length','petal.width']]

from sklearn.datasets import load_diabetes
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)

print("Dataset Preview:")
print(df.head())

# Step 3: Data Scaling
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Convert back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)

# Step 4: Elbow Method to find optimal K
sse = []

for k in range(1,11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_scaled)
    sse.append(km.inertia_)

# Plot elbow graph
plt.plot(range(1,11), sse, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("SSE")
plt.title("Elbow Method")
plt.show()

# Step 5: Apply K-Means
k = int(input("Enter number of clusters: "))

kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

# Step 6: Add cluster column
df['Cluster'] = clusters

# Step 7: Print results
print("\nClustered Dataset:")
print(df)

# Step 8: Visualization (first two features)
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=df['Cluster'])
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title("K-Means Clustering")
plt.show()