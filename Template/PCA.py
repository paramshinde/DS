import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


'''
df = pd.read_csv('file')

print("\nColumns in dataset:")
print(df.columns)


X = df.drop('target', axis=1)
y = df['target']
X = df
y = None

'''
data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)
X = df
y = data.target

# -------------------------
# HANDLE CATEGORICAL DATA
# -------------------------
X = pd.get_dummies(X)

# -------------------------
# FEATURE SCALING
# -------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# APPLY PCA
# -------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# -------------------------
# VISUALIZATION
# -------------------------
plt.scatter(X_pca[:,0], X_pca[:,1], c=y if y is not None else None)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization")
plt.show()