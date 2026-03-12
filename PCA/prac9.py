'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardization
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Explained variance
print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# Visualization
plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")
plt.show()

'''

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()
X=iris.data
Y=iris.target

sc=StandardScaler()
Xscaled=sc.fit_transform(X)
pca=PCA()
XPCA=pca.fit_transform(Xscaled)
print("Explained Variance Ratio :",pca.explained_variance_ratio_)
plt.scatter(XPCA[:,0],XPCA[:,1],c=Y)
plt.show()
