# Import libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("iris.csv")

# Separate features and target
X = df.drop("variety", axis=1)   # independent variables
y = df["variety"]                # dependent variable (optional)

le=LabelEncoder()
y=le.fit_transform(y)

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Convert to dataframe
pca_df = pd.DataFrame(X_pca, columns=["PC1","PC2"])

print("PCA Dataset:")
print(pca_df)

# Variance explained
print("Explained Variance:", pca.explained_variance_ratio_)

# Visualization
plt.scatter(pca_df["PC1"], pca_df["PC2"], c=y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Result")
plt.show()