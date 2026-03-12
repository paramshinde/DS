import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder,StandardScaler
import matplotlib.pyplot as plt

df=pd.read_csv('winequality-red.csv')

X=df.drop('quality',axis=1)
Y=df['quality']

scaler=StandardScaler()
XScaled=scaler.fit_transform(X)

pca=PCA()
XPca=pca.fit_transform(XScaled)
print("Expeirenced Variance Ratio ",pca.explained_variance_ratio_)
plt.scatter(XPca[:,0],XPca[:,1],c=Y)
plt.show()