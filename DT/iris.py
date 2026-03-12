from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report
import pandas as pd

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
df['target']=iris.target
X=df.drop('target',axis=1)
Y=df['target']

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print(classification_report(ytest,ypred))