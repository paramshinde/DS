from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd
from sklearn.metrics import accuracy_score,classification_report,recall_score,precision_score

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)

df['target']=iris.target
X=df.drop('target',axis=1)
Y=df['target']

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(xtrain,ytrain)

ypred=model.predict(xtest)

print("Accuracy ",accuracy_score(ytest,ypred))
print(classification_report(ytest,ypred))