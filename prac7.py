import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
iris=load_iris()
iris_df=pd.DataFrame(data=np.c_[iris['data'],iris['target']],columns=iris['feature_names']+['target'])
binary_df=iris_df[iris_df['target']!=2]
x=binary_df.drop('target',axis=1)
y=binary_df['target']
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=42)
log_mod=LogisticRegression()
log_mod.fit(xtrain,ytrain)
ypred=log_mod.predict(xtest)
print("Logistic Regression Metrics")
print("Accuracy Score: ",accuracy_score(ytest,ypred))
print("Precision Score : ",precision_score(ytest,ypred))
print("Classifiction Report: ",classification_report(ytest,ypred))
print("Recall Score: \n",recall_score(ytest,ypred))

decision_model=DecisionTreeClassifier()
decision_model.fit(xtrain,ytrain)
ypred=decision_model.predict(xtest)
print("Decision Tree Metrics")
print("Accuracy Score: ",accuracy_score(ytest,ypred))
print("Precision Score : ",precision_score(ytest,ypred))
print("Classifiction Report: ",classification_report(ytest,ypred))
print("Recall Score: ",recall_score(ytest,ypred))

