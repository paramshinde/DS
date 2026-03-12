#implement a DT on titanic dataset
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data=pd.read_csv('titanic.csv')
#print(data.head())

le=LabelEncoder()
for col in data.columns:
    data[col]=le.fit_transform(data[col])

X=data.drop('survived',axis=1)
Y=data['survived']

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
print(classification_report(ytest,ypred))