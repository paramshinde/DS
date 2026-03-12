#construct a DT using python to classify whether a person will wait at a restaurant or not

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,classification_report
from sklearn.tree import DecisionTreeClassifier

df=pd.read_csv('restaurant.csv')
print(df.head())

le=LabelEncoder()
for col in df.columns:
    df[col]=le.fit_transform(df[col])

# Features and target
X = df.drop('Wait', axis=1)
y = df['Wait']

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

# Prediction
ypred = model.predict(X)

print("\nAccuracy:", accuracy_score(y, ypred))
print(classification_report(y,ypred))
