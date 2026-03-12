import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

'''

file = input("Enter CSV file name: ")
df = pd.read_csv(file)

print("\nColumns in dataset:")
print(df.columns)

target = input("\nEnter target column name: ")

'''

from sklearn.datasets import load_iris

data = load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
target = 'target'

# -------------------------
# DEFINE FEATURES & TARGET
# -------------------------
X = df.drop(target, axis=1)
y = df[target]

# Convert categorical data if present
X = pd.get_dummies(X)

# -------------------------
# TRAIN TEST SPLIT
# -------------------------
xtrain, xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# TRAIN MODEL
# -------------------------
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

# -------------------------
# PREDICTION
# -------------------------
ypred = model.predict(xtest)

# -------------------------
# EVALUATION
# -------------------------
print("\nAccuracy:", accuracy_score(ytest, ypred))
print("\nClassification Report:\n")
print(classification_report(ytest, ypred))