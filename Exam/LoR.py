import numpy as np
from sklearn.linear_model import LinearRegression

# dataset
height = np.array([151,174,138,128]).reshape(-1,1)
weight = np.array([63,81,56,91])

# model
model = LinearRegression()

# training
model.fit(height, weight)

# prediction for existing data
pred = model.predict(height)

print("Slope:", model.coef_)
print("Intercept:", model.intercept_)
print("Predicted weights:", pred)

# example prediction
print("Predicted weight for height 160:", model.predict([[160]]))