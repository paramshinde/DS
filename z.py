import pandas as pd
import numpy as np

data = {
    "Country": ["France", "Spain", "Germany", "Spain", "Germany", "France", "Spain", np.nan, "France", "Germany"],
    "Age": [44, 27, 30, 38, 40, np.nan, 35, 48, 50, 37],
    "Salary": [72000, 48000, 54000, 61000, np.nan, 58000, 52000, 79000, 83000, np.nan],
    "Purchased": ["No", "Yes", "No", "No", "Yes", "Yes", "No", "Yes", np.nan, "No"]
}

df = pd.DataFrame(data)

print(df)
df.to_csv("sales_dataset.csv", index=False)