#!/usr/bin/env python
# same as prac3_chi-square-test.ipynb

# Chi-Square Test

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sb
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

df = sb.load_dataset('mpg')
print("\nMPG Dataframe :-\n")
print(df)

print("\nHorsepower Description :-\n")
print(df['horsepower'].describe())

print("\nModel year description :-\n")
print(df['model_year'].describe())


bins = [0, 75, 150, 240]
df['horsepower_new'] = pd.cut(
    df['horsepower'], bins=bins, labels=['l', 'm', 'h'])
print("\nNew Horsepower data :-\n")
print(df['horsepower_new'])


ybins = [69, 72, 74, 84]
label = ['t1', 't2', 't3']
df['modelyear_new'] = pd.cut(df['model_year'], bins=ybins, labels=label)
print("\nNew Model Year data:-\n")
print(df['modelyear_new'])


df_chi = pd.crosstab(df['horsepower_new'], df['modelyear_new'])
print("\ndf_chi :-\n")
print(df_chi)


print("\nchi2_contingency of df_chi :-\n")
print(stats.chi2_contingency(df_chi))


print("\nConclusion: There is sufficient evidence to reject the null hypothesis,",
      "indicating that there is a significant association between 'horsepower_new'",
      "and 'modelyear_new' categories.")
