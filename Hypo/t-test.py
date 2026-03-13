#!/usr/bin/env python
# same as prac3_t-test.ipynb

# T-Test

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


## Generate two samples for demonstration purpose
sample1 = np.random.normal(loc=10, scale=2, size=30)
sample2 = np.random.normal(loc=12, scale=2, size=30)


## Perform a two-sample t-test
t_statistic, p_value = stats.ttest_ind(sample1, sample2)


## Set the significance level
alpha = 0.05

print("Result of two-sample t-test")
print("T-statistic:", t_statistic)
print("P-value:", p_value)
print("Degree of freedom:", len(sample1) + len(sample2) - 2)


## Plot the distributions
plt.figure(figsize=(10, 6))
plt.hist(sample1, alpha=0.5, label='Sample1', color='Blue')
plt.hist(sample2, alpha=0.5, label='Sample2', color='Orange')
plt.axvline(np.mean(sample1), color='blue', linestyle='dashed', linewidth=2)
plt.axvline(np.mean(sample2), color='orange', linestyle='dashed', linewidth=2)
plt.title('Distributions of sample1 and sample2')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.legend()

## Highlight the critical region if null hypothesis is rejected
if p_value < alpha:
    critical_region = np.linspace(min(sample1.min(), sample2.min()),
                                  max(sample1.max(), sample2.max()))
    plt.fill_between(critical_region, 0, 5, color='red',
                     alpha=0.3, label='Critical Region')
    plt.text(11, 5, "T-statistic: %.2f" % t_statistic, ha='center',
             va='center', color='black', backgroundcolor='white')


## Draw conclusions
if p_value < alpha:
    print("Conclusion: There is significant evidence to reject the null hypothesis.")
    if np.mean(sample1) > np.mean(sample2):
        print("Interpretation: The mean of sample 1 is significantly higher than that of sample 2.")
    else:
        print("Interpretation: The mean of sample 2 is significantly higher than that of sample 1.")
else:
    print("Conclusion: There is no significant evidence to reject the null hypothesis.")
    print("Interpretation: There is no significant difference in means between the two samples.")

plt.show()