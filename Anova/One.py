import pandas as pd
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

group1=[23,25,29,34]
group2=[19,20,22,24]
group3=[19,20,22,24]
group4=[19,20,22,24]

data=pd.DataFrame({
    'value':group1+group2+group3+group4,
    'group':['group1']*len(group1)+['group2']*len(group2)+['group3']*len(group3)+['group4']*len(group4)
})

fstats,pval=stats.f_oneway(group1,group2,group3,group4)

print("One Way ANOVA")
print("F-stat:",fstats)
print("P-value:",pval)

tukey_res=pairwise_tukeyhsd(data['value'],data['group'])
print(tukey_res)