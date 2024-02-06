import numpy as np
import dtale
import pandas as pd
from spellchecker import SpellChecker
import peter_romany_module
import re
df = pd.read_csv(r"Mall_Customers.csv")
pd.set_option("display.max_rows", None)
pd.set_option("expand_frame_repr", False)
# print(df.head())# understand data(cols and rows)

'''                                                         Data wrangling
                                preparing raw data for analysis via convert raw data into analysis-ready data.
'''

# Check data types for any type mismatch
# df.info()
'''
After conducting a thorough check for type mismatch within the dataset, these columns contain type mismatch:
So CustomerID column is numerical but does not have measurement unit and no mean to order or perform operations. So must be qualitative data(categorical).(type 1 error)
Gender column is object but must be string.(type 2 error)
'''

# Handle type mismatch
mismatch_features = ['CustomerID', 'Gender']
df[mismatch_features] = df[mismatch_features].astype(str)

# Check missing data
# col00='Spending Score (1-100)'
# print(df[df[col00].isnull()])#row has null
# print(df[df[col00]=="?"])#row has ?
# print(df[df[col00]==""])#row has blank
# print(df[df[col00]==0])#row has 0

'''
After conducting a thorough check for missing data within the dataset, the dataset was complete, and no values were missing in the specified columns.
'''

# Check for all types of errors that may exist to identify and obtain columns that are error-free.
# peter_romany_module.regex_for_str(df,'Gender')
# peter_romany_module.regex_for_int(df,'Age')
# peter_romany_module.regex_for_int(df,'Annual Income (k$)')
# peter_romany_module.regex_for_int(df,'Spending Score (1-100)')

'''
After conducting a thorough check for all types of errors within the dataset, the dataset is free from various types of errors.
'''

# Check misspellings
# print(SpellChecker().unknown(df['Gender'])) # Check specific column

'''
After conducting a thorough check for misspellings within the dataset, there are no columns contain misspellings.
'''

# Check form, schema and other inconsistent
# match_indices=peter_romany_module.regex_patterns(df,'Gender','[^"Male""Female"]',return_match_indices=True)

'''
After conducting a thorough check for form, schema, and other inconsistencies within the dataset, all issues have been addressed
and the data appears to be consistent and accurate.
'''

'''                           Data mining and analysis   or   Exploratory Data Analysis (EDA)
                                        extracting knowledge(insights) from data begins
'''

# Handle outliers
age_outliers=peter_romany_module.dealing_with_outlier(df,'Age',show_outliers=False)
income_outliers=peter_romany_module.dealing_with_outlier(df,'Annual Income (k$)',show_outliers=False)
spending_outliers=peter_romany_module.dealing_with_outlier(df,'Spending Score (1-100)',show_outliers=False)

'''The method dealing_with_outlier() alternates between activation and deactivation to ensure the acquisition of accurate and comprehensive information from actual data.
It identifies outliers, storing them for further study and insight extraction purposes.'''

peter_romany_module.insights_by_descriptive_analytics(df,'Annual Income (k$)')
peter_romany_module.insights_by_descriptive_analytics(df,'Age')
peter_romany_module.insights_by_descriptive_analytics(df,'Spending Score (1-100)')

adults_spending_score= df[((df['Age']>=df['Age'].min()) & (df['Age']<=37))]['Spending Score (1-100)']
elderly_spending_score= df[((df['Age']>=38) & (df['Age']<=df['Age'].max()))]['Spending Score (1-100)']
adults_spending_score.name='adults_spending_score'
elderly_spending_score.name='elderly_spending_score'
# Check for Entropy and data diversity
# print(adults_spending_score.count())
# print(elderly_spending_score.count())

'''
After conducting a comprehensive analysis of entropy and data diversity within the dataset, it is evident that the data demonstrates balance, ensuring more equitable predictions and promoting model fairness.
'''

var1,var2=adults_spending_score,elderly_spending_score

peter_romany_module.check_normality(var1,var2)
peter_romany_module.check_variance_homogeneity(var1,var2)
peter_romany_module.independent_sample_ttest(var1,var2,equal_variance=True)
peter_romany_module.one_way_anova(df,df['Gender'],df['Spending Score (1-100)'],robust_anova=True)

deep_dive = pd.DataFrame({'Male': df[df['Gender'] == 'Male']['Spending Score (1-100)'].tolist()})
peter_romany_module.insights_by_descriptive_analytics(deep_dive,'Male')
deep_dive2 = pd.DataFrame({'Female': df[df['Gender'] == 'Female']['Spending Score (1-100)'].tolist()})
peter_romany_module.insights_by_descriptive_analytics(deep_dive2,'Female')

variable1=df['Annual Income (k$)']
variable2=df['Spending Score (1-100)']
peter_romany_module.check_normality(variable1,variable2)
peter_romany_module.check_variance_homogeneity(variable1,variable2)
peter_romany_module.pearsonr(variable1,variable2)
peter_romany_module.spearmanr(variable1,variable2)
peter_romany_module.sorted_zscore(variable1,show_z_score=False)
peter_romany_module.sorted_rank(variable1,show_percent_rank=False)

# d = dtale.show(df, host='localhost', subprocess=False)
# d.open_browser()

'''                                                    Machine Learning
            Dive into machine learning, leveraging algorithms to extract meaningful insights and make informed predictions from data.
'''

# perform clustering based on Annual Income and Spending Score
optimal_k_means_model, optimal_cluster_labels = peter_romany_module.k_means_clustering(df, ['Annual Income (k$)','Spending Score (1-100)'])
