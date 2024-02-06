import numpy as np
import dtale
import pandas as pd
from spellchecker import SpellChecker
import peter_romany_module
import re
df = pd.read_csv(r"insurance.csv")
pd.set_option("display.max_rows", None)
pd.set_option("expand_frame_repr", False)
# print(df.head())# understand data(cols and rows)

'''                                                         Data wrangling
                                preparing raw data for analysis via convert raw data into analysis-ready data.
'''

# Check data types for any type mismatch
# df.info()

'''
After conducting a thorough check for type mismatch within the dataset, all data types in the dataset were consistent and matched their expected formats.
'''

# Check missing data
# col00='charges'
# print(df[df[col00].isnull()])#row has null
# print(df[df[col00]=="?"])#row has ?
# print(df[df[col00]==""])#row has blank
# print(df[df[col00]==0])#row has 0

'''
After conducting a thorough check for missing data within the dataset, the dataset was complete, and no values were missing in the specified columns.
'''

# Check for all types of errors that may exist to identify and obtain columns that are error-free.
# peter_romany_module.regex_for_str(df,'sex')
# peter_romany_module.regex_for_str(df,'region')
# peter_romany_module.regex_for_str(df,'smoker')
# peter_romany_module.regex_for_float(df,'charges')
# peter_romany_module.regex_for_float(df,'bmi')
# peter_romany_module.regex_for_int(df,'age')

'''
After conducting a thorough check for all types of errors within the dataset, the dataset is free from various types of errors.
'''

# Check misspellings
# print(SpellChecker().unknown(df)) # Get all columns that have misspelled
# print(SpellChecker().unknown(df['region'])) # Check specific column

'''
After conducting a thorough check for misspellings within the dataset, there are no columns contain misspellings.
'''

# Check form, schema and other inconsistent
# match_indices=peter_romany_module.regex_patterns(df,'sex','[^"male""female"]',return_match_indices=True)

'''
After conducting a thorough check for form, schema, and other inconsistencies within the dataset, all issues have been addressed
and the data appears to be consistent and accurate.
'''

'''                           Data mining and analysis   or   Exploratory Data Analysis (EDA)
                                        extracting knowledge(insights) from data begins
'''

# Handle outliers
age_outliers=peter_romany_module.dealing_with_outlier(df,'age',show_outliers=False)
charges_outliers=peter_romany_module.dealing_with_outlier(df,'charges',show_outliers=False)
children_outliers=peter_romany_module.dealing_with_outlier(df,'children',show_outliers=False)
bmi_outliers=peter_romany_module.dealing_with_outlier(df,'bmi',show_outliers=False)

'''The method dealing_with_outlier() alternates between activation and deactivation to ensure the acquisition of accurate and comprehensive information from actual data.
It identifies outliers, storing them for further study and insight extraction purposes.'''

peter_romany_module.insights_by_descriptive_analytics(df,'age')
peter_romany_module.insights_by_descriptive_analytics(df,'children')
peter_romany_module.insights_by_descriptive_analytics(df,'bmi')
peter_romany_module.insights_by_descriptive_analytics(df,'charges')

ages_of_males= df[df['sex']=='male']['age']
ages_of_females= df[df['sex']=='female']['age']
ages_of_males.name='ages_of_males'
ages_of_females.name='ages_of_females'
# Check for Entropy and data diversity
# print(ages_of_males.count())
# print(ages_of_females.count())

'''
After conducting a comprehensive analysis of entropy and data diversity within the dataset, it is evident that the data demonstrates balance, ensuring more equitable predictions and promoting model fairness.
'''

var1,var2=ages_of_males,ages_of_females
peter_romany_module.check_normality(var1,var2)
peter_romany_module.check_variance_homogeneity(var1,var2)
peter_romany_module.independent_sample_ttest(var1,var2,equal_variance=True)
peter_romany_module.one_way_anova(df,df['region'],df['charges'],robust_anova=True)

variable1=df['age']
variable2=df['charges']
peter_romany_module.check_normality(variable1,variable2)
peter_romany_module.check_variance_homogeneity(variable1,variable2)
peter_romany_module.pearsonr(variable1,variable2)
peter_romany_module.spearmanr(variable1,variable2)
peter_romany_module.sorted_zscore(variable1,show_z_score=False)
peter_romany_module.sorted_rank(variable1,show_percent_rank=False)

# d = dtale.show(df, host='localhost', subprocess=False)
# d.open_browser()

'''                                                       Machine Learning
                 Dive into machine learning, leveraging algorithms to extract meaningful insights and make informed predictions from data.
'''

# Encoding non-numeric values for numerical consistency in order to be able to make linear_regression
peter_romany_module.label_encoder(df,'smoker',show_label_codes=False)
peter_romany_module.label_encoder(df,'region',show_label_codes=False)
peter_romany_module.label_encoder(df,'sex',show_label_codes=False)

ridge_model=peter_romany_module.ridge_regression(df,['age','bmi','children','Encoded smoker','Encoded region','Encoded sex'],'charges',lamda=1,show_best_fit_line=False)

# Making a Predictive System
input_data = pd.DataFrame([[31,25.74,0,1,0,1]], columns=['age','bmi','children','Encoded smoker','Encoded region','Encoded sex'])
prediction = ridge_model.predict(input_data)
print('The insurance cost is USD ', prediction[0])
