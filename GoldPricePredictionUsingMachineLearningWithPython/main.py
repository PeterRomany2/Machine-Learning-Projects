import numpy as np
import dtale
import pandas as pd
from spellchecker import SpellChecker
import peter_romany_module
import re
df = pd.read_csv(r"gld_price_data.csv")
pd.set_option("display.max_rows", None)
pd.set_option("expand_frame_repr", False)
# print(df.head())# understand data(cols and rows)

'''                                                         Data wrangling
                                preparing raw data for analysis via convert raw data into analysis-ready data.
'''

# Check data types for any type mismatch
# df.info()

'''
After conducting a thorough check for type mismatch within the dataset, this column contains type mismatch:
Date column is object but must be datetime.
'''

# Handle type mismatch
df['Date'] = pd.to_datetime(df['Date'])

# Check missing data
# col00='EUR/USD'
# print(df[df[col00].isnull()])#row has null
# print(df[df[col00]=="?"])#row has ?
# print(df[df[col00]==""])#row has blank
# print(df[df[col00]==0])#row has 0

'''
After conducting a thorough check for missing data within the dataset, the dataset was complete, and no values were missing in the specified columns.
'''

# Check for all types of errors that may exist to identify and obtain columns that are error-free.
# peter_romany_module.regex_for_float(df,'EUR/USD')

'''
After conducting a thorough check for all types of errors within the dataset, the dataset is free from various types of errors.
'''

'''                           Data mining and analysis   or   Exploratory Data Analysis (EDA)
                                        extracting knowledge(insights) from data begins
'''

# Handle outliers
SPX_outliers=peter_romany_module.dealing_with_outlier(df,'SPX',show_outliers=False)
GLD_outliers=peter_romany_module.dealing_with_outlier(df,'GLD',show_outliers=False)
USO_outliers=peter_romany_module.dealing_with_outlier(df,'USO',show_outliers=False)
SLV_outliers=peter_romany_module.dealing_with_outlier(df,'SLV',show_outliers=False)
EUR_USD_outliers=peter_romany_module.dealing_with_outlier(df,'EUR/USD',show_outliers=False)

'''The method dealing_with_outlier() alternates between activation and deactivation to ensure the acquisition of accurate and comprehensive information from actual data.
It identifies outliers, storing them for further study and insight extraction purposes.'''

peter_romany_module.insights_by_descriptive_analytics(df,'GLD')

var1,var2=df['GLD'],df['SLV']
peter_romany_module.check_normality(var1,var2)
peter_romany_module.check_variance_homogeneity(var1,var2)
peter_romany_module.independent_sample_ttest(var1,var2,equal_variance=True)
# peter_romany_module.paired_sample_ttest(var1,var2)

variable1=df['GLD']
variable2=df['SLV']
peter_romany_module.check_normality(variable1,variable2)
peter_romany_module.check_variance_homogeneity(variable1,variable2)
peter_romany_module.pearsonr(variable1,variable2)
peter_romany_module.spearmanr(variable1,variable2)
peter_romany_module.sorted_zscore(variable1,show_z_score=False)
peter_romany_module.sorted_rank(variable1,show_percent_rank=False)

linear_model=peter_romany_module.linear_regression(df,['SPX', 'USO', 'SLV', 'EUR/USD'],'GLD',False)

# Making a Predictive System
input_data = pd.DataFrame([[2725.780029,14.405800,15.454200,1.182033]], columns=['SPX', 'USO', 'SLV', 'EUR/USD'])
prediction = linear_model.predict(input_data)
print("The prediction is: ",prediction)

# df = df.drop(columns='Dosage')
# d = dtale.show(df, host='localhost', subprocess=False)
# d.open_browser()