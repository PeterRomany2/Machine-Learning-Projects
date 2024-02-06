import numpy as np
import dtale
import pandas as pd
from spellchecker import SpellChecker
import peter_romany_module
import re
import sklearn.datasets as ds

# loading the data from sklearn
breast_cancer_dataset = ds.load_breast_cancer()
# loading the data to a data frame
df = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)
df['label'] = breast_cancer_dataset.target
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
# col00='worst fractal dimension'
# print(df[df[col00].isnull()])#row has null
# print(df[df[col00]=="?"])#row has ?
# print(df[df[col00]==""])#row has blank
# print(df[df[col00]==0])#row has 0

'''
After conducting a thorough check for missing data within the dataset, the dataset was complete, and no values were missing in the specified columns.
'''

# Check for all types of errors that may exist to identify and obtain columns that are error-free.
# peter_romany_module.regex_for_float(df,'worst fractal dimension')
# peter_romany_module.regex_for_int(df,'label')

'''
After conducting a thorough check for all types of errors within the dataset, the dataset is free from various types of errors.
'''

'''                           Data mining and analysis   or   Exploratory Data Analysis (EDA)
                                        extracting knowledge(insights) from data begins
'''

# Handle outliers
peter_romany_module.dealing_with_outlier(df,'mean radius',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'mean texture',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'mean perimeter',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'mean area',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'mean smoothness',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'mean compactness',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'mean concavity',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'mean concave points',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'mean symmetry',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'mean fractal dimension',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'radius error',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'texture error',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'perimeter error',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'area error',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'smoothness error',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'compactness error',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'concavity error',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'concave points error',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'symmetry error',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'fractal dimension error',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'worst radius',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'worst texture',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'worst perimeter',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'worst area',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'worst smoothness',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'worst compactness',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'worst concavity',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'worst concave points',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'worst symmetry',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'worst fractal dimension',show_outliers=False)

'''The method dealing_with_outlier() alternates between activation and deactivation to ensure the acquisition of accurate and comprehensive information from actual data.
It identifies outliers, storing them for further study and insight extraction purposes.'''

peter_romany_module.insights_by_descriptive_analytics(df,'fractal dimension error')

worst_symmetry_for_Malignant_Breast_Cancer=df[df['label']==0]['worst symmetry']
worst_symmetry_for_Benign_Breast_Cancer=df[df['label']==0]['worst symmetry']
worst_symmetry_for_Malignant_Breast_Cancer.name='worst_symmetry_for_Malignant_Breast_Cancer'
worst_symmetry_for_Benign_Breast_Cancer.name='worst_symmetry_for_Benign_Breast_Cancer'

# Check for Entropy and data diversity
# print(worst_symmetry_for_Malignant_Breast_Cancer.count())
# print(worst_symmetry_for_Benign_Breast_Cancer.count())

'''
After conducting a comprehensive analysis of entropy and data diversity within the dataset, it is evident that the data demonstrates balance, ensuring more equitable predictions and promoting model fairness.
'''

var1,var2=worst_symmetry_for_Malignant_Breast_Cancer,worst_symmetry_for_Benign_Breast_Cancer
peter_romany_module.check_normality(var1,var2)
peter_romany_module.check_variance_homogeneity(var1,var2)
peter_romany_module.independent_sample_ttest(var1,var2,equal_variance=True)
# peter_romany_module.paired_sample_ttest(var1,var2)

variable1=df['worst radius']
variable2=df['worst perimeter']
peter_romany_module.check_normality(variable1,variable2)
peter_romany_module.check_variance_homogeneity(variable1,variable2)
peter_romany_module.pearsonr(variable1,variable2)
peter_romany_module.spearmanr(variable1,variable2)

# d = dtale.show(df, host='localhost', subprocess=False)
# d.open_browser()

'''                                                       Machine Learning
                 Dive into machine learning, leveraging algorithms to extract meaningful insights and make informed predictions from data.
'''

independent_variables=df.drop(columns='label', axis=1)
logistic_model=peter_romany_module.logistic_regression(df,independent_variables.columns,'label',max_iters=2500)
# Making a Predictive System
input_data = pd.DataFrame([[17.99,10.38,122.8,1001.0,0.1184,0.22862,0.28241,0.1471,0.2419,0.07871,0.84865,0.9053,5.9835,86.2,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,1937.05,0.1622,0.62695,0.7119,0.2654,0.41915,0.1189]], columns=independent_variables.columns)
prediction = logistic_model.predict(input_data)
print("The prediction is: ",prediction)
if prediction[0] == 0: print('The Breast cancer is Malignant')
else: print('The Breast Cancer is Benign')
