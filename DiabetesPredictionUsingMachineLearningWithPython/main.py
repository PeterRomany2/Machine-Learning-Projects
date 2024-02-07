import numpy as np
import dtale
import pandas as pd
import peter_romany_module
import re
df = pd.read_csv(r"diabetes.csv")
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
# col00='Outcome'
# print(df[df[col00].isnull()])#row has null
# print(df[df[col00]=="?"])#row has ?
# print(df[df[col00]==""])#row has blank
# print(df[df[col00]==0])#row has 0

'''
After conducting a thorough check for missing data within the dataset, the dataset was complete, and no values were missing in the specified columns.
'''
# Check for all types of errors that may exist to identify and obtain columns that are error-free.
# peter_romany_module.regex_for_float(df,'BMI')
# peter_romany_module.regex_for_float(df,'DiabetesPedigreeFunction')
# peter_romany_module.regex_for_int(df,'Age')
# peter_romany_module.regex_for_int(df,'Outcome')
# peter_romany_module.regex_for_int(df,'Insulin')
# peter_romany_module.regex_for_int(df,'SkinThickness')
# peter_romany_module.regex_for_int(df,'BloodPressure')
# peter_romany_module.regex_for_int(df,'Glucose')
# peter_romany_module.regex_for_int(df,'Pregnancies')

'''
After conducting a thorough check for all types of errors within the dataset, the dataset is free from various types of errors.
'''

'''                           Data mining and analysis   or   Exploratory Data Analysis (EDA)
                                        extracting knowledge(insights) from data begins
'''
# Handle outliers
Pregnancies_outliers=peter_romany_module.dealing_with_outlier(df,'Pregnancies',show_outliers=False)
Glucose_outliers=peter_romany_module.dealing_with_outlier(df,'Glucose',show_outliers=False)
BloodPressure_outliers=peter_romany_module.dealing_with_outlier(df,'BloodPressure',show_outliers=False)
SkinThickness_outliers=peter_romany_module.dealing_with_outlier(df,'SkinThickness',show_outliers=False)
Insulin_outliers=peter_romany_module.dealing_with_outlier(df,'Insulin',show_outliers=False)
BMI_outliers=peter_romany_module.dealing_with_outlier(df,'BMI',show_outliers=False)
DiabetesPedigreeFunction_outliers=peter_romany_module.dealing_with_outlier(df,'DiabetesPedigreeFunction',show_outliers=False)
Age_outliers=peter_romany_module.dealing_with_outlier(df,'Age',show_outliers=False)
Outcome_outliers=peter_romany_module.dealing_with_outlier(df,'Outcome',show_outliers=False)

'''The method dealing_with_outlier() alternates between activation and deactivation to ensure the acquisition of accurate and comprehensive information from actual data.
It identifies outliers, storing them for further study and insight extraction purposes.'''

peter_romany_module.insights_by_descriptive_analytics(df,'Age')

var1,var2=df['Glucose'],df['BloodPressure']
peter_romany_module.check_normality(var1,var2)
peter_romany_module.check_variance_homogeneity(var1,var2)
peter_romany_module.independent_sample_ttest(var1,var2,equal_variance=True)
# peter_romany_module.paired_sample_ttest(var1,var2)
peter_romany_module.one_way_anova(df,df['Outcome'],df['BloodPressure'],robust_anova=True)

variable1=df['Age']
variable2=df['BloodPressure']
peter_romany_module.check_normality(variable1,variable2)
peter_romany_module.check_variance_homogeneity(variable1,variable2)
peter_romany_module.pearsonr(variable1,variable2)
peter_romany_module.spearmanr(variable1,variable2)

# d = dtale.show(df, host='localhost', subprocess=False)
# d.open_browser()

'''                                                       Machine Learning
                 Dive into machine learning, leveraging algorithms to extract meaningful insights and make informed predictions from data.
'''

svc_model=peter_romany_module.support_vector_classification(df,['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'],'Outcome',Kernel="linear",Regularization=1)

# Making a Predictive System
input_data = pd.DataFrame([[5,166,72,19,175,25.8,0.587,51]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age'])
prediction = svc_model.predict(input_data)
print("The prediction is: ",prediction)
if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
