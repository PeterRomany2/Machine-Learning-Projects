import numpy as np
import dtale
import pandas as pd
from spellchecker import SpellChecker
import peter_romany_module
import re
df = pd.read_csv(r"creditcard.csv")
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
# col00='Class'
# print(df[df[col00].isnull()])#row has null
# print(df[df[col00]=="?"])#row has ?
# print(df[df[col00]==""])#row has blank
# print(df[df[col00]==0])#row has 0

'''
After conducting a thorough check for missing data within the dataset, the dataset was complete, and no values were missing in the specified columns.
'''

# Check for all types of errors that may exist to identify and obtain columns that are error-free.
# peter_romany_module.regex_for_float(df,'Amount')
# peter_romany_module.regex_for_int(df,'Class')

'''
After conducting a thorough check for all types of errors within the dataset, the dataset is free from various types of errors.
'''

'''                           Data mining and analysis   or   Exploratory Data Analysis (EDA)
                                        extracting knowledge(insights) from data begins
'''

# Handle outliers
peter_romany_module.dealing_with_outlier(df,'Time',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V1',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V2',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V3',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V4',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V5',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V6',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V7',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V8',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V9',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V10',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V11',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V12',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V13',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V14',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V15',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V16',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V17',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V18',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V19',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V20',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V21',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V22',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V23',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V24',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V25',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V26',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V27',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'V28',show_outliers=False)
peter_romany_module.dealing_with_outlier(df,'Amount',show_outliers=False)
# peter_romany_module.dealing_with_outlier(df,'Class',show_outliers=False)

'''The method dealing_with_outlier() alternates between activation and deactivation to ensure the acquisition of accurate and comprehensive information from actual data.
It identifies outliers, storing them for further study and insight extraction purposes.'''

peter_romany_module.insights_by_descriptive_analytics(df,'Amount')

var1,var2=df['Amount'],df['Class']
peter_romany_module.check_normality(var1,var2)
peter_romany_module.check_variance_homogeneity(var1,var2)
peter_romany_module.independent_sample_ttest(var1,var2,equal_variance=True)

variable1=df['Time']
variable2=df['Amount']
peter_romany_module.check_normality(variable1,variable2)
peter_romany_module.check_variance_homogeneity(variable1,variable2)
peter_romany_module.pearsonr(variable1,variable2)
peter_romany_module.spearmanr(variable1,variable2)

# Check for Entropy and data diversity
print(df['Class'].value_counts())
'''
After conducting a thorough check for entropy and data diversity within the dataset, the Dataset imbalance, potentially leading to biased predictions and impacting model fairness.
'''

# separating the data for analysis
legit = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]

'''
Balancing Imbalanced Datasets: Under-Sampling Techniques.
'''

legit_sample = legit.sample(n=len(fraud))
new_dataset = pd.concat([legit_sample, fraud], axis=0)
independent_variables=new_dataset.drop(columns='Class', axis=1)
logistic_model=peter_romany_module.logistic_regression(new_dataset,independent_variables.columns,'Class')

# Making a Predictive System
input_data = pd.DataFrame([[0,-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,-0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,-0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,-0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,-0.189114843888824,0.133558376740387,-0.0210530534538215,149.62]], columns=independent_variables.columns)
prediction = logistic_model.predict(input_data)
print("The prediction is: ",prediction)
if (prediction[0] == 0):
  print('The credit card is legitimate')
else:
  print('The credit card is fraudulent')

# df = df.drop(columns='Dosage')
# d = dtale.show(df, host='localhost', subprocess=False)
# d.open_browser()