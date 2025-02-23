# Project
# Final project 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.metrics import roc_curve, auc

data1=pd.read_csv("My Drive/Diabetes_Data/diabetes_012_health_indicators_BRFSS2015.csv")
data2=pd.read_csv("My Drive/Diabetes_Data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")

df=pd.concat([data1, data2], ignore_index=True)

data1.columns

data2.columns

df

df.isnull().sum()

df['Diabetes_012'].unique()

df1=df.drop(['Diabetes_binary'],axis='columns')

df1.columns

df1.isnull().sum()

df1.isna().sum()

df1.shape

df2 = df1.dropna(subset=['Diabetes_012','PhysHlth','DiffWalk','Sex','Age','Education','Income'])

df2['Diabetes_012'].unique()

df2.shape

df2.isnull().sum()

df3=df2.drop(['AnyHealthcare','NoDocbcCost','Sex','Age','Education','Income'],axis='columns')

df3

# EDA
df2 = df1.dropna(subset=['Diabetes_012','PhysHlth','DiffWalk','Sex','Age','Education','Income'])df2 = df1.dropna(subset=['Diabetes_012','PhysHlth','DiffWalk','Sex','Age','Education','Income'])
