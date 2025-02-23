# Project
# Predict the onset of diabetes based on diagnostic measures and present a multi-modal system to track and trace it

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
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df3)

def grab_col_names(dataframe, cat_th=10, car_th=20):
   cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

   num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

   cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

   cat_cols = cat_cols + num_but_cat

   cat_cols = [col for col in cat_cols if col not in cat_but_car]

   num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

   num_cols = [col for col in num_cols if col not in num_but_cat]

   print(f"Observations: {dataframe.shape[0]}")
   print(f"Variables: {dataframe.shape[1]}")
   print(f'cat_cols: {len(cat_cols)}')
   print(f'num_cols: {len(num_cols)}')
   print(f'cat_but_car: {len(cat_but_car)}')
   print(f'num_but_cat: {len(num_but_cat)}')


   return cat_cols, num_cols, cat_but_car, num_but_cat


cat_cols, num_cols, cat_but_car,  num_but_cat = grab_col_names(df3)

cat_cols

num_cols

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

  cat_summary(df3,"Diabetes_012", plot=True)

# If there were more than one categorical variable, we would loop through all categorical variables one by one as follows to run the function.

for col in cat_cols:
    cat_summary(df3, col, plot=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

  if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

  for col in num_cols:
    num_summary(df3, col, plot=True)
