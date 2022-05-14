import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/marylin/文件/程式語言/Python/DS EDA Project 2022.05.14/covid-nationality.csv')

data.head()
data.columns
data.tail()
data.describe()
data.columns
sns.relplot(x = "total_cases", y = "recovered", data = data)
sns.relplot(x = "total_cases", y = "deaths", data = data)
sns.relplot(x = "total_cases", y = "total_confirmed_cases", data = data)
sns.relplot(x = "total_cases", y = "hospitalized_with_symptoms", data = data)
sns.relplot(x = "total_cases", y = "hospitalized_with_symptoms", hue = "recovered", data = data)
sns.pairplot(data)
data.columns
sns.relplot(x = "total_cases", y = "home_quarantine", kind = "line", data = data)
sns.relplot(x = "recovered", y = "home_quarantine", kind = "line", data = data)
data.columns
sns.catplot(x = "total_cases", y = "recovered", data = data)

#Three main conclusions in the project:
#1. More people are in recovery with the total number of cases if more people are under quarantined
#2. With more total cases, the number of hospitalized people is actually flattening
#3. The number of death is increasing
