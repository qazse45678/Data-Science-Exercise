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
