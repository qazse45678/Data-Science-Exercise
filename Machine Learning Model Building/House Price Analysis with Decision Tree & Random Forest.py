#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opendatasets --quiet --upgrade


# In[2]:


import opendatasets as od
download_url = 'https://www.kaggle.com/datasets/dansbecker/home-data-for-ml-course'

od.download(download_url)


# # Load the file

# In[3]:


import pandas as pd

file_path = './home-data-for-ml-course/train.csv'
data = pd.read_csv(file_path)


# # Understand the summary of the file 
# 
# ## 1. what are the statstics metrics of the data?
# ## 2. How many/ what are the columns of the data?

# In[4]:


data.describe()


# In[5]:


data.columns


# In[6]:


data.head()


# # Clean out invalid data

# In[7]:


data.dropna()


# # Specify prediction target and features (X, y)

# In[8]:


y = data.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[features]


# # Split data into train and valid data

# In[9]:


from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3)


# # Specify and fit the model
# 
# ## model: decision tree

# In[10]:


from sklearn.tree import DecisionTreeRegressor

data_model = DecisionTreeRegressor(random_state = 1)
data_model.fit(train_X, train_y)


# # Make Predictions with Validation data

# In[24]:


prediction_y = data_model.predict(test_X)
print(prediction_y)


# # Calculate the Mean Absolute Error in Validation Data

# In[13]:


from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(prediction_y, test_y)
print(mae)


# # Find the optimal tree leaves based on lowest MAE

# In[31]:


def get_mae(max_leaf_nodes, train_X, test_X, train_y, test_y):
    
    data_model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    data_model.fit(train_X, train_y)
    prediction_y = data_model.predict(test_X)
    mae = mean_absolute_error(test_y, prediction_y)
    
    return mae

leaves = [5, 25, 50, 100, 250, 500]
score = {i: get_mae(i, train_X, test_X, train_y, test_y) for i in leaves}
best_leaf = min(score, key = score.get)

final_model = DecisionTreeRegressor(max_leaf_nodes = best_leaf, random_state=0)

