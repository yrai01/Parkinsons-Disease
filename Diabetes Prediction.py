#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# The purpose is to predict whether the Pima Indian women shows signs of diabetes or not. We are using a dataset collected by 
# "National Institute of Diabetes and Digestive and Kidney Diseases" which consists of a number of attributes which would help us 
# to perform this prediction.
# 
# Constraints on data collection
# All patients whose data has been collected are females at least 21 years old of Pima Indian heritage

# ### Import Libraries and load dataset

# In[452]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[453]:


df = pd.read_csv('pima-indians-diabetes.csv')
df.head(10)


# preg = Number of times pregnant
# 
# plas = Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# 
# pres = Diastolic blood pressure (mm Hg)
# 
# skin = Triceps skin fold thickness (mm)
# 
# test = 2-Hour serum insulin (mu U/ml)
# 
# mass = Body mass index (weight in kg/(height in m)^2)
# 
# pedi = Diabetes pedigree function
# 
# age = Age (years)
# 
# class = Class variable (1:tested positive for diabetes, 0: tested negative for diabetes)

# In[454]:


df.dtypes


# Observation :
#     
# Our dataset contains int and float values 

# In[455]:


df.shape


# There are '0's in the data. Are they really valid '0's or they are missing values? Plasma, BP, skin thickness etc. these values 
# cannot be 0. look at column by column logically to understand this.

# In[456]:


df.columns


# In[457]:


df.head(5)


# Observation :
# 
# In pres column some values are founded as 0 which can not be true beacause A normal blood pressure level is less than 120/80 mmHg nut never 0 mmHg.
# 
# In skin column some values are founded as 0 which can not be true beacause which is 23 but never 0.
# 
# In test column some values are founded as 0 which can not be true beacause A normal measurement of free insulin is less than 17 mcU/mL but never 0 mcU/mL.
# 
# In mass column some values are founded as 0 which can not be true beacause A normal measurement of mass can be 
# 18.5 - 25 kg/m2 but never 0 kg/m2.
# 
# In age column some values are founded as 0 which can not be true beacause we can not count age as 0.
# 

# In[458]:


df[['Plas', 'Pres', 'skin', 'test', 'mass', 'pedi', 'age']]==0


# Observation :
# 
# The "True" values in this table shows that, at that particular number "0" value is founded.

# In[459]:


df[['Plas', 'Pres', 'skin', 'test', 'mass', 'pedi', 'age']].median()


# Median = [(n/2)th term + {(n/2)+1}th term]/2
# 
# With the help of median values of given columns we will replace "0" values with their Corresponding column.

# In[460]:


df['Plas'] = df['Plas'].replace(0,117)


# Observation :
#     
# Replace Plas's 0 values with 117.

# In[461]:


df['Pres'] = df['Pres'].replace(0,72)


# Observation :
#     
# Replace Pres's 0 values with 72.

# In[462]:


df['skin'] = df['skin'].replace(0,23)


# Observation :
#     
# Replace skin's 0 values with 23.

# In[463]:


df['test'] = df['test'].replace(0,30.5000)


# Observation :
#     
# Replace test's 0 values with 30.5000.

# In[464]:


df['mass'] = df['Plas'].replace(0,32.0000)


# Observation :
#     
# Replace mass's 0 values with 32.0000.

# In[465]:


df['pedi'] = df['pedi'].replace(0,0.3725)


# Observation :
#     
# Replace pedi's 0 values with 0.3725.

# In[466]:


df['age'] = df['age'].replace(0,29)


# Observation :
#     
# Replace age's 0 values with 29.

# In[467]:


df.describe()


# Minimum and Maximum values of 'Preg' is respectively 0,17 and mean is 3.845052
# 
# Minimum and Maximum values of 'Plas' is respectively 44,199 and mean is 121.656250
# 
# Minimum and Maximum values of 'Pres' is respectively 24,122 and mean is 72.386719
# 
# Minimum and Maximum values of 'skin' is respectively 7,99 and mean is 27.334635
# 
# Minimum and Maximum values of 'test' is respectively 14,846 and mean is 94.652344
# 
# Minimum and Maximum values of 'mass' is respectively 44,199 and mean is 121.656250
# 
# Minimum and Maximum values of 'mass' is respectively 0.078,2.42 and mean is 0.471876
# 
# Minimum and Maximum values of 'age' is respectively  21,81 and mean is 33.240885
# 
# Minimum and Maximum values of 'class' is respectively 0,1 and mean is 0.348958
# 
# Note : The mean is the sum of all values divided by the total number of values

# In[468]:


df['class'].value_counts().plot.bar()


# Observation :
#     
# This bar chart describes that more than 450 patients tested negative for diabities and hardly 300 patients tested positive for diabities.

# In[469]:


sns.pairplot(df)


# This pairplot shows that perfect correlation is founded between mass an Plas column.

# Check for correlation between variables whose values are >0.8

# In[470]:


sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)


# Observation :
#     
# Only one correlation between Plas and mass is founded more than 0.8
# 
# Second highest correlation founded between Preg and age which is 0.54
# 
# Lowest correlation founded between Pedi and Pres which is -0.0024

# In[471]:


#Apply train_test_split on dataframe and take 70% data as a input and 30% data as a output 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3 , random_state=10)


# Divide the whole data in 70:30 ratio for trainig and testing data.
# 
# set random state = 10

# In[472]:


#Top 5 values of X_train dataset
X_train.head()


# Import MinMaxScaler

# In[478]:


from sklearn.preprocessing import MinMaxScaler


# MinMaxScaler is use to Transform features by scaling each feature to a given range

# In[479]:


scaler = MinMaxScaler()


# Store MinMaxScaler in scaler 

# In[480]:


X_train = scaler.fit_transform(X_train)


# Perform fit and transform operation on X_train data.

# In[481]:


#Print X_train
X_train


# In[483]:


X_test = scaler.transform(X_test)


# Perform transform operation on X_test

# In[484]:


X_test


# Perform transform on X_test

# In[440]:


#Print X_test 
X_test


# In[420]:


y_train


# In[427]:


y_test


# Import DecisionTreeClassifier 

# In[401]:


from sklearn.tree import DecisionTreeClassifier


# Create a entropy with the help of criterion='entropy'on our datset.
# and fit it on X_train,y_train 

# In[402]:


clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train,y_train)


# In the context of Decision Trees, entropy is a measure of disorder or impurity in a node

# In[421]:


#Find accuracy on X_test,y_test 
accuracy = clf.score(X_test,y_test)


# In[422]:


print('Accuracy',accuracy)


# ### Observation :
# 
# #### Accuracy = 71%
# 
# Model tunned the accuracy around 71%.

# ### Model Validation

# In[448]:


y_pred = clf.predict([[1,85,30.5000,26.6,0.351,31]])


# Given values are predicted value of 'Preg',Plas','test','mass','pedi','age' columns respectively.

# In[450]:


y_pred


# According to given data our model predict that patient is tested negative in diabities.
