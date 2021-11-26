# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 23:42:57 2021

@author: Gourav Beura
"""
#Data Analysis and Manipulation
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


#Used for NLP
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#Imports for ML Algorithms
from sklearn.metrics import classification_report,accuracy_score
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings



#Create dataframe
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
print(train_df.head(5))
#Step 3: Project Planning
#Step 3.1: Understand nature of the data .info() .describe()
train_df.info()



'''
The steps involved in developing this project are:
1. Understanding the shape of data 
2. Data cleaning
3. Data Exploration
4. Feature Engineering
5. Data Processing for Model
6. Basic Model building
7. Model Tuning
8. Ensemble Model Building 
9. Results

Columns
 #   Column    Dtype 
---  ------    ----- 
 0   id        int64 
 1   keyword   object
 2   location  object
 3   text      object
 4   target    int64 
'''
print(train_df.describe())
print(train_df.columns)

#region Understanding the shape of data
'''
STEP 1: Understanding the shape of data
'''
# Visualizing the target classes
plt.title("Count of Target Classes")
sns.countplot(y=train_df["target"],linewidth=1,
                   edgecolor='black')

plt.show()

fig,(sbplt1,sbplt2) = plt.subplots(1,2,figsize=(10,5))
train_disaster_chars_len = train_df[train_df['target']==1]['text'].str.len()
sbplt1.hist(train_disaster_chars_len,color='red',edgecolor='black',linewidth=1)
sbplt1.set_title('Disaster Tweets')

train_nondisaster_chars_len = train_df[train_df['target']==0]['text'].str.len()
sbplt2.hist(train_nondisaster_chars_len,color='blue',edgecolor='black',linewidth=1)
sbplt2.set_title('Non-Disaster Tweets')

plt.suptitle('Tweet Length Distribution')
plt.tight_layout()
plt.show()
#endregion

#region Analysing number of words in text
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
char_len_dis = train_df[train_df['target']==1]['text'].str.split().map(lambda x: len(x))
ax1.hist(char_len_dis,color='red',edgecolor='black',linewidth=1)
ax1.set_title('Disaster Tweets')
char_len_ndis = train_df[train_df['target']==0]['text'].str.split().map(lambda x: len(x))
ax2.hist(train_nondisaster_chars_len,color='blue',edgecolor='black',linewidth=1)
ax2.set_title('Non-Disaster Tweets')
plt.suptitle("Length of words in text")
plt.tight_layout()
plt.show()
#endregion



