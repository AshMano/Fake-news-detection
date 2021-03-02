#!/usr/bin/env python
# coding: utf-8

# ## Fake news
# 
# ### Goal : Build a model that can accurately detect whether a piece of news is fake or real.
# 
# **What is a fake news ?**  
# 
# False information disseminated with the aim of manipulating the public
# 
# **TfidfVectorizer, PassiveAgressiveClassifier and machine learning classification algorithms**  
# - TF (Term Frequency) : the number of times that a word appears in a document.  
# - IDF (Inverse Document Frequency) : mesure of how significant a term is in the entire corpus.  
# - TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features.  
#  

# In[98]:


import numpy as np
import pandas as pd
import itertools

import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[165]:


#Read the data
df=pd.read_csv('news.csv')

#Get shape and head
df.shape


# In[166]:


df.head(10)


# In[76]:


#Get the labels
labels=df.label
labels.head()

#Remove useless column
df = df.drop(["Unnamed: 0","title"],axis=1)
df = df.sample(frac=1)
df.head(10)

df.isna().sum()

#Funtion to remove unnecessary character 
def word_drop(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

df["text"] = df["text"].apply(word_drop)
df.head(10)


# In[88]:


#Define dependent and independent variable as x and y
x = df["text"]
y = df["label"]


# In[114]:


#Split the dataset into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2)


# In[115]:


#Initialize a TfidfVectorizer to convert text to vector 
tfidf_vectorizer=TfidfVectorizer()
#stop word is word that is so common that it is unnecessary to index it or use it in a search 
#max_df argument removes words which appear in more than 70% of the document


# In[117]:


#Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[144]:


#Initialize a PassiveAggressiveClassifier
PAC=PassiveAggressiveClassifier(max_iter=50)
PAC.fit(tfidf_train,y_train)


# In[145]:


#Predict on the test set and calculate accuracy
pred_PAC=PAC.predict(tfidf_test)
score=accuracy_score(y_test,pred_PAC)
print(f'Accuracy: {round(score*100,2)}%')


# In[93]:


#We got an accuracy of 93.84% with this model


# In[146]:


#Build confusion matrix
confusion_matrix(y_test,pred_PAC, labels=['FAKE','REAL'])


# In[157]:


#Our model successfully predicted 585 positives.
#Our model successfully predicted 604 negatives.
#Our model predicted 39 false positives.
#Our model predicted 39 false negatives.


# In[137]:


print(classification_report(y_test, pred_PAC))


# #### Let's now use four machine learning algorithms to solve the fake news detection problem

# ### 1. Logistic Regression

# In[121]:


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(tfidf_train,y_train)


# In[123]:


pred_lr=LR.predict(tfidf_test)
LR.score(tfidf_test, y_test)


# In[124]:


print(classification_report(y_test, pred_lr))


# ### 2. Decision Tree Classification

# In[125]:


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(tfidf_train, y_train)


# In[126]:


pred_dt=DT.predict(tfidf_test)
LR.score(tfidf_test, y_test)


# In[107]:


print(classification_report(y_test, pred_dt))


# ### 3. Gradient Boosting Classifier

# In[127]:


from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(tfidf_train, y_train)


# In[128]:


pred_gbc = GBC.predict(tfidf_test)
GBC.score(tfidf_test, y_test)


# In[129]:


print(classification_report(y_test, pred_gbc))


# ### 4. Random Forest Classifier

# In[130]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(random_state=0)
RFC.fit(tfidf_train, y_train)


# In[131]:


pred_rfc = RFC.predict(tfidf_test)
RFC.score(tfidf_test, y_test)


# In[132]:


print(classification_report(y_test, pred_rfc))


# ## Model testing with manual entry

# In[161]:


def output_label(n):
    if n == 'FAKE':
        return "Fake News"
    elif n == 'REAL':
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(word_drop) 
    new_x_test = new_def_test["text"]
    new_tfidf_test = tfidf_vectorizer.transform(new_x_test)
    pred_PAC = PAC.predict(new_tfidf_test)
    pred_LR = LR.predict(new_tfidf_test)
    pred_DT = DT.predict(new_tfidf_test)
    pred_GBC = GBC.predict(new_tfidf_test)
    pred_RFC = RFC.predict(new_tfidf_test)
    
    return print("\nPAC Prediction: {} \nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_label(pred_PAC[0]),output_label(pred_LR[0]), 
                                                                                                              output_label(pred_DT[0]), 
                                                                                                              output_label(pred_GBC[0]), 
                                                                                                              output_label(pred_RFC[0])))


# In[162]:


news = str(input())
manual_testing(news)


# In[ ]:




