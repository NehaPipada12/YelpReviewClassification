#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import Counter
import re


# In[2]:


df = pd.read_csv('reviews.csv')


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


vocab = {}


# In[6]:


def initializeVocabulary():
    unkToken = '<UNK>'
    vocab['t_2_i'] = {}
    vocab['i_2_t'] = {}
    vocab['unkToken'] = unkToken
    idx = addToken(unkToken)
    vocab['unkTokenIdx'] = idx


# In[7]:


def addToken(token):
    if token in vocab['t_2_i']:
        idx = vocab['t_2_i'][token]
    else:
        idx = len(vocab['t_2_i'])
        vocab['t_2_i'][token] = idx
        vocab['i_2_t'][idx] = token
    return idx


# In[8]:


def addManyTokens(tokens):
    idxes = [addToken(token) for token in tokens]
    return idxes


# In[9]:


def lookUpToken(token):
    if vocab['unkTokenIdx']>=0:
        return vocab['t_2_i'].get(token,vocab['unkTokenIdx'])
    else:
        return vocab['t_2_i'][token]


# In[10]:


#Looking for token and rturn index
def lookUpIndex(idx):
    if idx not in vocab['i_2_t']:
        raise KeyError("the index (%d) is not there" % idx)
    return vocab['i_2_t'][idx]


# In[11]:


def vocabularyFromDataFrame(df,cutoff=25):
    initializeVocabulary()
    wordCounts = Counter()
    for r in df.review:
        for word in re.split('\W+',r):
            wordCounts[word] += 1
    for word,count in wordCounts.items():
        if count > cutoff:
            addToken(word)


# In[12]:


def vocabularyFromCorpus(Corpus,cutoff=25):
    initializeVocabulary()
    wordCounts = Counter()
    for doc in Corpus:
        for word in re.split('\W+',doc):
            wordCounts[word] += 1
    for word,count in wordCounts.items():
        if count > cutoff:
            addToken(word)


# In[13]:


df = pd.read_csv('reviews.csv')


# In[14]:


#vocabularyFromDataFrame(df)
Corpus = np.asarray(df['review'])
vocabularyFromCorpus(Corpus)


# In[15]:


lookUpToken('the')


# In[16]:


lookUpIndex(38)


# In[17]:


len(vocab['t_2_i'])


# In[18]:


def oneHotVector(token,N):
    oneHot = np.zeros((N,1))
    oneHot[lookUpToken(token)] = 1
    return oneHot


# In[19]:


N = len(vocab['t_2_i'])
token = 'the'
oneHot = oneHotVector(token,N)


# In[20]:


oneHot[38]


# In[21]:


def computeFeatures(doc,N):
    isFirst = True
    for token in doc:
        oneHot = oneHotVector(token,N)
        if isFirst:
            xF = oneHot
            isFirst = False
        else:
            xF = np.hstack((xF,oneHot))
    return np.mean(xF,axis=1)[:,np.newaxis]


# In[22]:


def computeFeatures_fast(doc,N):
    fv = np.zeros(N)
    numTokens = 0
    for token in doc:
        fv[lookUpToken(token)] += 1
        numTokens += 1
    return fv/numTokens


# In[23]:


def corpusToFeatureMatrix(Corpus,N):
    isFirst = True
    for doc in Corpus:
        fv = computeFeatures(doc,N)
        if isFirst:
            fM = fv
            isFirst = False
        else:
            fM = np.hstack((fM,fv))
    return fM.T


# In[24]:


def corpusToFeatureMatrix_fast(Corpus,N):
    fM = np.zeros((N,len(Corpus)))
    i = 0
    for doc in Corpus:
        fM[:,i] = computeFeatures_fast(doc,N)
        i+=1
    return fM.T


# In[25]:


get_ipython().run_line_magic('timeit', "fv = computeFeatures_fast(Corpus[0],len(vocab['t_2_i']))")


# In[26]:


get_ipython().run_line_magic('timeit', "fv = computeFeatures(Corpus[0],len(vocab['t_2_i']))")


# In[27]:


df = pd.read_csv('reviews.csv')
X = np.asarray(df['review'])
y = np.asarray(df['rating'])


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,shuffle=True)


# In[30]:


vocabularyFromCorpus(Xtrain)


# In[31]:


N = len(vocab['t_2_i'])
Xtrain_fM = corpusToFeatureMatrix_fast(Xtrain,N)
Xtest_fM = corpusToFeatureMatrix_fast(Xtest,N)


# In[32]:


Xtrain_fM.shape


# In[33]:


Xtest_fM.shape


# In[34]:


from sklearn.linear_model import LogisticRegression as clf
#from sklearn.naive_bayes import GaussianNB as clf
#from sklearn.ensemble import RandomForestClassifier as clf
#from sklearn.svm import SVC as clf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
sns.set()


# In[35]:


M = clf().fit(Xtrain_fM,ytrain)


# In[36]:


y_pred = M.predict(Xtest_fM)


# In[37]:


mat = confusion_matrix(ytest,y_pred)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,
           xticklabels=np.unique(y),yticklabels=np.unique(y))
plt.xlabel("True Label")
plt.ylabel("Predicted Label")


# In[38]:


Xtrain.shape


# In[ ]:




