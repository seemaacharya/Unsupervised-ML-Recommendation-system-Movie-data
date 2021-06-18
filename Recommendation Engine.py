l#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
import numpy as np


# In[5]:


movies_df = pd.read_csv('Movie.csv')


# In[7]:


movies_df[0:5]


# In[9]:


#number of unique users in the dataset
len(movies_df.userId.unique())


# In[10]:


len(movies_df.movie.unique())


# In[11]:


user_movies_df = movies_df.pivot(index='userId',
                                 columns='movie',
                                 values='rating').reset_index(drop=True)


# In[12]:


user_movies_df


# In[14]:


user_movies_df.index = movies_df.userId.unique()


# In[15]:


user_movies_df


# In[16]:


#Impute those NaNs with 0 values
user_movies_df.fillna(0, inplace=True)


# In[17]:


user_movies_df


# In[18]:


#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[19]:


user_sim = 1 - pairwise_distances( user_movies_df.values,metric='cosine')


# In[20]:


user_sim


# In[21]:


#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)


# In[22]:


#Set the index and column names to user ids 
user_sim_df.index = movies_df.userId.unique()
user_sim_df.columns = movies_df.userId.unique()


# In[24]:


user_sim_df.iloc[0:5, 0:5]


# In[28]:


np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]


# In[75]:


#Most Similar Users
user_sim_df.idxmax(axis=1)[0:5]


# In[42]:


movies_df[(movies_df['userId']==6) | (movies_df['userId']==168)]


# In[58]:


user_1=movies_df[movies_df['userId']==6]


# In[59]:


user_2=movies_df[movies_df['userId']==11]


# In[61]:


user_2.movie


# In[64]:


user_1.movie


# In[76]:


pd.merge(user_1,user_2,on='movie',how='outer')


# In[ ]:




