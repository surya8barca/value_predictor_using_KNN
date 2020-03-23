#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('Classified Data',index_col=0)


# In[3]:


df.head()


# In[4]:


from sklearn.preprocessing import StandardScaler


# In[5]:


ss=StandardScaler()


# In[6]:


newdf=df.drop('TARGET CLASS',axis=1)


# In[7]:


ss.fit(newdf)


# In[8]:


df2=ss.transform(newdf)


# In[12]:


newdf.columns


# In[13]:


final_df=pd.DataFrame(df2,columns=newdf.columns)


# In[14]:


final_df.head()


# In[15]:


x=final_df


# In[16]:


y=df['TARGET CLASS']


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5)


# In[19]:


from sklearn.neighbors import KNeighborsClassifier


# In[20]:


knn=KNeighborsClassifier()


# In[21]:


knn.fit(x_train,y_train)


# In[22]:


pred=knn.predict(x_test)


# In[23]:


from sklearn.metrics import classification_report,confusion_matrix


# In[24]:


print(confusion_matrix(y_test,pred))


# In[25]:


print(classification_report(y_test,pred))


# In[26]:


#k value search using elbow method


# In[27]:


error_rate=[]

for i in range(1,50):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i=knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[31]:


plt.plot(range(1,50),error_rate,marker='o')


# In[32]:


knn=KNeighborsClassifier(n_neighbors=50)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
print(confusion_matrix(y_test,pred))
print()
print(classification_report(y_test,pred))


# In[33]:


print(classification_report(y_test,pred))


# In[ ]:




