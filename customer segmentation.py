#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[17]:


df=pd.read_excel("E:\\dataset\\customer.csv.xlsx")
print(df)
print(df.shape)
print(df.isnull().sum())
x=df.iloc[:,[3,4]].values
print(x)


# In[18]:


plt.scatter(x[...,0],x[...,1])
plt.xlabel("Total Income")
plt.ylabel("Spending score")
plt.show()


# In[20]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
wcss


# In[22]:


plt.plot(range(1,11),wcss)
plt.title("The Elblow Method")
plt.xlabel("Number of Cluster")
plt.ylabel("WCSS")
plt.show()


# In[25]:


kmeans=KMeans(n_clusters=4,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x)
y_kmeans
y_kmeans==0
[x[y_kmeans]==0,0]
[x[y_kmeans]==0,1]


# In[26]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],label='cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,0],label='cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,0],label='cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,0],label='cluster 4')
plt.xlabel('Annual Income ')
plt.ylabel('Spending score')
plt.legend()
plt.show()


# In[27]:


df['Target']=y_kmeans
print(df)


# In[ ]:




