#!/usr/bin/env python
# coding: utf-8

# In[12]:


i=pd.read_csv('insurance1.csv')


# In[13]:


i=pd.read_csv('insurance1.csv')
i.head()


# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset = pd.read_csv('insurance.csv', skiprows = 5, header = None)
dataset.head(10)

X = dataset.iloc[:,0].values
X = X.reshape([X.shape[0], 1])
Y = dataset.iloc[:, -1].values
print(X)
print(Y)


# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 1)


# In[10]:


X_train


# In[11]:


Y_train


# In[12]:


X_mean = np.mean(X)
X_mean


# In[13]:


Y_mean = np.mean(Y)
Y_mean


# In[14]:


X_variance = X.var()
X_variance


# In[15]:


Y_variance = Y.var()
Y_variance


# In[18]:


def covariance(X, Y):
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    covar = 0.0
    for i in range(len(X)):
        covar += (X[i] - x_mean) * (Y[i] - y_mean)
    return covar/len(X)



covar_xy = covariance(X, Y)
print(f'Cov(X,Y): {covar_xy}')


# In[19]:


dataset.cov()


# In[20]:


def mse(y_true, y_pred):
    
    sq = ((y_true) - (y_pred)).astype('float')**2
    mse_value = np.mean(sq)
    
    return mse_value


# In[21]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)


# In[22]:


y_pred = regressor.predict(X_train)
y_pred


# In[23]:


plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, y_pred)
plt.xlabel('Claims')
plt.ylabel('Total payment')
plt.show()


# In[24]:


y_test_pred = regressor.predict(X_test)
y_test_pred


# In[25]:


plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_test, y_test_pred)
plt.xlabel('Claims')
plt.ylabel('Total payment')
plt.show()


# In[26]:


print(regressor.predict([[50.5]]))


# In[27]:


mse(Y_train, y_pred)


# In[28]:


from sklearn.metrics import mean_squared_error
mse_al = mean_squared_error(Y_train, y_pred, squared= True)
mse_al


# In[29]:


print("Coefficient of linear regression(b1): ", regressor.coef_)
print("Intercept of linear regressor(b0): ", regressor.intercept_)


# In[ ]:




