#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math
import pandas as pd
import numpy as np


# In[2]:


r = 102317167


# In[3]:


data = pd.read_csv(r"C:\Users\ASUS\Downloads\data.csv",encoding="latin1")


# In[4]:


no2 = data["no2"].dropna()


# In[5]:


Ar = 0.05 * (r % 7)
Br = 0.3 * ((r % 5) + 1)


# In[6]:


print(Ar)


# In[7]:


print(Br)


# In[8]:


x = no2.values
z = x + Ar * np.sin(Br * x)


# In[9]:


# Step 2
mean = np.mean(z)
var = np.var(z)
std = np.sqrt(var)


# In[11]:


lambda1 = 1.0 / (2.0 * var)
c = 1.0 / (std * np.sqrt(2.0 * np.pi))


# In[12]:


print(f"Lambda  : {lambda1}")
print(f"Mu      : {mean}")
print(f"c       : {c}")


# In[ ]:




