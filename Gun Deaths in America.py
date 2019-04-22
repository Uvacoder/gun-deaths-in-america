#!/usr/bin/env python
# coding: utf-8

# # Gun Deaths in America
# 
# The data is from [FiveThirtyEight's _Gun Deaths in America_ project](https://github.com/fivethirtyeight/guns-data). Source: CDC.
# 
# Author: Ken Norton

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Gun-Deaths-in-America" data-toc-modified-id="Gun-Deaths-in-America-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Gun Deaths in America</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Age-distribution:-by-Race" data-toc-modified-id="Age-distribution:-by-Race-1.0.1"><span class="toc-item-num">1.0.1&nbsp;&nbsp;</span>Age distribution: by Race</a></span></li><li><span><a href="#Age-distribution:-Homicide" data-toc-modified-id="Age-distribution:-Homicide-1.0.2"><span class="toc-item-num">1.0.2&nbsp;&nbsp;</span>Age distribution: Homicide</a></span></li><li><span><a href="#Age-distribution:-Suicide" data-toc-modified-id="Age-distribution:-Suicide-1.0.3"><span class="toc-item-num">1.0.3&nbsp;&nbsp;</span>Age distribution: Suicide</a></span></li></ul></li></ul></li></ul></div>

# In[1]:


import unicodecsv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
from pylab import rcParams
from scipy.stats import pearsonr


# In[2]:


plt.style.use('seaborn-talk')
plt.style.use('fivethirtyeight')


# In[3]:


guns = pd.read_csv('full_data.csv')
guns = pd.DataFrame.from_dict(guns)
guns.describe()


# In[4]:


guns.sample(5)


# In[5]:


guns.head()


# In[6]:


guns_duplicates = guns.duplicated()
print('Number of duplicate entries: {}'.format(guns_duplicates.sum()))


# In[7]:


guns.isnull().sum()


# In[8]:


missing_age = pd.isnull(guns['age'])
guns[missing_age].head()


# In[9]:


missing_intent = pd.isnull(guns['intent'])
guns[missing_intent].head()


# In[10]:


guns.describe()


# In[11]:


guns_clean = guns[['race', 'sex']].dropna()
guns_clean.groupby(['race', 'sex']).size().unstack(fill_value=0).plot.bar()


# In[12]:


plt.xticks(rotation=90)
sns.boxplot(x='race', y='age', data=guns)


# In[13]:


guns_clean = guns[['intent', 'race']].dropna()
guns_clean.groupby(['intent', 'race']).size().unstack(fill_value=0).plot.bar()


# In[14]:


guns_clean = guns[['place', 'race']].dropna()
guns_clean.groupby(['place', 'race']).size().unstack(fill_value=0).plot.bar()


# In[15]:


guns_clean = guns[['intent', 'place']].dropna()
guns_clean.groupby(['intent', 'place']).size().unstack(fill_value=0).plot.bar()


# In[16]:


ax = guns.groupby('race')['intent'].value_counts(normalize=True).unstack(level=1).plot.bar(stacked=True)
ax.legend(bbox_to_anchor=(1.1, 1.05))


# In[17]:


plt.xticks(rotation=90)
sns.boxplot(x='place', y='age', data=guns)


# In[18]:


plt.xticks(rotation=90)
sns.boxplot(x='intent', y='age', data=guns)


# ### Age distribution: by Race

# In[19]:


guns.race.unique()


# In[20]:


guns_filtered = guns[['race', 'age']].dropna()

for x in guns_filtered.race.unique():
    y = guns_filtered[guns_filtered['race'] == x]
    sns.distplot(y['age'], label = x)

plt.legend()
plt.show()


# In[21]:


guns_filtered = guns[['sex', 'age']].dropna()

for x in guns_filtered.sex.unique():
    y = guns_filtered[guns_filtered['sex'] == x]
    sns.distplot(y['age'], label = x)

plt.legend()
plt.show()


# In[23]:


x = guns[(guns['race'] == 'White') & (guns['sex'] == 'M')]
x = x['age']
plt.hist(x, density=True, bins=25)


# In[24]:


x = guns[(guns['race'] == 'Black') & (guns['sex'] == 'M')]
x = x['age']
plt.hist(x, density=True, bins=25)


# ### Age distribution: Homicide

# In[25]:


x = guns[['intent', 'age']].dropna()
x = guns[guns['intent'] == 'Homicide']
x = x['age']
plt.hist(x, density=True, bins=25)


# ### Age distribution: Suicide

# In[26]:


x = guns[['intent', 'age']].dropna()
x = guns[guns['intent'] == 'Suicide']
x = x['age']
plt.hist(x, density=True, bins=25)


# In[27]:


ax = guns.groupby('intent')['place'].value_counts(normalize=True).unstack(level=1).plot.bar(stacked=True)
ax.legend(bbox_to_anchor=(1.1, 1.05))


# In[28]:


ax = guns.groupby('race')['place'].value_counts(normalize=True).unstack(level=1).plot.bar(stacked=True)
ax.legend(bbox_to_anchor=(1.1, 1.05))


# In[29]:


ax = guns.groupby('race')['education'].value_counts(normalize=True).unstack(level=1).plot.bar(stacked=True)
ax.legend(bbox_to_anchor=(1.1, 1.05))


# In[30]:


ax = guns.groupby('education')['intent'].value_counts(normalize=True).unstack(level=1).plot.bar(stacked=True)
ax.legend(bbox_to_anchor=(1.1, 1.05))


# In[31]:


ax = guns.groupby('intent')['education'].value_counts(normalize=True).unstack(level=1).plot.bar(stacked=True)
ax.legend(bbox_to_anchor=(1.1, 1.05))


# In[32]:


ax = guns.groupby('sex')['intent'].value_counts(normalize=True).unstack(level=1).plot.bar(stacked=True)
ax.legend(bbox_to_anchor=(1.1, 1.05))


# In[33]:


ax = guns.groupby('intent')['sex'].value_counts(normalize=True).unstack(level=1).plot.bar(stacked=True)
ax.legend(bbox_to_anchor=(1.1, 1.05))


# In[34]:


ax = guns.groupby('race')['sex'].value_counts(normalize=True).unstack(level=1).plot.bar(stacked=True)
ax.legend(bbox_to_anchor=(1.1, 1.05))


# In[35]:


plt.xticks(rotation=90)
sns.violinplot(x="race", y="age", data=guns[guns['intent'] == 'Homicide'])


# In[36]:


plt.xticks(rotation=90)
sns.violinplot(x="race", y="age", data=guns[guns['intent'] == 'Suicide'])


# In[ ]:




