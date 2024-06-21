#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[140]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')


# Loading The Data Sets

# In[141]:


ind=pd.read_csv('IND')


# In[142]:


ind.shape


# In[143]:


ind.info()


# In[144]:


ind.isna().sum()


# In[145]:


for col in ind.columns:
    print(ind[col].value_counts())
    print('\n')


# In[146]:


ind['longitude']=ind['longitude'].interpolate()


# In[147]:


ind['commissioning_year']=ind['commissioning_year'].fillna(ind['commissioning_year'].median())


# In[148]:


ind.isna().sum()


# In[ ]:





# In[ ]:





# In[149]:


aus=pd.read_csv('AUS')


# In[150]:


aus.shape


# In[151]:


aus.info()


# In[152]:


aus['commissioning_year']=aus['commissioning_year'].fillna(aus['commissioning_year'].median())


# In[153]:


aus=aus.dropna(subset='capacity_mw')


# In[154]:


aus.shape


# In[155]:


usa=pd.read_csv('USA')


# In[156]:


usa.shape


# In[157]:


usa.isna().sum()


# In[158]:


usa['commissioning_year']=usa['commissioning_year'].fillna(usa['commissioning_year'].median())


# In[159]:


usa['latitude']=usa['latitude'].interpolate()
usa['longitude']=usa['longitude'].interpolate()


# In[160]:


usa['primary_fuel']=usa['primary_fuel'].fillna(usa['primary_fuel'].mode()[0])


# Mergeing the data

# In[161]:


frames=[ind,usa,aus]
df=pd.concat(frames)


# In[ ]:





# In[162]:


pd.set_option('display.max_columns',100,'display.max_rows',600)


# In[ ]:





# In[163]:


df.shape


# In[164]:


df.info()


# Working on Cleaning the data to be able to interpret the details

# we can drop some columns form the data as we cant get any information from those columns like country and country_long 

# In[165]:


df.drop(columns=['country_long','gppd_idnr','url','geolocation_source','wepp_id','estimated_generation_gwh'],axis=1,inplace=True)


# In[166]:


df.info()


# In[167]:


df['other_fuel1']=df['other_fuel1'].fillna('NONE')


# In[168]:


df['other_fuel2']=df['other_fuel2'].fillna('NONE')
df['other_fuel3']=df['other_fuel3'].fillna('NONE')


# In[169]:


df


# In[170]:


df['year_of_capacity_data'].value_counts()


#  we can drop the some of the columns like  year_of_capacity_data because it has only one value in the  rows so we can drop that

# In[171]:


df.drop(columns='year_of_capacity_data',inplace=True)


# In[172]:


df.isna().sum()


# In[173]:


df.drop(columns='owner',inplace=True)


# In[174]:


df['generation_gwh_2013'] = df['generation_gwh_2013'].fillna(df['generation_gwh_2013'].mean())
df['generation_gwh_2014'] = df['generation_gwh_2014'].fillna(df['generation_gwh_2014'].mean())
df['generation_gwh_2015'] = df['generation_gwh_2015'].fillna(df['generation_gwh_2015'].mean())
df['generation_gwh_2016'] = df['generation_gwh_2016'].fillna(df['generation_gwh_2016'].mean())
df['generation_gwh_2017'] = df['generation_gwh_2017'].fillna(df['generation_gwh_2017'].mean())
df['generation_gwh_2018'] = df['generation_gwh_2018'].fillna(df['generation_gwh_2018'].mean())


# In[175]:


df.dtypes


# In[176]:


df['generation_gwh_2019']=df['generation_gwh_2019'].fillna(0)


# In[177]:


df['generation_gwh_2019'].dtypes


# In[178]:


df.dtypes


# In[179]:


df['generation_gwh_2019'] = pd.to_numeric(df['generation_gwh_2019'], errors='coerce')


# In[180]:


df['Total Generation']=df['generation_gwh_2019']+df['generation_gwh_2013']+df['generation_gwh_2014']+df['generation_gwh_2015']+df['generation_gwh_2016']+df['generation_gwh_2017']+df['generation_gwh_2018']


# In[181]:


df.drop(columns=['generation_gwh_2019','generation_gwh_2018','generation_gwh_2017','generation_gwh_2016','generation_gwh_2015','generation_gwh_2014','generation_gwh_2013'],inplace=True)


# In[182]:


df


# In[183]:


df.corr()


# univarient/bivarient analysis

# In[184]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=df,x='primary_fuel',y='capacity_mw')


# In[185]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=df,x='commissioning_year',y='capacity_mw')


# in early 90s generating power is less compared to late 90s

# In[189]:


df


# for classification problem
# Primary fuel

# In[123]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


# In[190]:


le=LabelEncoder()


# In[191]:


df['country']=le.fit_transform(df['country'])


# In[193]:


df['primary_fuel']=le.fit_transform(df['primary_fuel'])


# In[206]:


df['other_fuel1']=le.fit_transform(df['other_fuel1'])
df['other_fuel2']=le.fit_transform(df['other_fuel2'])
df['other_fuel3']=le.fit_transform(df['other_fuel3'])


# In[207]:


df['generation_data_source']=le.fit_transform(df['generation_data_source'])


# In[208]:


df['source']=le.fit_transform(df['source'])


# In[195]:


df.drop(columns='values',inplace=True)


# In[197]:


df.drop(columns='name',inplace=True)


# In[209]:


df


# In[217]:


s=StandardScaler()
s.fit_transform(X)


# In[212]:


X=df.drop(columns='primary_fuel')
y=df['primary_fuel']


# In[213]:


y=df['primary_fuel']


# In[214]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[215]:


model=[
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    LogisticRegression()
]


# In[ ]:


for i in model:
    i.fit(X_train,y_train)
    y_pred=i.predict(X_test)
    print(i)
    print(accuracy_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    score = cross_val_score(i,X_train, y_train)
    print(score)
    print(score.mean())
    print("Difference btw accurancy and CV score is  ",accuracy_score(y_test, y_pred) - score.mean())
    print('\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# There are some missing values are present in the major columns like latitude longitude

# we are using geopy to fill null values in latitide and longitude columns

# In[17]:


missing_coordinates = ind[ind['latitude'].isna() | ind['longitude'].isna()]
print(missing_coordinates)


# In[18]:


for i in ind['name']:
    if (ind['latitude'],ind['longitude'])==NaN:
        return pd.Series([i.latitude, i.longitude])
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


what is interpolate


# In[ ]:





# In[25]:


ind['latitude']=ind['latitude'].interpolate()


# Mergeing the data

# In[26]:


ind['latitude'].isna().sum()


# In[37]:


df


# In[39]:


df.info()


# In[9]:


usa.info()


# In[10]:


ind.info()


# In[38]:


aus.info()


# from the above information about the data we can observe
# There is no values in estimated_generation_gwh,wepp_id columns hence, these columns are not providing any information so we can drop them.
# In other fuel 2&3 columns only less amount of data is provided and for predicting the primary fuel column we dont need them.
# 

# In[12]:


df.drop(columns=['estimated_generation_gwh','wepp_id'],axis=1,inplace=True)


# In[13]:


df


# In[14]:


df. isna().sum()


# In[15]:


df.replace(' ',"")


# In[16]:


df.isna().sum()


# In[17]:


df.describe()


# In[24]:


usa['primary_fuel'].value_counts()


# In[25]:


ind['primary_fuel'].value_counts()


# In[26]:


aus['primary_fuel'].value_counts()


# In[29]:


usa['primary_fuel']=usa['primary_fuel'].fillna(usa['primary_fuel'].mode()[0])


# In[31]:


ind['primary_fuel']=ind['primary_fuel'].fillna(ind['primary_fuel'].mode()[0])


# In[32]:


aus['primary_fuel']=aus['primary_fuel'].fillna(aus['primary_fuel'].mode()[0])


# In[41]:


df['primary_fuel'].isna().sum()


# In[ ]:


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

