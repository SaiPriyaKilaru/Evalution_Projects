#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')


# Loading The Data Sets

# In[67]:


ind=pd.read_csv('IND')


# In[68]:


ind.shape


# In[69]:


ind.info()


# In[5]:


ind.isna().sum()


# In[6]:


for col in ind.columns:
    print(ind[col].value_counts())
    print('\n')


# In[71]:


ind['longitude']=ind['longitude'].interpolate()
ind['latitude']=ind['latitude'].interpolate()


# In[8]:


ind['commissioning_year']=ind['commissioning_year'].fillna(ind['commissioning_year'].median())


# In[72]:


ind.isna().sum()


# In[ ]:





# In[ ]:





# In[73]:


aus=pd.read_csv('AUS')


# In[74]:


aus.shape


# In[75]:


aus.info()


# In[76]:


aus['commissioning_year']=aus['commissioning_year'].fillna(aus['commissioning_year'].median())


# In[77]:


aus=aus.dropna(subset='capacity_mw')


# In[78]:


aus.shape


# In[79]:


usa=pd.read_csv('USA')


# In[80]:


usa.shape


# In[81]:


usa.isna().sum()


# In[82]:


usa['commissioning_year']=usa['commissioning_year'].fillna(usa['commissioning_year'].median())


# In[83]:


usa['latitude']=usa['latitude'].interpolate()
usa['longitude']=usa['longitude'].interpolate()


# In[84]:


usa['primary_fuel']=usa['primary_fuel'].fillna(usa['primary_fuel'].mode()[0])


# Mergeing the data

# In[85]:


frames=[ind,usa,aus]
df=pd.concat(frames)


# In[ ]:





# In[86]:


pd.set_option('display.max_columns',100,'display.max_rows',600)


# In[ ]:





# In[87]:


df.shape


# In[88]:


df.info()


# Working on Cleaning the data to be able to interpret the details

# we can drop some columns form the data as we cant get any information from those columns like country and country_long 

# In[89]:


df.drop(columns=['country_long','gppd_idnr','url','geolocation_source','wepp_id','estimated_generation_gwh'],axis=1,inplace=True)


# In[90]:


df.info()


# In[91]:


df['other_fuel1']=df['other_fuel1'].fillna('NONE')


# In[92]:


df['other_fuel2']=df['other_fuel2'].fillna('NONE')
df['other_fuel3']=df['other_fuel3'].fillna('NONE')


# In[93]:


df


# In[94]:


df['year_of_capacity_data'].value_counts()


#  we can drop the some of the columns like  year_of_capacity_data because it has only one value in the  rows so we can drop that

# In[95]:


df.drop(columns='year_of_capacity_data',inplace=True)


# In[96]:


df.isna().sum()


# In[97]:


df.drop(columns='owner',inplace=True)


# In[98]:


df['generation_gwh_2013'] = df['generation_gwh_2013'].fillna(df['generation_gwh_2013'].mean())
df['generation_gwh_2014'] = df['generation_gwh_2014'].fillna(df['generation_gwh_2014'].mean())
df['generation_gwh_2015'] = df['generation_gwh_2015'].fillna(df['generation_gwh_2015'].mean())
df['generation_gwh_2016'] = df['generation_gwh_2016'].fillna(df['generation_gwh_2016'].mean())
df['generation_gwh_2017'] = df['generation_gwh_2017'].fillna(df['generation_gwh_2017'].mean())
df['generation_gwh_2018'] = df['generation_gwh_2018'].fillna(df['generation_gwh_2018'].mean())


# In[99]:


df.dtypes


# In[100]:


df['generation_gwh_2019']=df['generation_gwh_2019'].fillna(0)


# In[101]:


df['generation_gwh_2019'].dtypes


# In[102]:


df.dtypes


# In[103]:


df['generation_gwh_2019'] = pd.to_numeric(df['generation_gwh_2019'], errors='coerce')


# In[104]:


df['Total Generation']=df['generation_gwh_2019']+df['generation_gwh_2013']+df['generation_gwh_2014']+df['generation_gwh_2015']+df['generation_gwh_2016']+df['generation_gwh_2017']+df['generation_gwh_2018']


# In[105]:


df.drop(columns=['generation_gwh_2019','generation_gwh_2018','generation_gwh_2017','generation_gwh_2016','generation_gwh_2015','generation_gwh_2014','generation_gwh_2013'],inplace=True)


# In[106]:


df


# In[107]:


df.corr()


# univarient/bivarient analysis

# In[108]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=df,x='primary_fuel',y='capacity_mw')


# In[109]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=df,x='commissioning_year',y='capacity_mw')


# in early 90s generating power is less compared to late 90s

# In[110]:


df


# for classification problem
# Primary fuel

# In[111]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


# In[112]:


le=LabelEncoder()


# In[113]:


df['country']=le.fit_transform(df['country'])


# In[114]:


df['primary_fuel']=le.fit_transform(df['primary_fuel'])


# In[115]:


df['other_fuel1']=le.fit_transform(df['other_fuel1'])
df['other_fuel2']=le.fit_transform(df['other_fuel2'])
df['other_fuel3']=le.fit_transform(df['other_fuel3'])


# In[116]:


df['generation_data_source']=le.fit_transform(df['generation_data_source'])


# In[117]:


df['source']=le.fit_transform(df['source'])


# In[118]:


df.drop(columns='name',inplace=True)


# In[119]:


df


# In[134]:


df['Total Generation'] = df['Total Generation'].fillna(0)


# In[136]:


df['commissioning_year']=df['commissioning_year'].fillna(0)


# In[137]:


df.isna().sum()


# In[138]:


X=df.drop(columns='primary_fuel')
y=df['primary_fuel']


# In[139]:


s=StandardScaler()
s.fit_transform(X)


# In[140]:


X=pd.DataFrame(X,columns=X.columns)


# In[141]:


y=df['primary_fuel']


# In[142]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[143]:


model=[
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    LogisticRegression()
]


# In[144]:


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


# Random Forest Classifier gives the best accuracy 

# In[146]:


params={
    'n_estimators':[100,10,30,40,60,70,80],
    'criterion':['gini','entropy','log_loss'],
    'max_depth':['None',10,20],
    'max_features':['sqrt','log2','None'],
    
}


# In[147]:


grid_search=GridSearchCV(estimator=RandomForestClassifier(),param_grid=params,n_jobs=-2,cv=10)


# In[148]:


grid_search.fit(X,y)


# In[149]:


grid_search.best_estimator_


# In[150]:


grid_search.best_params_


# In[151]:


rf=RandomForestClassifier(criterion='entropy',max_depth=20,max_features='sqrt',n_estimators=100)


# In[152]:


rf.fit(X_train,y_train)


# In[153]:


y_pred=rf.predict(X_test)


# In[154]:


print(accuracy_score(y_test,y_pred))


# In[155]:


print(confusion_matrix(y_test,y_pred))


# In[156]:


print(classification_report(y_test,y_pred))


# Regression Problem

# In[158]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import xgboost as xgb


# In[159]:


model=[
    DecisionTreeRegressor(),
    ExtraTreeRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor(),
    Ridge(),
    Lasso()
]


# In[162]:


y1=df['capacity_mw']


# In[165]:


X_train,X_test,y_train,y_test=train_test_split(X,y1,test_size=0.3,random_state=42)


# In[166]:


for i in model:
    i.fit(X_train,y_train)
    y_pred=i.predict(X_test)
    print(i)
    print(r2_score(y_test,y_pred))
    print(mean_squared_error(y_test,y_pred))
    score = cross_val_score(i,X_train, y_train)
    print(score)
    print(score.mean())
    print("Difference btw accurancy and CV score is  ",r2_score(y_test, y_pred) - score.mean())
    print('\n')
    


# Ridge has 100percenta accuracy

# In[167]:


r=Ridge()


# In[168]:


r.fit(X_train,y_train)


# In[169]:


y_pred=r.predict(X_test)


# In[170]:


print(r2_score(y_test,y_pred))


# In[ ]:





# In[ ]:




