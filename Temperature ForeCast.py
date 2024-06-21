#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# In[48]:


df=pd.read_csv('Temperature')


# In[49]:


df


# In[50]:


pd.set_option('display.max_columns',100,'display.max_rows',100)


# In[51]:


df


# In[52]:


df['Date'].iloc[7750]='31-08-2017'
df['Date'].iloc[7751]='31-08-2017'


# In[53]:


df.info()


# In[54]:


df['station'].iloc[7750]=1.0
df['station'].iloc[7751]=2.0


# In[55]:


df['Date']=pd.to_datetime(df['Date'])


# In[56]:


df.info()


# In[57]:


df.index=df['Date']
     


# In[58]:


df.index.year.value_counts().sort_index()


# In[59]:


df.index.month.value_counts().sort_index()


# In[60]:


df.index.date


# In[61]:


df.describe()


# In[62]:


df['Date'].value_counts()


# In[63]:


df.nunique()


# In[64]:


columns_with_nan = df.columns[df.isna().any()].tolist()


# In[65]:


columns_to_check = [
    'LDAPS_RHmin', 'LDAPS_RHmax', 'LDAPS_Tmax_lapse', 'LDAPS_Tmin_lapse', 
    'LDAPS_WS', 'LDAPS_LH', 'LDAPS_CC1', 'LDAPS_CC2', 'LDAPS_CC3', 
    'LDAPS_CC4', 'LDAPS_PPT1', 'LDAPS_PPT2', 'LDAPS_PPT3', 'LDAPS_PPT4'
]

rows_with_nan = df[df[columns_to_check].isna().any(axis=1)]


print(rows_with_nan[columns_to_check])


# In[66]:


df.iloc[1023:1055,:]


# in some rows data is not present like LDAPS_RHmin,LDAPS_RHmax,LDAPS_Tmax_lapse,LDAPS_Tmin_lapse,LDAPS_WS,LDAPS_LH,LDAPS_CC1,LDAPS_CC2,LDAPS_CC3,LDAPS_CC4,LDAPS_PPT1,LDAPS_PPT2,LDAPS_PPT3,LDAPS_PPT4, so we can drop the rows

# In[67]:


df.columns


# In[68]:


cleaned_df=df.dropna(subset=['LDAPS_RHmax'])


# In[69]:


df


# before we have 7752 entries 25 columns but now after deleting some rows we have now 7677 rows and 25 columns are there
# we dont need station column and it doesnt impact temparature so we can drop both station and date column

# Still we have some null values we can fill them with the mean of the column

# In[70]:


cleaned_df['Present_Tmax']=cleaned_df['Present_Tmax'].fillna(cleaned_df['Present_Tmax'].mean())


# In[71]:


cleaned_df['Present_Tmin']=cleaned_df['Present_Tmin'].fillna(cleaned_df['Present_Tmin'].mean())


# In[72]:


cleaned_df['Next_Tmax']=cleaned_df['Next_Tmax'].fillna(cleaned_df['Next_Tmax'].mean())


# In[73]:


cleaned_df['Next_Tmin']=cleaned_df['Next_Tmin'].fillna(cleaned_df['Next_Tmin'].mean())


# In[74]:


cleaned_df.isna().sum()


# In[75]:


cleaned_df.corr()


# From the above corelation table we can observe that 
# Present_Tmax,Present_Tmin are highly corelated with the target variables
# LDAPS_CC1,LDAPS_CC2,LDAPS_CC3 and LDAPS_CC4 are redundent features as which they provide same information so we can drop couple of columns.
# as from the table lon lat are less corelated with target variable so we can drop them as well
# station column doesnt effect the target variable so we can drop that as well

# In[76]:


cleaned_df


# In[77]:


cleaned_df.isna().sum()


# Upto now we handeled the data if there is any null values 

# In[78]:


plt.figure(figsize=(15,30))
p=1
for col in cleaned_df.columns:
    if col not in ['station', 'Date']:  
        if p<25:
            plt.subplot(5,10,p)
            sns.boxplot(df[col])
            plt.xlabel(col)
        p+=1
plt.show()


# from the above box plots we can observe that some outliers are present in the dataset
# we can remove them by using zscore

# In[79]:


from scipy.stats import zscore


# In[ ]:


df


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[80]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=cleaned_df,x='Present_Tmax',y='Next_Tmax')


# In[81]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=cleaned_df,x='Present_Tmin',y='Next_Tmin')


# In[82]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=cleaned_df,x='LDAPS_RHmin',y='Next_Tmin')


# In[83]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=cleaned_df,x='LDAPS_RHmax',y='Next_Tmax')


# In[84]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=cleaned_df,x='LDAPS_WS',y='Next_Tmax')


# In[85]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=cleaned_df,x='LDAPS_LH',y='Next_Tmax')


# In[86]:


plt.figure(figsize=(10,6))
sns.catplot(data=cleaned_df,x='Date',y='LDAPS_LH',cmap='viridis')


# In[87]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=cleaned_df,x='Solar radiation',y='Next_Tmax')


# In[88]:


plt.figure(figsize=(10,6))
sns.scatterplot(data=cleaned_df,x='Solar radiation',y='Next_Tmin')


# In[89]:


cleaned_df.drop(columns='Date',inplace=True,axis=1)


# In[90]:


def threhold(z,d):
    for i in np.arange(3,5,0.3):
        data=d.copy()
        data=data[(z)]


# In[91]:


z=np.abs(zscore(cleaned_df))
cleaned_df=cleaned_df[(z<4.2).all(axis=1)]


# In[ ]:





# In[92]:


cleaned_df


# as we can see from the above 852 rows are having outliers if we remove these we loose most of the data  so we have to choose alternative method like scaling 

# In[93]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[96]:


x=cleaned_df.copy()
x.drop(['Next_Tmax','Next_Tmin'],axis=1,inplace=True)
ymax=cleaned_df['Next_Tmax']
ymin=cleaned_df['Next_Tmin']


# In[98]:


ss=scaler.fit_transform(x)
X=pd.DataFrame(ss,columns=x.columns)


# In[99]:


from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score


# In[88]:


scaled_df = pd.DataFrame(scaled_df, columns=column_names)


# In[86]:


column_names = ['Present_Tmax', 'Present_Tmin', 'LDAPS_RHmin', 'LDAPS_RHmax', 'LDAPS_Tmax_lapse', 'LDAPS_Tmin_lapse',
                'LDAPS_WS', 'LDAPS_LH', 'LDAPS_CC1', 'LDAPS_CC2', 'LDAPS_PPT1', 'LDAPS_PPT2', 'LDAPS_PPT3',
                'LDAPS_PPT4', 'DEM', 'Slope', 'Solar_radiation', 'Next_Tmax', 'Next_Tmin']


# In[89]:


scaled_df


# In[92]:


X=scaled_df.drop(columns=['Next_Tmax','Next_Tmin'])
y1=scaled_df['Next_Tmax']
y2=scaled_df['Next_Tmin']


# In[101]:


from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.svm import SVR


# In[104]:


X_train1,X_test1,y_train1,y_test1=train_test_split(X,ymax,test_size=0.2,random_state=42)

X_train2,X_test2,y_train2,y_test2=train_test_split(X,ymin,test_size=0.2,random_state=42)


# In[102]:


model=[
    DecisionTreeRegressor(),
    LinearRegression(),
    Ridge(),
    Lasso(),
    RandomForestRegressor(),
    SVR()
]


# In[105]:


for i in model:
    i.fit(X_train1,y_train1)
    i.fit(X_train2,y_train2)
    y_pred1=i.predict(X_test1)
    y_pred2=i.predict(X_test2)
    print(i)
    print(r2_score(y_test1,y_pred1))
    print(r2_score(y_test2,y_pred2))
    print(mean_squared_error(y_test1,y_pred1))
    print(mean_squared_error(y_test2,y_pred2))
    print('\n')


# In[106]:


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[112]:


grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)


# In[108]:


model=RandomForestRegressor()


# In[113]:


grid_search


# In[114]:


grid_search.fit(X, ymax)


# In[ ]:


best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# In[ ]:





# In[ ]:




