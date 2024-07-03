#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('Income.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.dtypes


# In[7]:


df.isna().sum()


# In[8]:


for i in df.columns:
    print(df[i].value_counts())
    print('\n')


# although there is no null values in the dataset there is some missing values  present in the form of ? 
# we consider it as null values

# In[9]:


df.replace(' ?',np.nan,inplace=True)


# In[10]:


df.isna().sum()


# In[11]:


for i in df.columns:
    print(df[i].value_counts())
    print('\n')


# there are some null values are present in Occupation,Workclass and Native_country columns
#  Filling these columns by using mode 

# In[12]:


df['Workclass']=df['Workclass'].fillna(df['Workclass'].mode()[0])


# In[13]:


df['Native_country']=df['Native_country'].fillna(df['Native_country'].mode()[0])


# In[14]:


df['Occupation']=df['Occupation'].fillna(df['Occupation'].mode()[0])


# In[15]:


df.isna().sum()


# Upto here i worked on Null values 
# 
# working on analysis

# In[16]:


sns.pairplot(df)


# In[17]:


sns.histplot(data=df,x='Age',bins=10)


# In[18]:


sns.countplot(data=df, x='Education', hue='Income')
plt.xticks(rotation=45)


# from the above graph those who completed the Doctorates,prof school,Masters are getting greater than 50K \
# the most of the people who did Bachelors Degree are getting >=50k

# In[19]:


sns.countplot(data=df, x='Occupation', hue='Income')
plt.xticks(rotation=45)


# In[20]:


sns.countplot(data=df, x='Workclass', hue='Income')
plt.xticks(rotation=45)


# In[21]:


sns.countplot(data=df, x='Marital_status', hue='Income')
plt.xticks(rotation=45)


# In[22]:


sns.countplot(data=df, x='Relationship', hue='Income')
plt.xticks(rotation=45)


# In[23]:


sns.countplot(data=df, x='Sex', hue='Income')
plt.xticks(rotation=45)


# In[24]:


sns.boxplot(data=df,x='Income', y='Capital_loss',)
plt.xticks(rotation=45)


# In[25]:


sns.boxplot(data=df,x='Income',y='Capital_gain')
plt.xticks(rotation=45)


# In[26]:


sns.boxplot(data=df,x='Income',y='Hours_per_week')
plt.xticks(rotation=45)


# In[27]:


df['Race'].value_counts()


# In[28]:


df['Native_country'].value_counts()


# In[29]:


sns.countplot(data=df, x='Race', hue='Income')
plt.xticks(rotation=45)


# from the above graphs we can conclude that workclass,education and occupation is highly co related with the income
# we can drop the education column as education num represents same data
# Capital_gain and Capital_gain also corelated with the imcome
# if the imcome is >=50k mostlikely to capital_gain,<50k capital_loss is mostly likely

# In[30]:


df


# In[31]:


sns.scatterplot(data=df,x='Income',y='Hours_per_week')


# In[32]:


sns.distplot(data=df,x='Hours_per_week',y='Income')


# In[33]:


sns.histplot(data=df, x='Hours_per_week', hue='Income', multiple='stack', bins=10)


# In[34]:


sns.histplot(data=df, x='Age', hue='Income', multiple='stack', bins=10)


# In[35]:


sns.histplot(data=df, x='Capital_gain', hue='Income', multiple='stack', bins=10)



# In[36]:


sns.boxplot(data=df,x='Income',y='Fnlwgt')


# In[37]:


sns.violinplot(data=df, x='Income', y='Fnlwgt')


# In[38]:


sns.scatterplot(data=df, x='Income', y='Fnlwgt')


# In[ ]:





# we can drop the fnlwgt column as well as it doesnt related to population related target

# In[39]:


df.drop(columns=['Fnlwgt','Education','Race'],inplace=True)


# In[40]:


df


# In[41]:


df['Income'].value_counts()


# its a imbalnced data set

# In[42]:


from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[43]:


le=LabelEncoder()


# In[44]:


col=['Workclass','Marital_status','Occupation','Relationship','Sex','Native_country','Income']
for i in col:
    df[i]=le.fit_transform(df[i])


# In[45]:


sns


# In[46]:


plt.figure(figsize=(10,6))
p=1
for i in df:
    if p<13:
        plt.subplot(5,4,p)
        sns.boxplot(df[i])
    p+=1
plt.show()


# In[47]:


X=df.drop(columns='Income')
y=df['Income']


# In[48]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[49]:


smt=SMOTE(random_state=42)


# In[50]:


X_trainre,y_trainre=smt.fit_resample(X_train,y_train)


# In[51]:


from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score,GridSearchCV


# In[52]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[53]:


model=[
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    LogisticRegression()
]


# In[54]:


for i in model:
    i.fit(X_trainre,y_trainre)
    y_pred=i.predict(X_test)
    print(i)
    print(accuracy_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    


# RandomForest classifier gives the best accuracy 

# Hyper Parameter Tuning

# In[55]:


params={
    'n_estimators':[10,30,50,70,100],
    'criterion':['gini', 'entropy', 'log_loss'],
    'max_features':['sqrt','log2','None']
}


# In[56]:


grid_search=GridSearchCV(estimator=RandomForestClassifier(),param_grid=params,cv=30)


# In[57]:


grid_search.fit(X_trainre,y_trainre)


# In[58]:


print(grid_search.best_estimator_)
print(grid_search.best_params_)


# In[ ]:





# In[ ]:





# In[ ]:




