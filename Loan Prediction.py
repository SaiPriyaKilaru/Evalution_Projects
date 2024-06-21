#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('Loan.csv')


# In[3]:


df


# In[4]:


df['Loan_Status  ']


# This data set has 614 rows and 12 columns

# In[5]:


df.isna().sum()


# In[6]:


df.columns


# In[7]:


df.nunique()


# In[8]:


for i in df.columns:
    print(df[i].value_counts())


# In[9]:


null_rows=df[df.isnull().any(axis=1)]
print(null_rows)


# Handling Null values

# In[10]:


df.replace(" ",'')


# In[11]:


df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])


# In[12]:


df['Married']=df['Married'].fillna(df['Married'].mode()[0])


# In[13]:


df['Self_Employed'].isna().sum()


# In[14]:


df['Dependents']=df['Dependents'].replace('3+','3').fillna(1).astype(int)


# In[15]:


df['Self_Employed'].value_counts()


# In[16]:


df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])


# In[17]:


df['Loan_Amount']=df['Loan_Amount'].fillna(df['Loan_Amount'].median())


# In[18]:


df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median())


# In[19]:


df['Credit History']=df['Credit History'].fillna(df['Credit History'].mode()[0])


# In[20]:


df.isna().sum()


# Upto here we handedled Null values in the dataset

# as the loan id column doesnt influence the target variable we can drop that 

# In[21]:


df.drop(columns='Loan_ID',inplace=True)


# In[22]:


df


# In[27]:


plt.figure(figsize=(20,10))
sns.histplot(data=df,x='Applicant Income',hue='Loan_Status  ',)


# In[28]:


plt.figure(figsize=(20,10))
sns.histplot(data=df,x='Gender',hue='Loan_Status  ',)


# In[29]:


plt.figure(figsize=(20,10))
sns.histplot(data=df,x='Education',hue='Loan_Status  ',)


# In[30]:


plt.figure(figsize=(20,10))
sns.histplot(data=df,x='Credit History',hue='Loan_Status  ',)


# In[32]:


df.boxplot(column='Applicant Income',by="Gender");


# males are earning more than females

# In[33]:


df.boxplot(column='Applicant Income',by="Education");


# In[96]:


sns.violinplot(data=df,x='Education',y='Applicant Income',hue='Loan_Status  ',kind='violin')


# graduates are earning more 

# In[98]:


sns.stripplot(data=df,x='Education',y='Applicant Income',hue='Loan_Status  ')


# In[35]:


df.boxplot(column='Loan_Amount',by="Gender");


# In[36]:


df.boxplot(column='Loan_Amount',by="Applicant Income");


# In[51]:


plt.figure(figsize=(10, 6))
sns.histplot(df['Loan_Amount'], bins=30, kde=True)


# In[72]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


# In[45]:


le=LabelEncoder()


# In[46]:


colum=['Gender','Married','Education','Self_Employed','Loan_Status  ']


# In[47]:


for col in colum:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])


# In[48]:


df_dummies=pd.get_dummies(df['Property_Area'],prefix='Property_Area')


# In[49]:


df = pd.concat([df, df_dummies], axis=1)


# In[50]:


df.drop(columns='Property_Area',axis=1,inplace=True)


# In[54]:


x=df.drop(columns='Loan_Status  ')


# In[56]:


ss=StandardScaler()


# In[57]:


X=ss.fit_transform(x)


# In[58]:


X


# In[59]:


y=df['Loan_Status  ']


# In[61]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[63]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[66]:


model=[
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    LogisticRegression()
]


# In[69]:


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


# Logistic Regression gives the best accuracy

# In[80]:


params={
    'penalty':['l2', None],
    'dual':[True,False],
    'C':[100,0.1,0.01,10,1],
    'class_weight':['balanced','dict'],
    'solver':['lbfgs', 'liblinear', 'newton-cg',' newton-cholesky', 'sag', 'saga'],
    'multi_class':['auto', 'ovr', 'multinomial'],
    'max_iter':[10,20,50,90,100]
}


# In[81]:


l=RandomizedSearchCV(LogisticRegression(),params,cv=10,n_jobs=-2)


# In[82]:


l.fit(X_train,y_train)


# In[84]:


print(l.best_estimator_)
print(l.best_params_)
print(l.best_score_)


# In[86]:


lr=LogisticRegression(C=1, class_weight='balanced', max_iter=10, solver='sag')
lr.fit(X_train,y_train)
p=lr.predict(X_test)


# In[87]:


score=cross_val_score(lr,X,y,cv=10)


# In[88]:


print(accuracy_score(y_test, p))
print(confusion_matrix(y_test, p))


# In[91]:


score.mean()

