#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# Loading the data set

# In[3]:


df=pd.read_csv("Rainfall.csv")


# In[4]:


pd.set_option('display.max_columns',100,'display.max_rows',300)


# In[5]:


df


# In[ ]:





# In[6]:


df.head()


# In[ ]:





# In[7]:


df.shape


# This data set contains 8425 records and 23 columns

# In[8]:


df.columns


# In[54]:


df.info()


# In[47]:


df['Date']=pd.to_datetime(df['Date'])


# In[10]:


numerical_features=[feature for feature in df.columns if df[feature].dtypes !='O']
discrete_features=[feature for feature in numerical_features if len(df[feature].unique())<25]
contineous_features=[feature for feature in numerical_features if feature not in discrete_features]
categorical_features=[feature for feature in df.columns if feature not in numerical_features ]


# In[11]:


print('numerical_features{}'.format(len(numerical_features)))
print("discrete_features{}".format(len(discrete_features)))
print("contineous_features{}".format(len(contineous_features)))
print("categorical_features{}".format(len(categorical_features)))


# filling the null values by using Random Simple Imputation method

# In[16]:


def randomsimpleimputation(df,variable):
    df[variable]=df[variable]
    random_sample=df[variable].dropna().sample(df[variable].isna().sum(),random_state=0)
    random_sample.index = df[df[variable].isna()].index
    df.loc[df[variable].isnull(),variable]=random_sample


# In[17]:


randomsimpleimputation(df,'Evaporation')
randomsimpleimputation(df,'Sunshine')
randomsimpleimputation(df,'Cloud9am')
randomsimpleimputation(df,'Cloud3pm')


# In[41]:


df.describe()


# In[37]:


df['MinTemp']=df['MinTemp'].fillna(df['MinTemp'].mean())
df['MaxTemp']=df['MaxTemp'].fillna(df['MaxTemp'].mean())
df['WindGustSpeed']=df['WindGustSpeed'].fillna(df['WindGustSpeed'].mean())
df['WindSpeed9am']=df['WindSpeed9am'].fillna(df['WindSpeed9am'].mean())
df['WindSpeed3pm']=df['WindSpeed3pm'].fillna(df['WindSpeed3pm'].mean())
df['Pressure9am']=df['Pressure9am'].fillna(df['Pressure9am'].mean())
df['Pressure3pm']=df['Pressure3pm'].fillna(df['Pressure3pm'].mean())
df['Temp9am']=df['Temp9am'].fillna(df['Temp9am'].mean())
df['Temp3pm']=df['Temp3pm'].fillna(df['Temp3pm'].mean())


# In[33]:


df['Rainfall']=df['Rainfall'].fillna(method='ffill')
df['Humidity9am']=df['Humidity9am'].fillna(method='ffill')
df['Humidity3pm']=df['Humidity3pm'].fillna(method='ffill')


# In[38]:


df['WindGustDir']=df['WindGustDir'].fillna(df['WindGustDir'].mode()[0])
df['WindDir9am']=df['WindDir9am'].fillna(df['WindDir9am'].mode()[0])
df['WindDir3pm']=df['WindDir3pm'].fillna(df['WindDir3pm'].mode()[0])
df['RainToday']=df['RainToday'].fillna(df['RainToday'].mode()[0])
df['RainTomorrow']=df['RainTomorrow'].fillna(df['RainTomorrow'].mode()[0])


# In[39]:


df.isna().sum()


# In[40]:


df


# In[21]:


for i in df.columns:
    print(df[i].value_counts())
    print('\n')


# In[66]:


for i in contineous_features:
    data=df.copy()
    sns.distplot(df[i])
    plt.xlabel(i)
    plt.ylabel('count')
    plt.title(i)
    plt.show()


# In[67]:


df.skew()


# In[ ]:





# In[45]:


sns.countplot(data=df,x='WindGustDir',hue='RainTomorrow')


# In[49]:


sns.countplot(data=df,x='Location',hue='RainTomorrow')
plt.xticks(rotation=45)


# In[51]:


sns.countplot(data=df,x='Location',hue='RainToday')
plt.xticks(rotation=45)


# In[50]:


sns.countplot(data=df,x='RainToday',hue='RainTomorrow')
plt.xticks(rotation=45)


# In[52]:


sns.scatterplot(data=df,x='WindGustDir',y='Rainfall')


# In[55]:


sns.scatterplot(data=df,x='Date',y='Rainfall')


# In[58]:


df.plot(x='Date', y='Rainfall', kind='line')


# In[53]:


sns.barplot(data=df,x='WindGustDir',y='Rainfall',hue='RainTomorrow')


# In[75]:


for i in contineous_features:
    data=df.copy()
    sns.boxplot(df[i])
    plt.xlabel(i)
    plt.ylabel('count')
    plt.title(i)
    plt.show()


# some outliers are present in the data set to remove those we are using IQR method

# In[81]:


IQR=df.MinTemp.quantile(0.75)-df.MinTemp.quantile(0.25)
lower_bridge=df.MinTemp.quantile(0.25)-(IQR*1.5)
upper_bridge=df.MinTemp.quantile(0.75)+(IQR*1.5)
print(lower_bridge, upper_bridge)


# In[82]:


df.loc[df['MinTemp']>=29.3,'MinTemp']=29.3
df.loc[df['MinTemp']<=-2.7,'MinTemp']=-2.7


# In[83]:


IQR=df.MaxTemp.quantile(0.75)-df.MaxTemp.quantile(0.25)
lower_fence=df.MaxTemp.quantile(0.75)-(1.5*IQR)
upper_fence=df.MaxTemp.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[85]:


df.loc[df['MaxTemp']>=32.35,'MaxTemp']=32.35
df.loc[df['MaxTemp']<=14.9,'MaxTemp']=14.9


# In[87]:


IQR=df.Rainfall.quantile(0.75)-df.Rainfall.quantile(0.25)
lower_fence=df.Rainfall.quantile(0.75)-(1.5*IQR)
upper_fence=df.Rainfall.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[88]:


df.loc[df['Rainfall']>=1.5,'Rainfall']=1.5
df.loc[df['Rainfall']<=-0.5,'Rainfall']=-0.5


# In[89]:


IQR=df.Evaporation.quantile(0.75)-df.Evaporation.quantile(0.25)
lower_fence=df.Evaporation.quantile(0.75)-(1.5*IQR)
upper_fence=df.Evaporation.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[90]:


df.loc[df['Evaporation']>=9.2,'Evaporation']=9.2
df.loc[df['Evaporation']<=-0.4,'Evaporation']=-0.4


# In[ ]:


IQR=df.Rainfall.quantile(0.75)-df.Rainfall.quantile(0.25)
lower_fence=df.Rainfall.quantile(0.75)-(1.5*IQR)
upper_fence=df.Rainfall.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[ ]:


df.loc[df['Evaporation']>=9.2,'Evaporation']=9.2
df.loc[df['Evaporation']<=-0.4,'Evaporation']=-0.4


# In[91]:


IQR=df.WindGustSpeed.quantile(0.75)-df.WindGustSpeed.quantile(0.25)
lower_fence=df.WindGustSpeed.quantile(0.75)-(1.5*IQR)
upper_fence=df.WindGustSpeed.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[92]:


df.loc[df['WindGustSpeed']>=56.5,'WindGustSpeed']=56.5
df.loc[df['WindGustSpeed']<=22.5,'WindGustSpeed']=22.5


# In[93]:


IQR=df.WindSpeed9am.quantile(0.75)-df.WindSpeed9am.quantile(0.25)
lower_fence=df.WindSpeed9am.quantile(0.75)-(1.5*IQR)
upper_fence=df.WindSpeed9am.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[94]:


df.loc[df['WindSpeed9am']>=27.0,'WindSpeed9am']=27.0
df.loc[df['WindSpeed9am']<=-1.0,'WindSpeed9am']=-1.0


# In[95]:


IQR=df.WindSpeed3pm.quantile(0.75)-df.WindSpeed3pm.quantile(0.25)
lower_fence=df.WindSpeed3pm.quantile(0.75)-(1.5*IQR)
upper_fence=df.WindSpeed3pm.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[96]:


df.loc[df['WindSpeed3pm']>=30.5,'WindSpeed3pm']=30.5
df.loc[df['WindSpeed3pm']<=4.5,'WindSpeed3pm']=4.5


# In[97]:


IQR=df.Humidity9am.quantile(0.75)-df.Humidity9am.quantile(0.25)
lower_fence=df.Humidity9am.quantile(0.75)-(1.5*IQR)
upper_fence=df.Humidity9am.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[98]:


df.loc[df['Humidity9am']>=92.0,'Humidity9am']=92.0
df.loc[df['Humidity9am']<=44.0,'Humidity9am']=44.0


# In[99]:


IQR=df.Humidity3pm.quantile(0.75)-df.Humidity3pm.quantile(0.25)
lower_fence=df.Humidity3pm.quantile(0.75)-(1.5*IQR)
upper_fence=df.Humidity3pm.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[100]:


df.loc[df['Humidity3pm']>=75.0,'Humidity3pm']=75.0
df.loc[df['Humidity3pm']<=27.0,'Humidity3pm']=27.0


# In[101]:


IQR=df.Pressure9am.quantile(0.75)-df.Pressure9am.quantile(0.25)
lower_fence=df.Pressure9am.quantile(0.75)-(1.5*IQR)
upper_fence=df.Pressure9am.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[102]:


df.loc[df['Pressure9am']>=1024.95,'Pressure9am']=1024.95
df.loc[df['Pressure9am']<=1010.35,'Pressure9am']=1010.35


# In[103]:


IQR=df.Pressure3pm.quantile(0.75)-df.Pressure3pm.quantile(0.25)
lower_fence=df.Pressure3pm.quantile(0.75)-(1.5*IQR)
upper_fence=df.Pressure3pm.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[104]:


df.loc[df['Pressure3pm']>=1022.5,'Pressure3pm']=1022.5
df.loc[df['Pressure3pm']<=1007.7,'Pressure3pm']=1007.7


# In[105]:


IQR=df.Temp9am.quantile(0.75)-df.Temp9am.quantile(0.25)
lower_fence=df.Temp9am.quantile(0.75)-(1.5*IQR)
upper_fence=df.Temp9am.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[106]:


df.loc[df['Temp9am']>=26,'Temp9am']=26
df.loc[df['Temp9am']<=9.75,'Temp9am']=9.75


# In[107]:


IQR=df.Temp3pm.quantile(0.75)-df.Temp3pm.quantile(0.25)
lower_fence=df.Temp3pm.quantile(0.75)-(1.5*IQR)
upper_fence=df.Temp3pm.quantile(0.25)+(1.5*IQR)
print(lower_fence,upper_fence)


# In[108]:


df.loc[df['Temp3pm']>=30.4,'Temp3pm']=30.4
df.loc[df['Temp3pm']<=14.0,'Temp3pm']=14.0


# In[109]:


for i in contineous_features:
    data=df.copy()
    sns.boxplot(df[i])
    plt.xlabel(i)
    plt.ylabel('count')
    plt.title(i)
    plt.show()


# In[111]:


def qq_plots(df, variable):
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[variable].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[variable], dist="norm", plot=plt)
    plt.show()


# In[115]:


from scipy import stats
for feature in contineous_features:
    print(feature)
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    df[feature].hist()
    plt.subplot(1, 2, 2)
    stats.probplot(df[feature], dist="norm", plot=plt)
    plt.show()


# In[68]:


from sklearn.preprocessing import LabelEncoder
Le=LabelEncoder()


# In[70]:


for i in categorical_features:
    df[i]=Le.fit_transform(df[i])


# In[71]:


df


# In[118]:


from imblearn.over_sampling import SMOTE
smt=SMOTE()
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score


# In[117]:


x=df.drop(columns='RainTomorrow')
y=df['RainTomorrow']


# In[119]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[120]:


x_res,y_res=smt.fit_resample(X_train,y_train)


# In[125]:


print(x_res.shape)
print(y_res.shape)


# In[126]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


# In[123]:


model=[
    DecisionTreeClassifier(),
    LogisticRegression(),
    AdaBoostClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier()
    
]


# In[132]:


for i in model:
    i.fit(x_res,y_res)
    y_pred=i.predict(X_test)
    print(i)
    print(accuracy_score(y_test,y_pred))
    print(confusion_matrix(y_test,y_pred))
    
    scores = cross_val_score(i, x_res, y_res, cv=5)
    print(scores.mean())
    print('diff bet score and accuracy',accuracy_score(y_test,y_pred)-(scores.mean()))
    print('\n')


# Random Forest Classifier gives the best accuracy of 87%

# In[141]:


params={
    'n_estimators':[100,20,40,60,80],
    'criterion':['gini','entropy','log_loss'],
    'max_depth':[None,100,50]
}


# In[142]:


grid_search=GridSearchCV(RandomForestClassifier(),param_grid=params,cv=10)


# In[143]:


grid_search.fit(x_res,y_res)


# In[144]:


grid_search.best_params_


# In[148]:


rf=RandomForestClassifier(criterion='log_loss',max_depth=50,max_features='log2',n_estimators=100)


# In[149]:


rf.fit(x_res,y_res)


# In[150]:


rf.fit(x,y)

