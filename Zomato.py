#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[26]:


df = pd.read_csv('zomato (5).csv', encoding='latin1')


# In[27]:


pd.set_option('display.max_columns',100,'display.max_rows',200)


# In[28]:


df_country = pd.read_excel('Country-Code (1).xlsx')


# In[29]:


df=pd.merge(df,df_country,on= 'Country Code', how= 'left')


# In[30]:


df.columns


# In[31]:


df.dtypes


# In[32]:


for i in df.columns:
    print(df[i].value_counts())
    print('\n')


# In[33]:


df


# In[13]:


miss_val=df['Cuisines'].isnull()
sm=df[miss_val]
sm


# In[15]:


df['City'].value_counts()


# Null columns are present in cuisines column 
# we can drop those columns 

# In[34]:


df=df.dropna()


# In[35]:


df.shape


# In[36]:


df.drop_duplicates()


# Upto here i handled the null values and mergeing the files
# now working on Analysis

# In[44]:


from collections import Counter
all_cuisines = df['Cuisines'].str.split(', ').explode()

cuisine_counts = Counter(all_cuisines)

cuisine_counts_df = pd.DataFrame.from_dict(cuisine_counts, orient='index', columns=['Count']).reset_index()
cuisine_counts_df = cuisine_counts_df.rename(columns={'index': 'Cuisine'})

print(cuisine_counts_df.sort_values(by='Count', ascending=False))


# In[58]:


df['n_indian_cuisine'] = df.Cuisines.mask(df.Cuisines.str.contains('North Indian'),1)


# In[59]:


df['n_indian_cuisine'] = df['n_indian_cuisine'].mask(~df.Cuisines.str.contains('North Indian'),0)


# In[60]:


df['s_indian_cusine']=df.Cuisines.mask(df.Cuisines.str.contains('South Indian'),1)
df['s_indian_cusine'] = df['s_indian_cusine'].mask(~df.Cuisines.str.contains('South Indian'),0)


# In[61]:


df['Chinese']=df.Cuisines.mask(df.Cuisines.str.contains('Chinese'),1)
df['Chinese'] = df['Chinese'].mask(~df.Cuisines.str.contains('Chinese'),0)


# In[48]:


df['non_indian_cuisine']=df.Cuisines.mask(df.Cuisines.str.contains('American|Arabian|Argentine|Asian|AsianFusion|Australian|Belgian|Brazilian|British|Burmese|Cajun|Canadian|Cantonese|Caribbean|Chinese|Cuban|Danish|DimSum|Dutch|Egyptian|European|Filipino|French|German|Greek|Hawaiian|Indian|Indonesian|Iranian|Irish|Israeli|Italian|Japanese|Kebab|Korean|Lbanese|Malay|Malaysian|Mediterranean|Mexican|Middle_Eastern|Moroccan|Nepalese|New_American|Pakistani|Parsi|Peranakan|Peruvian|Portuguese|Pub_Food|Raw Meats|Ramen|Scottish|Singaporean|South African|South American|Spanish|Sri Lankan|Sunda|Swedish|Swiss|Taiwanese|Tapas|Tex-Mex|Turkish|Turkish Pizza|Vietnamese'),1)
df['non_indian_cuisine'] = df['non_indian_cuisine'].mask(~df.Cuisines.str.contains('American|Arabian|Argentine|Asian|AsianFusion|Australian|Belgian|Brazilian|British|Burmese|Cajun|Canadian|Cantonese|Caribbean|Chinese|Cuban|Danish|DimSum|Dutch|Egyptian|European|Filipino|French|German|Greek|Hawaiian|Indian|Indonesian|Iranian|Irish|Israeli|Italian|Japanese|Kebab|Korean|Lbanese|Malay|Malaysian|Mediterranean|Mexican|Middle_Eastern|Moroccan|Nepalese|New_American|Pakistani|Parsi|Peranakan|Peruvian|Portuguese|Pub_Food|Raw Meats|Ramen|Scottish|Singaporean|South African|South American|Spanish|Sri Lankan|Sunda|Swedish|Swiss|Taiwanese|Tapas|Tex-Mex|Turkish|Turkish Pizza|Vietnamese'),0)


# In[52]:


df['regional_cuisine'] = df.Cuisines.mask(df.Cuisines.str.contains('Mithai|Hyderabadi|Kerala|Rajasthani|Kashmiri|Goan|Bengali|Lucknowi|Gujarati|Chettinad|Maharashtrian|Assamese|Mangalorean|Andhra|Oriya|Awadhi'),1)
df['regional_cuisine'] = df['regional_cuisine'].mask(~df.Cuisines.str.contains('Mithai|Hyderabadi|Kerala|Rajasthani|Kashmiri|Goan|Bengali|Lucknowi|Gujarati|Chettinad|Maharashtrian|Assamese|Mangalorean|Andhra|Oriya|Awadhi'),0)


# In[53]:


df['bakery_desserts_cafe_cuisine'] = df.Cuisines.mask(df.Cuisines.str.contains('Desserts|Ice Cream|Bakery|Bubble Tea|Tea|Coffee And Tea|Restaurant Cafe'),1)
df['bakery_desserts_cafe_cuisine'] = df['bakery_desserts_cafe_cuisine'].mask(~df.Cuisines.str.contains('Desserts|Ice Cream|Bakery|Bubble Tea|Tea|Coffee And Tea|Restaurant Cafe'),0)


# In[55]:


df['Meat']=df.Cuisines.mask(df.Cuisines.str.contains('Steak|Raw Meats|Kebab|Turkish Pizza|Barbecue|Bbq|Charcoal Grill|Grill'),1)
df['Meat']=df['Meat'].mask(~df.Cuisines.str.contains('Steak|Raw Meats|Kebab|Turkish Pizza|Barbecue|Bbq|Charcoal Grill|Grill'),0)


# In[56]:


df['fast_food_cuisine'] = df.Cuisines.mask(df.Cuisines.str.contains('Pizza|Burger|Fast Food|Street Food|Finger Food|Rolls|Sandwiches|Warps|Momo|Fish And Chips'),1)
df['fast_food_cuisine'] = df['fast_food_cuisine'].mask(~df.Cuisines.str.contains('Pizza|Burger|Fast Food|Street Food|Finger Food|Rolls|Sandwiches|BBQ|Warps|Momo|Fish And Chips'),0)


# In[70]:


non_indian_cuisine = ['American', 'Arabian', 'Argentine', 'Asian', 'AsianFusion', 'Australian', 'Belgian', 'Brazilian', 
                      'British', 'Burmese', 'Cajun', 'Canadian', 'Cantonese', 'Caribbean', 'Chinese', 'Cuban', 'Danish', 
                      'DimSum', 'Dutch', 'Egyptian', 'European', 'Filipino', 'French', 'German', 'Greek', 'Hawaiian', 
                      'Indonesian', 'Iranian', 'Irish', 'Israeli', 'Italian', 'Japanese', 'Kebab', 'Korean', 'Lebanese', 
                      'Malay', 'Malaysian', 'Mediterranean', 'Mexican', 'Middle Eastern', 'Moroccan', 'Nepalese', 
                      'New American', 'Pakistani', 'Parsi', 'Peranakan', 'Peruvian', 'Portuguese', 'Pub Food', 
                      'Raw Meats', 'Ramen', 'Scottish', 'Singaporean', 'South African', 'South American', 'Spanish', 
                      'Sri Lankan', 'Sunda', 'Swedish', 'Swiss', 'Taiwanese', 'Tapas', 'Tex-Mex', 'Turkish', 
                      'Turkish Pizza', 'Vietnamese']
regional_cuisine = ['Kashmiri', 'Rajasthani', 'Bengali', 'Hyderabadi', 'Kerala', 'Mangalorean', 'Assamese', 'Gujarati', 'Lucknowi', 'Maharashtrian']
bakery_desserts_cafe_cuisine = ['Bakery', 'Desserts', 'Cafe']
meat_cuisine = ['Steak', 'Barbecue', 'Charcoal Grill', 'Raw Meats']
fast_food_cuisine = ['Fast Food', 'Burger', 'Pizza']
n_indian_cuisine = ['North Indian']
s_indian_cuisine = ['South Indian']
chinese_cuisine = ['Chinese']



# In[75]:


existing_cuisines = non_indian_cuisine + regional_cuisine + bakery_desserts_cafe_cuisine + meat_cuisine + fast_food_cuisine + n_indian_cuisine + s_indian_cuisine + chinese_cuisine
def mark_other_cuisine(cuisine):
    for existing_cuisine in existing_cuisines:
        if existing_cuisine.lower() in cuisine.lower():
            return 0
    return 1
df['Others'] = df['Cuisines'].apply(mark_other_cuisine)
other_cuisines = df[df['Others'] == 1]



# In[77]:


#now we can drop the Cusines column
df.drop(columns='Cuisines',inplace=True)


# In[25]:


df4 = df.groupby(['Aggregate rating','Rating color', 'Rating text']).size().reset_index().rename(columns={0:'Rating Count'})
df4


# In[21]:


df1 = df.groupby(['Aggregate rating', 'Rating color', 'Rating text']).size().reset_index(name='Rating Count')


# In[22]:


df1


# 2000 orders are not rated  

# In[26]:


df2=df.groupby(['Aggregate rating','Country']).size().reset_index().rename(columns={0:'Country'})
df2


# most of the unrated orders are from india 

# In[27]:


df3=df.groupby(['Rating color','Country']).size().reset_index().rename(columns={0:'Country'})
df3


# In[17]:


sns.barplot(data=df,x='Country',y='Price range')
plt.xticks(rotation=45)


# price range is low in india and high in Singapur and Qatar

# In[20]:


sns.barplot(data=df,x='Country',y='Average Cost for two')
plt.xticks(rotation=45)


# In[19]:


country_val=df['Country'].value_counts()


# In[21]:


Country_name=df['Country'].value_counts().index


# In[23]:


plt.pie(country_val[:4],labels=Country_name[:4],autopct='%1.2f%%')


# from the above graph we can say that around 94% of the restaurents are froom india and 5% from US

# In[24]:


sns.countplot(data=df,x='Has Online delivery')


# In[25]:


sns.countplot(data=df,x='Is delivering now')


# In[26]:


sns.countplot(data=df,x='Switch to order menu')


# we can drop the Switch to order menu column as it doesnot provide any information to predict  target variable

# In[ ]:





# In[29]:


sns.scatterplot(data=df,x='Average Cost for two',y='Aggregate rating')


# In[30]:


df['Cuisines'].value_counts()


# In[79]:


print(df[['Address', 'Locality', 'Locality Verbose']].describe())


# In[80]:


print(df['Address'].nunique())
print(df['Locality'].nunique())
print(df['Locality Verbose'].nunique())

# Value counts
print(df['Address'].value_counts().head(10))
print(df['Locality'].value_counts().head(10))
print(df['Locality Verbose'].value_counts().head(10))



# In[81]:


locality_rating = df.groupby('Locality')['Aggregate rating'].mean().sort_values(ascending=False)
print(locality_rating)


# In[84]:


df.drop(columns=['Country Code','Address','Locality Verbose','Switch to order menu','Is delivering now','Restaurant Name'],inplace=True)


# In[85]:


df


# In[106]:


city_counts = df['City'].value_counts()
popular_cities = city_counts[city_counts > 200].index
avg_cities = city_counts[(city_counts >= 20) & (city_counts <= 200)].index
less_popular_cities = city_counts[city_counts < 20].index


# In[115]:


cities_str = ', '.join(less_popular_cities)
cities_str.replace(',',"|")


# In[ ]:


Ghaziabad|Ahmedabad|Guwahati|Amritsar|Bhubaneshwar|Lucknow|Dubai|Kochi|Indore|Jaipur|Kanpur|Ludhiana|Kolkata|Agra|Mangalore|Mumbai|Nagpur|Pensacola|Pocatello|Abu Dhabi|Goa|Nashik|Singapore|Sioux City|Tampa Bay|Sharjah|Dehradun|Valdosta|Coimbatore|Chennai|Bhopal|Bangalore|Aurangabad|Waterloo|Mysore|Patna|Cedar Rapids|Iowa City|Wellington City|Birmingham|Augusta|Athens|Edinburgh|Sí£o Paulo|Rio de Janeiro|Brasí_lia|London|Manchester|Doha|Cape Town|Pretoria|Colombo|Ankara|Boise|Allahabad|Auckland|Vadodara|Puducherry|Pune|Vizag|Columbus|Varanasi|Dalton|Ranchi|Surat|Dubuque|Davenport|Des Moines'


# In[103]:


df['popular_cities']=df.City.mask(df.City.str.contains('New Delhi|Gurgaon|Noida|Faridabad'),1)
df['popular_cities'] = df['popular_cities'].mask(~df.City.str.contains('New Delhi|Gurgaon|Noida|Faridabad'),0)


# In[111]:


df['avg_cities']=df.City.mask(df.City.str.contains('Ghaziabad|Ahmedabad|Guwahati|Amritsar|Bhubaneshwar|Lucknow|Dubai|Kochi|Indore|Jaipur|Kanpur|Ludhiana|Kolkata|Agra|Mangalore|Mumbai|Nagpur|Pensacola|Pocatello|Abu Dhabi|Goa|Nashik|Singapore|Sioux City|Tampa Bay|Sharjah|Dehradun|Valdosta|Coimbatore|Chennai|Bhopal|Bangalore|Aurangabad|Waterloo|Mysore|Patna|Cedar Rapids|Iowa City|Wellington City|Birmingham|Augusta|Athens|Edinburgh|Sí£o Paulo|Rio de Janeiro|Brasí_lia|London|Manchester|Doha|Cape Town|Pretoria|Colombo|Ankara|Boise|Allahabad|Auckland|Vadodara|Puducherry|Pune|Vizag|Columbus|Varanasi|Dalton|Ranchi|Surat|Dubuque|Davenport|Des Moines'),1)
df['avg_cities'] = df['avg_cities'].mask(~df.City.str.contains('Ghaziabad|Ahmedabad|Guwahati|Amritsar|Bhubaneshwar|Lucknow|Dubai|Kochi|Indore|Jaipur|Kanpur|Ludhiana|Kolkata|Agra|Mangalore|Mumbai|Nagpur|Pensacola|Pocatello|Abu Dhabi|Goa|Nashik|Singapore|Sioux City|Tampa Bay|Sharjah|Dehradun|Valdosta|Coimbatore|Chennai|Bhopal|Bangalore|Aurangabad|Waterloo|Mysore|Patna|Cedar Rapids|Iowa City|Wellington City|Birmingham|Augusta|Athens|Edinburgh|Sí£o Paulo|Rio de Janeiro|Brasí_lia|London|Manchester|Doha|Cape Town|Pretoria|Colombo|Ankara|Boise|Allahabad|Auckland|Vadodara|Puducherry|Pune|Vizag|Columbus|Varanasi|Dalton|Ranchi|Surat|Dubuque|Davenport|Des Moines'),0)


# In[116]:


df['less_popular_cities']=df.City.mask(df.City.str.contains('Macon| Rest of Hawaii| Gainesville| Orlando| Savannah| Hyderabad| Chandigarh| Albany| Jakarta| ÛÁstanbul| Sandton| Johannesburg| Mandaluyong City| Taguig City| Pasay City| Pasig City| Makati City| Santa Rosa| Tangerang| Hepburn Springs| Secunderabad| San Juan City| Inner City| Bogor| Bandung| East Ballina| Fernley| Dicky Beach| Consort| Flaxton| Cochrane| Balingup| Clatskanie| Chatham-Kent| Beechworth| Armidale| Tagaytay City| Quezon City| Randburg| Forrest| Yorkton| Winchester Bay| Huskisson| Weirton| Vineland Station| Victor Harbor| Vernonia| Trentham East| Tanunda| Princeton| Potrero| Phillip Island| Penola| Paynesville| Palm Cove| Ojo Caliente| Montville| Monroe| Middleton Beach| Mc Millan| Mayfield| Macedon| Mohali| Lorn| Lincoln| Lakeview| Lakes Entrance| Inverloch| Panchkula'),1)
df['less_popular_cities'] = df['less_popular_cities'].mask(~df.City.str.contains('Macon| Rest of Hawaii| Gainesville| Orlando| Savannah| Hyderabad| Chandigarh| Albany| Jakarta| ÛÁstanbul| Sandton| Johannesburg| Mandaluyong City| Taguig City| Pasay City| Pasig City| Makati City| Santa Rosa| Tangerang| Hepburn Springs| Secunderabad| San Juan City| Inner City| Bogor| Bandung| East Ballina| Fernley| Dicky Beach| Consort| Flaxton| Cochrane| Balingup| Clatskanie| Chatham-Kent| Beechworth| Armidale| Tagaytay City| Quezon City| Randburg| Forrest| Yorkton| Winchester Bay| Huskisson| Weirton| Vineland Station| Victor Harbor| Vernonia| Trentham East| Tanunda| Princeton| Potrero| Phillip Island| Penola| Paynesville| Palm Cove| Ojo Caliente| Montville| Monroe| Middleton Beach| Mc Millan| Mayfield| Macedon| Mohali| Lorn| Lincoln| Lakeview| Lakes Entrance| Inverloch| Panchkula'),0)


# In[118]:


df.drop(columns=['Count','count','City','Locality'],inplace=True)


# In[125]:


categorical_columns=[feature for feature in df.columns if df[feature].dtypes=='object']


# In[127]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[128]:


for i in categorical_columns:
    df[i]=le.fit_transform(df[i])


# In[129]:


df


# Multi class classification Problem

# In[131]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


# In[132]:


X=df.drop(columns='Price range')
y=df['Price range']


# In[133]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[134]:


model=[
    DecisionTreeClassifier(),
    ExtraTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    LogisticRegression()
]


# In[135]:


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


# In[142]:


params={
    'criterion':['gini','entropy'],
    'splitter':['best','random'],
    'max_depth':[None,60,10,40,50],  # No maximum depth
    'min_samples_split':[2,4,6,10],  # Minimum samples required to split an internal node
    'min_samples_leaf':[1,2,3,4],  # Minimum samples required to be at a leaf node
    'max_features':[None,3,4,6,10,5,15,20,30],  # Consider all features
    'max_leaf_nodes':[None,10,20,30,50,60],  # No maximum number of leaf nodes
    'min_impurity_decrease':[0.0,0.2,0.3,0.5,0.4]  # No minimum impurity decrease
}  


# In[143]:


l=RandomizedSearchCV(DecisionTreeClassifier(),params,cv=10,n_jobs=-2)


# In[144]:


l.fit(X_train,y_train)


# In[145]:


print(l.best_estimator_)
print(l.best_params_)
print(l.best_score_)


# In[157]:


lr=DecisionTreeClassifier()
lr.fit(X_train,y_train)
p=lr.predict(X_test)


# In[159]:


print(accuracy_score(y_test, p))
print(confusion_matrix(y_test, p))


# Regression Problem
# cost of 2 people

# In[161]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,RandomizedSearchCV
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import xgboost as xgb


# In[164]:


X=df.drop(columns='Average Cost for two')
y=df['Average Cost for two']


# In[165]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


# In[170]:


model=[
    DecisionTreeRegressor(),
    ExtraTreeRegressor(),
    RandomForestRegressor(),
    AdaBoostRegressor(),
    GradientBoostingRegressor(),
    Ridge(),
    Lasso(),
    LinearRegression()
]


# In[171]:


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
    


# In[173]:


gb=GradientBoostingRegressor()
gb.fit(X_train,y_train)

