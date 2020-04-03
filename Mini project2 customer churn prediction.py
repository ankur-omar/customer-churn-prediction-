#!/usr/bin/env python
# coding: utf-8

# In[2]:


#import necessary libraries
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sbn
sbn.set(style ="whitegrid")
get_ipython().system('pip install pandas_profiling')


# In[3]:


import pandas_profiling
pandas_profiling.ProfileReport(pd.read_csv(r"C:\Users\me\Desktop\custumerchurn.csv"))


# In[4]:


#import customer churn data set
df =pd.read_csv(r"C:\Users\me\Desktop\custumerchurn.csv")
#view top five data entries
df.head()


# In[7]:


df.describe(include ="all")


# In[4]:


pd.set_option('display.width',3000)
pd.set_option('display.max_column',20)
pd.set_option('precision',2)
df.describe(include ='all')


# In[5]:


df.dtypes


# In[6]:


pd.isnull(df).sum()


# In[7]:


#converting churn value no and yes in 1 and 0
df.loc[df.Churn=='No','Churn'] = 0 
df.loc[df.Churn=='Yes','Churn'] = 1


# In[8]:


#convert no internet services to 'NO'
columns =['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for i in columns:
    df[i] =df[i].replace({'No internet service':'No'})

    


# In[9]:


#replace all the spaces with null values
df['TotalCharges'] =df['TotalCharges'].replace(" ",np.nan)
#drop null values of total charges feature
df= df[df["TotalCharges"].notnull()]
#
df = df.reset_index()[df.columns]
#convert total charges column value to float data type
df['TotalCharges'] =df['TotalCharges'].astype(float)


# In[10]:


df['Churn'].value_counts().values


# In[11]:


#visualize  total customer churn
sizes = df['Churn'].value_counts(sort = True)
colors = ["orange","green"]
labels =['No',"Yes"]
plt.pie(sizes, labels =labels,colors=colors, 
        autopct='%1.1f%%', shadow=True, startangle=400,)
plt.title('Percentage of Churn in Dataset')
plt.show()


# In[12]:


#Data visulization part
#churn rate visulisation by gender
sbn.barplot(x ='gender',y ='Churn',data =df)


# In[13]:


#churn rate by tech support
sbn.barplot(x= 'TechSupport',y ='Churn',data =df)


# In[14]:


# visulization of churn rate  by internet services
sbn.barplot(x ='InternetService',y ='Churn',data =df)


# In[15]:


## visulization of churn rate  by payment method
sbn.barplot(x ='PaymentMethod',y ='Churn',data =df)


# In[16]:


# visulization of churn rate  by contract duration
sbn.barplot(x ='Contract',y ='Churn',data =df)


# 
# sbn.barplot(x ='tenure',y ='Churn',data =df)

# In[17]:


#perform onehot encoding by the method get_dummies()
df =pd.get_dummies(df,columns =['Contract','Dependents','DeviceProtection','gender',
                                'InternetService','MultipleLines','OnlineBackup','OnlineSecurity','PaperlessBilling',
                                'Partner','PaymentMethod','PhoneService',
                                'SeniorCitizen','StreamingMovies','StreamingTV','TechSupport'
    
],drop_first =True)


# In[18]:


#perform feature scaling
from sklearn.preprocessing import StandardScaler
stander_Scaler =StandardScaler()
columns_feature_scaling =['tenure','MonthlyCharges','TotalCharges']
df[columns_feature_scaling] =stander_Scaler.fit_transform(df[columns_feature_scaling])


# In[19]:


df.head()


# In[20]:


y =df['Churn']
X =df.drop(['Churn','customerID'],axis =1)


# In[21]:


#number of columns increased due to get_dummies()
df.columns


# In[22]:


#split the data in trainnig set ang testing set 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(X,y,test_size =0.30,random_state =0)


# In[23]:


# here now we start to implement algorithms

#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
l_model =LogisticRegression(random_state =50)
l_model.fit(x_train,y_train)
pred =l_model.predict(x_test)
model_accuracy =round(metrics.accuracy_score(y_test,pred)*100,2)
print("THE accuracy of logistic regression model is",model_accuracy)


# In[24]:


#K-NEAREST NEIGHBORS CLASSIFIERS
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
k_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
k_model.fit(x_train, y_train) 
k_pred = k_model.predict(x_test)
knn_accuracy = round(metrics.accuracy_score(y_test, k_pred) * 100, 2)
print("THE accuracy of k-nearest neighbors  model is",knn_accuracy)


# In[25]:


#RANDOM FOREST CLASSIFIERS
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
r_model = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
r_model.fit(x_train, y_train) 
r_pred = r_model.predict(x_test)
r_accuracy = round(metrics.accuracy_score(y_test, r_pred) * 100, 2)
print("THE accuracy of random forest  model is",r_accuracy)


# In[26]:


#DECISION TREE CLASSIFIERS
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
d_model = DecisionTreeClassifier(criterion = "gini", random_state = 50)
d_model.fit(x_train, y_train) 
d_pred = d_model.predict(x_test)
d_accuracy = round(metrics.accuracy_score(y_test, d_pred) * 100, 2)
print("THE accuracy of decision tree model is",d_accuracy)


# In[27]:


#Support vector machine
from sklearn.svm import SVC
from sklearn import metrics
s_model = SVC(kernel='linear', random_state=50, probability=True)
s_model.fit(x_train,y_train)
s_pred = s_model.predict(x_test)
s_accuracy = round(metrics.accuracy_score(y_test, s_pred) * 100, 2)
print("THE accuracy of support vector machine  model is",s_accuracy)


# In[28]:


#now we comparison aur model accuracy
import pandas as pd
model_comp ={'Model':['LOGISTIC REGRESSION','K-NEAREST NEIGHBORS CLASSIFIERS','RANDOM FOREST CLASSIFIERS',
                      'DECISION TREE CLASSIFIER'
                            ,'SUPPORT VECTOR MACHINE'  ],'Score':[model_accuracy,knn_accuracy,r_accuracy,d_accuracy,s_accuracy]}
df1 =pd.DataFrame(model_comp)
Model_Compa = df1.sort_values(by='Score', ascending=False)
Model_Compa = Model_Compa.set_index('Model')
Model_Compa.reset_index()


# In[29]:


from sklearn.metrics import confusion_matrix
conf_mat_model = confusion_matrix(y_test,pred)
conf_mat_model


# In[30]:


# Predict the probability of Churn of each customer
df['Probability of Churn'] = s_model.predict_proba(df[x_test.columns])[:,1]


# In[31]:


# Create a Dataframe showcasing probability of Churn of each customer
df[['customerID','Probability of Churn']].head(10)


# In[ ]:





# In[ ]:




