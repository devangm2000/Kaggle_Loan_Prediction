#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


dataset=pd.read_csv("train_loan.csv")
data1=pd.read_csv("test_loan.csv")


# In[4]:


dataset


# In[5]:


dataset.isnull().sum()


# In[6]:


#replacing null values
dataset['Gender'].fillna(value='Male',inplace=True)
dataset['Married'].fillna(value='No',inplace=True)
dataset['Dependents'].fillna(value=1,inplace=True)
dataset['Self_Employed'].fillna(value='Yes',inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean(),inplace=True)
dataset['Loan_Amount_Term'].fillna(value=180,inplace=True)
dataset['Credit_History'].fillna(value=0,inplace=True)


# In[7]:


#encoding categorical data
dataset['Gender'] = dataset['Gender'].replace(['Female','Male'],[0,1])
dataset['Married'] = dataset['Married'].replace(['No','Yes'],[0,1])
dataset['Education'] = dataset['Education'].replace(['Not Graduate','Graduate'],[0,1])
dataset['Self_Employed'] = dataset['Self_Employed'].replace(['No','Yes'],[0,1])
dataset['Property_Area'] = dataset['Property_Area'].replace(['Urban','Semiurban','Rural'],[0,1,2])
dataset['Loan_Status'] = dataset['Loan_Status'].replace(['N','Y'],[0,1])
dataset['Dependents'] = dataset['Dependents'].replace(["3+"],[4])


# In[8]:


dataset


# In[9]:


X=dataset.drop(columns=['Loan_ID','Loan_Status'])
X


# In[10]:


y=dataset[['Loan_Status']]
y


# In[11]:


#selecting best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
fs = SelectKBest(score_func=chi2, k='all')
fs.fit(X, y)
X_fs = fs.transform(X)
X_fs = fs.transform(X)
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))


# In[12]:


X.head()


# In[13]:


X=X.drop(columns=['Gender','Dependents','Education','Self_Employed','Loan_Amount_Term','Property_Area'])


# In[14]:


X


# In[15]:


#splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=42)


# In[16]:


#model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train.values.ravel())
#predicting
y_pred = classifier.predict(X_test)


# In[17]:


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[18]:


print("Accuracy for training set:",classifier.score(X_train,y_train)*100,"%")
print("Accuracy for test set:",classifier.score(X_test,y_test)*100,"%")


# In[19]:


print("Please enter the following details:\n")
gender=int(input("Please enter 0/1 for female/male:\n"))
married=int(input("Please enter 0/1 for not maried/married:\n"))
dependents=int(input("Please enter total dependents:\n"))
education=int(input("Please enter 0/1 for non graduate/graduate:\n"))
self_employed=int(input("Please enter 0/1 for if you're not self-employed/self-employed:\n"))
income=int(input("Please enter applicant's income:\n"))
coincome=int(input("Please enter co-applicant's income:\n"))
loanamount=int(input("Please enter loan amount you wish to take:\n"))
term=int(input("Please enter total loan amount term in days:\n"))
credithistory=int(input("Please enter 0/1 if no credit history/credit history:\n"))
propertyarea=int(input("Please enter 0/1/2 if you live in urban/semiurban/rural area:\n"))


# In[21]:


new_input=np.array([married,income,coincome,loanamount,credithistory]).reshape(1,-1)
new_output = classifier.predict(new_input)
if new_output==0:
    print("Loan not approved")
else:
    print("Loan Approved")


# In[ ]:


#visualisation to be done

