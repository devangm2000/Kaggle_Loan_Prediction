#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dataset=pd.read_csv("train_loan.csv")
data1=pd.read_csv("test_loan.csv")


# In[3]:


dataset


# In[4]:


dataset.isnull().sum()


# In[5]:


#replacing null values
dataset['Gender'].fillna(value='Male',inplace=True)
dataset['Married'].fillna(value='No',inplace=True)
dataset['Dependents'].fillna(value=1,inplace=True)
dataset['Self_Employed'].fillna(value='Yes',inplace=True)
dataset['LoanAmount'].fillna(dataset['LoanAmount'].mean(),inplace=True)
dataset['Loan_Amount_Term'].fillna(value=180,inplace=True)
dataset['Credit_History'].fillna(value=0,inplace=True)


# In[6]:


#encoding categorical data
dataset['Gender'] = dataset['Gender'].replace(['Female','Male'],[0,1])
dataset['Married'] = dataset['Married'].replace(['No','Yes'],[0,1])
dataset['Education'] = dataset['Education'].replace(['Not Graduate','Graduate'],[0,1])
dataset['Self_Employed'] = dataset['Self_Employed'].replace(['No','Yes'],[0,1])
dataset['Property_Area'] = dataset['Property_Area'].replace(['Urban','Semiurban','Rural'],[0,1,2])
dataset['Loan_Status'] = dataset['Loan_Status'].replace(['N','Y'],[0,1])
dataset['Dependents'] = dataset['Dependents'].replace(["3+"],[4])


# In[7]:


dataset


# In[8]:


X=dataset.drop(columns=['Loan_ID','Loan_Status'])
X


# In[9]:


y=dataset[['Loan_Status']]
y=y.iloc[:,:].values


# In[10]:


#selecting best features
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
fs = SelectKBest(score_func=chi2, k='all')
fs.fit(X, y)
X_fs = fs.transform(X)
X_fs = fs.transform(X)
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))


# In[11]:


X.head()


# In[12]:


X=X.drop(columns=['Gender','Dependents','Education','Self_Employed','Loan_Amount_Term','Property_Area'])


# In[13]:


X=X.iloc[:,:].values


# In[14]:


#splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,random_state=42)


# In[20]:


#model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train.ravel())
#predicting
y_pred = classifier.predict(X_test)


# In[21]:


#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[22]:


#Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
acc


# In[23]:


tp=cm[0][0]
fp=cm[0][1]
tn=cm[1][1]
fn=cm[1][0]
precision=tp/(tp+fp)
precision


# In[24]:


recall=tp/(tp+fn)
recall


# In[25]:


#F1 = 2 * (precision * recall) / (precision + recall)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, zero_division=1)


# In[26]:


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


# In[27]:


new_input=np.array([married,income,coincome,loanamount,credithistory]).reshape(1,-1)
new_output = classifier.predict(new_input)
if new_output==0:
    print("Loan not approved")
else:
    print("Loan Approved")


# In[ ]:


#visualisation to be done

