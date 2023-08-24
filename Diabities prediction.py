#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# In[4]:


df=pd.read_csv(r"C:\Users\Akash\Downloads\archive (10).zip")
df


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df['Outcome'].value_counts()


# In[9]:


df.groupby('Outcome').mean()


# In[10]:


X=df.drop('Outcome',axis=1)
X


# In[11]:


Y=df['Outcome']
Y


# # Data standarization

# In[12]:


Scaler=StandardScaler()
standarized_data=Scaler.fit_transform(X)
standarized_data


# In[13]:


X=standarized_data
Y=df['Outcome']


# In[14]:


X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[15]:


X.shape,X_train.shape,x_test.shape


# In[16]:


from sklearn import svm
classifier=svm.SVC(kernel='linear')


# In[17]:


classifier.fit(X_train,Y_train)


# In[18]:


prediction_on_training_data=classifier.predict(X_train)
training_data_accuracy=accuracy_score(Y_train,prediction_on_training_data)
training_data_accuracy


# In[19]:


prediction_on_testing_data=classifier.predict(x_test)
testing_data_accuracy=accuracy_score(y_test,prediction_on_testing_data)
testing_data_accuracy


# # predictive system

# In[20]:


input_data=(0,137,40,35,168,43.1,2.288,33)


# In[21]:


input_data_as_numpy_array=np.asarray(input_data)


# In[22]:


input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)#predicting for one instance


# In[23]:


#standarize the input data
std_data=Scaler.transform(input_data_reshaped)
print(std_data)


# In[24]:


prediction = classifier.predict(std_data)
prediction


# In[25]:


if(prediction[0]==0):
    print('The preditin is not diabetic')
else:
    print('The person is diabetic')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




