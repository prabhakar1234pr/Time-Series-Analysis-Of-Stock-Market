#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data


# In[2]:


pip install yfinance


# In[3]:


import yfinance as yf


# In[4]:


start = '2010-01-01'
end = '2019-12-31'

# Fetch stock data using yfinance
df = yf.download('AAPL', start=start, end=end)


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df=df.reset_index()
df.head()


# In[8]:


df=df.drop(['Date','Adj Close'],axis=1)
df.head()


# In[9]:


plt.plot(df.Close)


# In[10]:


(df)


# In[11]:


ma100=df.Close.rolling(100).mean()
ma100


# In[12]:


plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')


# In[13]:


ma200 = df.Close.rolling(200).mean()
print(ma200)


# In[14]:


plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')


# In[15]:


df.shape


# splitting data into training and testing

# In[17]:


data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
                          


# In[18]:


data_training.shape


# In[19]:


data_testing.shape


# In[20]:


data_training.head()


# In[21]:


data_testing.head()


# In[22]:


from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler(feature_range =(0,1))


# In[23]:


data_training_scaler = Scaler.fit_transform(data_training)
data_training_scaler


# In[24]:


x_train = []
y_train = []
for i in range(100,data_training_scaler.shape[0]):
    x_train.append(data_training_scaler[i-100:i])
    y_train.append(data_training_scaler[i,0])
x_train,y_train =np.array(x_train),np.array(y_train)


# ML model

# In[26]:


pip install keras


# In[27]:


pip install tensorflow


# In[28]:


from keras.layers import Dense,Dropout,LSTM
from keras.models import Sequential


# In[29]:


from keras.layers import Input
model = Sequential()
model.add(Input(shape=(x_train.shape[1], 1)))
model.add(LSTM(units = 50, activation = 'relu',return_sequences=True ))
model.add(Dropout(0.2))

model.add(LSTM(units = 60, activation = 'relu',return_sequences=True ))
model.add(Dropout(0.3))

model.add(LSTM(units = 80, activation = 'relu',return_sequences=True ))
model.add(Dropout(0.4))

model.add(LSTM(units = 120, activation = 'relu' ))
model.add(Dropout(0.5))

model.add(Dense(units = 1))


# In[30]:


model.summary()


# In[113]:


from keras.optimizers import Adam
optimizer = Adam(clipvalue=1.0)
model.compile(optimizer=optimizer, loss='mean_squared_error')
model.fit(x_train,y_train,epochs=100)


# In[118]:


model.save('my_model.keras')


# In[120]:


data_testing.head()


# In[122]:


last_100_days = data_training.tail(100)


# In[124]:


final_df = pd.concat([last_100_days, data_testing], ignore_index=True)


# In[126]:


final_df


# In[128]:


input_data =Scaler.fit_transform(final_df)
input_data


# In[130]:


input_data.shape


# In[132]:


x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])


# In[134]:


x_test,y_test = np.array(x_test),np.array(y_test)
print(x_test.shape)
print(y_test.shape)


# In[136]:


x_test.shape


# In[138]:


y_test


# Making Predictions 

# In[140]:


y_predicted = model.predict(x_test)


# In[142]:


y_predicted.shape


# In[144]:


y_test.shape


# In[146]:


y_predicted


# In[148]:


y_test


# In[150]:


Scaler.scale_


# In[152]:


Scale_factor = 1/Scaler.scale_[0]
y_predicted = y_predicted*Scale_factor
y_test = y_test * Scale_factor


# In[154]:


plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()


# In[107]:


print(np.isnan(x_test).sum())  # Should be 0 if there are no NaN values


# In[105]:


print(x_test.shape)


# In[ ]:




