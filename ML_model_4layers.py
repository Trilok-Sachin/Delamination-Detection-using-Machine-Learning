#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import sklearn
from xgboost import XGBRegressor


# In[2]:


df = pd.read_csv('Delam_data_diff_4layers_total.csv')
df.describe()


# In[3]:


plt.plot(df['Area'], df['F20'])


# In[4]:


df['Layer_no'] = df['Layer_no'].replace(to_replace=0.01, value=0)
df['Layer_no'] = df['Layer_no'].replace(to_replace=0.02, value=1)
df.describe()


# In[5]:


zero_data = df[df['F1']==0]
plt.hist(zero_data['Area'], bins=100)
plt.title('Zero Areas')


# In[6]:


nonzero_data = df[df['F1']!=0]
plt.hist(nonzero_data['Area'], bins=100)
plt.title('Non Zero Areas')
nonzero_data.describe()


# In[7]:


plt.plot(nonzero_data['Area'], nonzero_data['F20'])


# In[8]:


plt.plot(df['X'], df['F10'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


#df.hist(bins=100, figsize=(100, 80))


# In[10]:


FREQs = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20']
from pandas.plotting import scatter_matrix
#scatter_matrix(df[FREQs], figsize=(20,12))

corr_matrix = nonzero_data.corr()

sn.heatmap(corr_matrix, annot=False)


# In[11]:


for FREQ in FREQs:
    print('{}'.format(FREQ), df['F1'].corr(df[FREQ]))


# In[12]:


drop_attr = ['X1',"X2", 'Y1', 'Y2','Layer_no','Area','X','Y']
X = nonzero_data.drop(columns=['X1',"X2", 'Y1', 'Y2','Layer_no','Area','X','Y','F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20'], axis=1)
#X = nonzero_data.drop(columns=FREQs[5:20], axis=1)
print(X.head())
X = X.to_numpy()

y_area = nonzero_data['Area']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_prepared = scaler.fit_transform(X)
print(X_prepared.shape)


# In[12]:


#Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train_area, y_test_area = train_test_split(X_prepared, y_area, test_size=0.2)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train_area)


# In[13]:


def predictions(X, y, model):
    #some_data_prepared = scaler.fit_transform(some_data)
    predictions = model.predict(X)
    print('Model: ',model.__class__.__name__ )
    print('Predictions: ',predictions)
    print('Labels: ',list(y))
    return np.array(predictions)

print('Training Predict...')
predictions(X_train[:5], y_train_area[:5], lin_reg)
print('Testing Predict...')
predictions(X_test[:5], y_test_area[:5], lin_reg)


# In[14]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def error_calc(X, y, model):
    pred = model.predict(X)
    mse = mean_squared_error(y, pred)
    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mse)
    print('Model: ',model.__class__.__name__ )
    print('RMSE: ',rmse)
    print('MAE: ',mae)


def plot(Act_val, Pred):
    plt.plot(np.linspace(1, len(Act_val), len(Act_val)),Act_val, label='Actual Values')
    plt.plot(np.linspace(1, len(Pred), len(Pred)), Pred, label='Predictions')
    plt.legend()
    plt.show()
    
print('Training Plot')
plot(y_train_area[:100], predictions(X_train[:100], y_train_area[:100], lin_reg))
print('Testing Plot')
plot(y_test_area[:100], predictions(X_test[:100], y_test_area[:100], lin_reg))


# In[15]:


print('Train Error...')
error_calc(X_train, y_train_area, lin_reg)
print('Test Error...')
error_calc(X_test, y_test_area, lin_reg)


# In[16]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train_area)

print('Train Error...')
error_calc(X_train, y_train_area, forest_reg)
print('Test Error...')
error_calc(X_test, y_test_area, forest_reg)


# In[17]:


print('Train Predictions...')
predictions(X_train[:5], y_train_area[:5], forest_reg)
print('Test Predictions...')
predictions(X_test[:5], y_test_area[:5], forest_reg)


# In[18]:


print('Train Plot...')
plot(y_train_area[:200], predictions(X_train[:200], y_train_area[:200], forest_reg))
print('Test Plot...')
plot(y_test_area[:200], predictions(X_test[:200], y_test_area[:200], forest_reg))


# In[19]:


Locations_X, Locations_Y = nonzero_data['X'], nonzero_data['Y']

y_Locations = np.vstack((Locations_X, Locations_Y))
y_Locations = y_Locations.reshape(1935,2)
print(y_Locations.shape, X_prepared.shape)


# In[20]:


X_train, X_test, y_train_loc, y_test_loc = train_test_split(X_prepared, y_Locations, test_size=0.2)
print(X_test.shape, y_test_loc.shape)


# In[21]:


#Linear Regression
lin_reg_loc = LinearRegression()
lin_reg_loc.fit(X_train, y_train_loc)


# In[22]:


print('Testing on train set..')
predictions(X_train[:5], y_train_loc[:5], lin_reg_loc)[:,1]
print('Testing on test set..')
predictions(X_test[:5], y_test_loc[:5], lin_reg_loc)[:,1]


# In[23]:


print('Train Error...')
error_calc(X_train, y_train_loc, lin_reg_loc)
print('Test Error...')
error_calc(X_test, y_test_loc, lin_reg_loc)


# In[24]:


from pylab import *

subplot(2,1,1)
plt.plot(y_test_loc[:100, 0], label='Actual Values')
plt.plot(predictions(X_test[:100], y_test_loc[:100], lin_reg_loc)[:,0], label='Predictions')
plt.title('Location X')
plt.legend()

subplot(2,1,2)
plt.plot(y_test_loc[:100, 0], label='Actual Values')
plt.plot(predictions(X_test[:100], y_test_loc[:100], lin_reg_loc)[:,1], label='Predictions')
plt.title('Location Y')


# In[25]:


subplot(2,1,1)
plt.plot(y_train_loc[:100, 0], label='Actual Values')
plt.plot(predictions(X_train[:100], y_train_loc[:100], lin_reg_loc)[:,0], label='Predictions')
plt.title('Location X')
plt.legend()

subplot(2,1,2)
plt.plot(y_train_loc[:100, 0], label='Actual Values')
plt.plot(predictions(X_train[:100], y_train_loc[:100], lin_reg_loc)[:,1], label='Predictions')
plt.title('Location Y')


# In[49]:


#RandomForest Regressor

forest_reg_loc = RandomForestRegressor(
                    n_estimators=25, max_depth=10, 
                    max_features=3, criterion='mae')
forest_reg_loc.fit(X_train, y_train_loc)
#n_estimators=30, max_depth=12, max_features=3, criterion='mae'
print('')


# In[45]:


print('Train Error...')
error_calc(X_train, y_train_loc, forest_reg_loc)
print('Test Error...')
error_calc(X_test, y_test_loc, forest_reg_loc)


# In[50]:


from pylab import *

subplot(2,1,1)
plt.plot(y_train_loc[:100,0], label='Actual Values')
plt.plot(predictions(X_train[:100], y_train_loc[:100], forest_reg_loc)[:,0], label='Predictions')
plt.title('Location X')
plt.legend()

subplot(2,1,2)
plt.plot(y_train_loc[:100,1], label='Actual Values')
plt.plot(predictions(X_train[:100], y_train_loc[:100], forest_reg_loc)[:,1], label='Predictions')
plt.title('Location Y')
plt.legend()


# In[51]:


subplot(2,1,1)
plt.plot(y_test_loc[:100, 0], label='Actual Values')
plt.plot(predictions(X_test[:100], y_test_loc[:100], forest_reg_loc)[:,0], label='Predictions')
plt.title('Location X')
plt.legend()

subplot(2,1,2)
plt.plot(y_test_loc[:100, 0], label='Actual Values')
plt.plot(predictions(X_test[:100], y_test_loc[:100], forest_reg_loc)[:,1], label='Predictions')
plt.title('Location Y')


# In[68]:


#XGBoost Area
xgb_reg_area = XGBRegressor(max_depth=2, 
                            learning_rate=0.1,
                            )
xgb_reg_area.fit(X_train, y_train_area)

print('Train Error...')
error_calc(X_train, y_train_area, xgb_reg_area)
print('Test Error...')
error_calc(X_test, y_test_area, xgb_reg_area)

print('Train Predictions...')
predictions(X_train[:5], y_train_area[:5], xgb_reg_area)
print('Test Predictions...')
predictions(X_test[:5], y_test_area[:5], xgb_reg_area)


prediction_train = predictions(X_train[:100], y_train_area[:100], xgb_reg_area)
prediction_test = predictions(X_test[:100], y_test_area[:100], xgb_reg_area)

subplot(2,1,1)
plt.plot(np.linspace(1, 100, 100), y_train_area[:100], label = 'Actual Values')
plt.plot(np.linspace(1, 100, 100), prediction_train, label = 'Predictions')
plt.legend()
plt.title('Training Plot')

subplot(2,1,2)
plt.plot(np.linspace(1, 100, 100), y_test_area[:100], label = 'Actual Values')
plt.plot(np.linspace(1, 100, 100), prediction_test, label = 'Predictions')
plt.legend()
plt.title('Test Plot')
'''
print('Training Plot')
plot(y_train_area[:100], prediction_train)
print('Testing Plot')
plot(y_test_area[:100], prediction_test)
'''


# In[73]:


#XGBoost Location
from sklearn.multioutput import MultiOutputRegressor 

xgb_reg_loc = XGBRegressor(max_depth=4, 
                            learning_rate=0.1,
                            gamma=0.2)
xgb_reg_loc = MultiOutputRegressor(xgb_reg_loc)
xgb_reg_loc.fit(X_train, y_train_loc)

print('Train Error...')
error_calc(X_train, y_train_loc, xgb_reg_loc)
print('Test Error...')
error_calc(X_test, y_test_loc, xgb_reg_loc)


print('Train Predictions...')
predictions(X_train[:5], y_train_loc[:5], xgb_reg_loc)
print('Test Predictions...')
predictions(X_test[:5], y_test_loc[:5], xgb_reg_loc)


subplot(2,1,1)
plt.plot(y_train_loc[:100, 0], label='Actual Values')
plt.plot(predictions(X_train[:100], y_train_loc[:100], xgb_reg_loc)[:,0], label='Predictions')
plt.title('Location X')
plt.legend()

subplot(2,1,2)
plt.plot(y_train_loc[:100, 1], label='Actual Values')
plt.plot(predictions(X_train[:100], y_train_loc[:100], xgb_reg_loc)[:,1], label='Predictions')
plt.title('Location Y')


# In[74]:


subplot(2,1,1)
plt.plot(y_test_loc[:100, 0], label='Actual Values')
plt.plot(predictions(X_test[:100], y_test_loc[:100], xgb_reg_loc)[:,0], label='Predictions')
plt.title('Location X')
plt.legend()

subplot(2,1,2)
plt.plot(y_test_loc[:100, 0], label='Actual Values')
plt.plot(predictions(X_test[:100], y_test_loc[:100], xgb_reg_loc)[:,1], label='Predictions')
plt.title('Location Y')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




