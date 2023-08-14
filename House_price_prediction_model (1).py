#!/usr/bin/env python
# coding: utf-8

# In[218]:


#import all the important libraries required for developing the model.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from math import sqrt


# In[219]:


#reading the dataset with the help of pandas library.
kc = pd.read_csv("C:\\Users\Suyash Pandey\OneDrive\Desktop\FINAL_TF2_FILES\TF_2_Notebooks_and_Data\DATA\kc_house_data.csv")


# In[220]:


#displaying first five rows of the dataframe using the head command
kc.head(10)


# In[221]:


#Other information of the dataset
kc.info()


# In[222]:


#description of the dataset
kc.describe()


# In[223]:


#graph  between sqft_living15 and price of the house
x = kc['sqft_living15']
y = kc[['price']]
plt.scatter(x,y)
plt.xlabel('sqft_liv')
plt.ylabel('price of the house')
plt.title('graph between sqft living and price of the house')


# In[224]:


#taking all the numerical values in a single variable for plotting the correlation heatmap.
x_numerical_values = kc[['id','price','bedrooms','bathrooms','sqft_living','sqft_lot', 'floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']]


# In[225]:


#correlation heatmap of all the numerical values
f, ax = plt.subplots(figsize = (20,20))
sns.heatmap(x_numerical_values.corr(), annot = True)


# In[226]:


#total columns in the dataset
kc.columns


# In[227]:


#features selected as the predictors
selected_features = {'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
       'lat', 'long', 'sqft_living15', 'sqft_lot15'}


# In[228]:


#taking X as predictor variable and y has been already assigned as the target variable which is the price of the house
X = kc[selected_features]


# In[229]:


X


# In[230]:


y


# In[231]:


#displaying the shape of the predictor and target variable
print(X.shape,y.shape)


# In[232]:


#normalization using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
X_scaled = Scaler.fit_transform(X)


# In[233]:


#scaled predictor values
X_scaled


# In[234]:


#shape of scaled values
print(X_scaled.shape)


# In[235]:


#scaling of target values
y_scaled = Scaler.fit_transform(y)


# In[236]:


#scaled target values
y_scaled


# In[237]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.25)


# In[238]:


#shape of training and testing dataset
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[239]:


#fitting of training values into the model and calculation of training and testing accuracy of the model.
from sklearn.linear_model import LinearRegression
modellm = LinearRegression()
modellm.fit(X_train,y_train)
print("Training_Score:",modellm.score(X_train, y_train)*100)
print("Testing_score :",modellm.score(X_test, y_test)*100)


# In[240]:


#model predicting its own values from the testing dataset
y_predict = modellm.predict(X_test)


# In[241]:


#dislaying the predicting values
print(y_predict)


# In[242]:


#graph between y_test and y_predict.
plt.plot(y_test,y_predict, '^', color = 'r')
plt.xlabel('y_test')
plt.ylabel('y_predict')


# In[243]:


#graph between y_test_original and y_predict_original.
y_predict_original = Scaler.inverse_transform(y_predict)
y_test_original = Scaler.inverse_transform(y_test)
plt.plot(y_test_original,y_predict_original,'^',color = 'b')
plt.xlabel('model_predictions')
plt.ylabel('true_values')


# In[244]:


#calculating the value of n 
k = X_test.shape
n = len(X_test)
print('value of n:',n)


# In[245]:


#calculation of Root Mean Squared Error
RMSE = float(format(np.sqrt(mean_squared_error(y_test_original,y_predict_original)), '0.3f'))
print(RMSE)


# In[246]:


#calculation of Mean Squared Error
MSE = mean_squared_error(y_test_original,y_predict_original)
print(MSE)


# In[247]:


#calculation of Mean Absolute Error
MAE = mean_absolute_error(y_test_original,y_predict_original)
print(MAE)


# In[248]:


#calculation of R2 score
r2 = r2_score(y_test_original,y_predict_original)
print(r2)

