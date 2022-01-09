import pandas as pd
from pandas import read_csv
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import SVR


import os

os.chdir("C:\\Users\\Giorgos\\Desktop\\ergasies_metaptxiakwn\\ergasia filippaki\\εργασια εξαμηνου")

data=pd.read_csv("Online Retail.csv")

########################### DATA PREPROCESSING #################################

pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns",None)
print(data.head())
print("==========================DATA INFO===========================")
print(data.info())
print("==========================DATA DESCRIPTION===========================")
print(data.describe())
print("==========================MISSING VALUES===========================")
print(data.isnull().sum())
data_new=data[(data.Quantity>0) & (data.UnitPrice>0)]
#Transform InvoiceDate to datetime
data_new['InvoiceDate'] = pd.to_datetime(data_new['InvoiceDate'])
print("========================DATA_NEW DESCRIPTION==========================")
print(data_new.describe())

#removing rows with missing Customer ID
data_new = data_new.dropna(subset=['CustomerID'])

#checking missing dates in our sales
datelist=pd.date_range(start='2010-12-01 08:26:00',end='2011-12-09 12:50:00')

#Creating columns for year and month when order is made
data_new['TotalPrice'] = data_new['Quantity']*data_new['UnitPrice']

data_new['Year'] = pd.DatetimeIndex(data_new['InvoiceDate']).year

data_new['Month'] = pd.DatetimeIndex(data_new['InvoiceDate']).month
data_new.info()
print(data_new.head())
#Getting knownledge about amount of invoices per month. The most active month is November
data_new.groupby(['Year', 'Month']).InvoiceNo.count().plot(kind='bar', title='Amount of invoices per month')
plt.show()
data_new.groupby(["Year","Month"]).CustomerID.count().plot(kind='bar',title="Amount of customers per month")
plt.show()

#Getting knowledge about Total revenue per month. The best one is November
#(it is expected, becouse November was the most active month for sales)
#px.bar(data_new[['InvoiceDate','TotalPrice']].set_index('InvoiceDate').resample('M').sum().reset_index(),x='InvoiceDate', y='TotalPrice', title = 'Total Revenue per month')
fig, (axis1) = plt.subplots(1 , figsize=(15, 4))
sns.barplot(x='Month', y='TotalPrice', data=data_new, ax=axis1)
#plt.show()
data_new = data_new [['Quantity', 'UnitPrice', 'TotalPrice', 'Year', 'Month','Country']]
print(data_new.head())
array=data_new.values
X = array[:,0:4]
Y = array[:,4]
'''
kfold = KFold(n_splits=10,random_state=7,shuffle=True)
model=LinearRegression()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Linear Regression neg mse:",results.mean())

num_folds=10
kfold = KFold(n_splits=10,random_state=7,shuffle=True)
model=Ridge()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
model=Ridge()
print("Ridge neg mse:",results.mean())

kfold = KFold(n_splits=10,random_state=7,shuffle=True)
model=Lasso()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Lasso neg mse:",results.mean())
'''
kfold = KFold(n_splits=2,random_state=7,shuffle=True)
model=SVR()
scoring = 'neg_mean_squared_error'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Support Vector Machines neg mse:",results.mean())




