import numpy as np # linear algebra
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import datetime as datetime
import plotly.graph_objs as go
from plotly.offline import iplot
from datetime import datetime,date
#from mlxtend.frequent_patterns import apriori
#from mlxtend.frequent_patterns import association_rules
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
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
data_new = data_new [['Quantity', 'UnitPrice', 'TotalPrice', 'Year', 'Month', 'Country']]
print(data_new.head())
'''
array = data_new.values
X = array[:,0:5]
Y = array[:,5]
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

# prepare models
models = []
models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#%test split
n_splits = 10
test_size = 0.33
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print("K-Nearest Neighbor")
print("Percentage split 67%")
print("Accuracy:", round(results.mean(),8)*100,"%","stdev:",results.std()*100)

n_splits = 10
test_size = 0.30
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print("K-Nearest Neighbor")
print("Percentage split 70%")
print("Accuracy:", round(results.mean(),8)*100,"%","stdev:",results.std()*100)

n_splits = 10
test_size = 0.2
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print("K-Nearest Neighbor")
print("Percentage split 80%")
print("Accuracy:", round(results.mean(),8)*100,"%","stdev:",results.std()*100)

n_splits = 10
test_size = 0.1
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print("K-Nearest Neighbor")
print("Percentage split 90%")
print("Accuracy:", round(results.mean(),8)*100,"%","stdev:",results.std()*100)

n_splits = 10
test_size = 0.5
seed = 7
kfold = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
model = KNeighborsClassifier()
results = cross_val_score(model, X, Y, cv=kfold)
print("K-Nearest Neighbor")
print("Percentage split 50%")
print("Accuracy:", round(results.mean(),8)*100,"%","stdev:",results.std()*100)
'''

