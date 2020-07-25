"""
Product Recommendation System

@Eclature

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

mobiles = pd.read_csv('mobile-reviews & iteam Merged.csv')
mobiles.head(5)

#Pre-Pocessing

price = mobiles[['price']]
price

from sklearn.impute import SimpleImputer, MissingIndicator
imp = SimpleImputer(missing_values=0, strategy='median')
imp.fit(price)
SimpleImputer()
price = (imp.transform(price))
price = pd.DataFrame(price) 
price.head()
# Concatinating the Imputed price column with the main dataframe 
mb = mobiles.drop(['price'], axis=1)
mb.head()
mb.shape
mobiles_pr = pd.concat([mb, price], axis=1, sort=False)
mobiles_pr.head(5)
mobiles_pr = mobiles_pr.set_axis(['productid', 'brand','title' ,'product_rating','totalReviews','originalprice','name','userid','user_rating','date','verifed','title.1','body','helpfulvotes','price'], axis=1, inplace=False)
mobiles_pr.head(5)
mobiles = mobiles_pr[['productid','brand','product_rating','totalReviews','user_rating','price']]
mobiles


from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder() 
mobiles['brand']= le.fit_transform(mobiles['brand'])
mobiles['productid'] = le.fit_transform(mobiles['productid'])
mobiles

import seaborn as sns
plt.figure(figsize=(5,5))
c = mobiles.corr()
sns.heatmap(c,cmap='BrBG',annot=True)
print(c)


X = mobiles.iloc[:, :5]
X

y = mobiles.iloc[:, -1]

# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


model = LinearRegression()
model.fit(x_train, y_train)

print(model.coef_)
print(model.intercept_)

predictions = model.predict(x_test)

plt.hist(y_test - predictions)

print(metrics.mean_absolute_error(y_test, predictions))
print(metrics.mean_squared_error(y_test, predictions))
print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


print(model.predict([[16, 8, 3.2, 107,4]]))