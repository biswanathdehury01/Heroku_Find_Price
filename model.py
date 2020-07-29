# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:17:53 2020

@author: Rakesh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

mobiles = pd.read_csv('C:/Users/Rakesh/Downloads/Recommendation-Engine-master1/Recommendation-Engine-master/procedural programming/mobile-reviews & iteam Merged.csv')
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


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


print(model.predict([[16, 8, 3.2, 107,4]]))

















