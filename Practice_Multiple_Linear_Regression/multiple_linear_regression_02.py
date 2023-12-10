import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Practice_Multiple_Linear_Regression\kc_house_data.csv')
data = data.drop(['id','date'], axis=1)

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(y_pred)