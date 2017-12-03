# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # need colon to specify x as a matrix
y = dataset.iloc[:, 2].values

"""# Splitting the dataset into the Training set and Test set --not necessary for such a small dataset
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y) # didn't split into test and training set this time

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # Transforms original matrix of features x into new matrix containing original independent variables and it's associated polynomial terms
X_poly = poly_reg.fit_transform(X) # automatically creates column of ones also
lin_reg_2 = LinearRegression # Once new matrix of polynomial features is created, create a new LinearRegression object to fit to the new matrix x and original dependent variable y
lin_reg_2.fit(X_poly,y)

# Visualizing the Linear Regression results
plt.scatter(X, y,color = 'red')
plt.plot(X, lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the Polynomial Regression results
plt.scatter(X, y ,color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()