import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Salary_Data.csv")

#divide into x , y x= independent , yrs of exp
#iloc[:,:-1] 1st param all rows , 2nd col -1, dropping Salary column
X=dataset.iloc[:,:-1].values  # gets only values

y=dataset.iloc[:,1].values # giving col num to get only salary col

#packaged changed sklearn.cross_validation to sklearn.model_selection

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=1/3,random_state=0)

#implement classifier based oon simple linear regression

from sklearn.linear_model import LinearRegression
simpleLinearRegression = LinearRegression()
simpleLinearRegression.fit(X_train,y_train)

y_predict = simpleLinearRegression.predict(X_test)

#predicting value if person is 11 yrs experienced
#y_predict_val = simpleLinearRegression.predict(11)
#implement graph
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,simpleLinearRegression.predict(X_train))
plt.show()


huseData = pd.read_csv("kc_house_data.csv")
space=huseData['sqft_living'].values
price=huseData['price'].values

x = np.array(space).reshape(-1, 1)
y = np.array(price)

xtrain, xtest, ytrain, ytest= train_test_split(x,y,test_size=1/3, random_state=0)

regressor = LinearRegression()
regressor.fit(xtrain,ytrain)

#Predicting the prices
pred = regressor.predict(xtest)


#Visualizing the training Test Results 
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()


#Visualizing the Test Results 
plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()
