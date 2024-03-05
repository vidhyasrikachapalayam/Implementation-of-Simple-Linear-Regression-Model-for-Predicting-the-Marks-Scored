# EX 1 Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VIDHYASRI.K
RegisterNumber:  212222230170
*/
```

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred =reg.predict(X_test)
print(Y_pred)
print(Y_test)

#Graph plot for training data
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


## Output:
## 1. df.head()
![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477817/5ab8651b-4a4c-4094-9702-10ed29469f8c)
## df.tail()
![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477817/37d043db-2a5f-4fc0-871c-f697ce0f9232)
## 3. Array value of X
![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477817/f1ade88a-6b92-4d4e-9647-51eb0964acac)
## 4. Array value of Y
![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477817/8c69af2d-8489-4285-91c5-083d7cf037d1)
## Values of Y prediction
![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477817/0ddcf331-1294-4f71-b402-b67ef571eef8)
## 6. Array values of Y test
![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477817/2f93448f-78c0-4ec6-8d53-91e578d2365d)
## 7. Training Set Graph
![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477817/eb714bd0-7eb2-440c-8e07-0118a130e5bd)
## TEST SET GRAPH
![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477817/862b0254-0140-45e3-9dd2-5633045197f8)
## 9. Values of MSE, MAE and RMSE

![image](https://github.com/vidhyasrikachapalayam/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119477817/0851f0b0-e9e5-4a37-9e59-c9bc0ef8067d)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
