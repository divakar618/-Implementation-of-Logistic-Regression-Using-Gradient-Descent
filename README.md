# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm



1. Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.
 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: DIVAKAR R
RegisterNumber: 212222240026
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("dataset/ex2data1.txt", delimiter = ",")
X = data[:, [0, 1]]
y = data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0],  X[y == 1][:, 1], label = "Admitted")
plt.scatter(X[y == 0][:, 0],  X[y == 0][:, 1], label = "Not Admitted")
plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
plt.plot()
X_plot = np.linspace(-10, 10 , 100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    grad = np.dot(X.T, h - y) / X.shape[0]
    return J, grad
    
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([-24, 0.2, 0.2])
J, grad = costFunction(theta, X_train, y)
print(J)
print(grad)

def cost(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(y, np.log(h)) + np.dot(1 - y, np.log(1 - h))) / X.shape[0]
    return J
def gradient(theta, X, y):
    h = sigmoid(np.dot(X, theta))
    grad = np.dot(X.T, h - y) / X.shape[0]
    return grad
    
X_train = np.hstack((np.ones((X.shape[0], 1)), X))
theta = np.array([0, 0, 0])
res = optimize.minimize(fun = cost, x0 = theta, args = (X_train, y), method = "Newton-CG", jac = gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    plt.figure()
    plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
    plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
    
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
prob

def predict(theta, X):
    X_train = np.hstack((np.ones((X.shape[0], 1)), X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
    
np.mean(predict(res.x,X)==y)
*/
```

## Output:

![Exp5_1](https://github.com/divakar618/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121932143/7a8c4910-8090-48a4-844a-4da1f2969270)

![Exp5_2](https://github.com/divakar618/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121932143/fad8e07a-017d-4d4f-9f27-393e0f4f2a5f)


![Exp5_3](https://github.com/divakar618/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121932143/31871cf4-ecfe-470b-a1f7-9a707650ff41)
![Exp5_4](https://github.com/divakar618/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121932143/faea0637-e532-4f56-9111-a9476a0606d8)

![Exp5_5](https://github.com/divakar618/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121932143/129ff411-9e53-4d97-a62e-0d7a03aafd61)

![Exp5_6](https://github.com/divakar618/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121932143/20f396c0-b6a6-4a63-b342-6d408de830f0)

![Exp5_7](https://github.com/divakar618/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121932143/415bf7c2-d5f3-453b-963a-f3d4744399c1)

![Exp5_10](https://github.com/divakar618/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121932143/6df87fd8-2521-4f94-8e8e-80346b5c775c)



![Exp5_8](https://github.com/divakar618/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121932143/70d15f98-ef60-4c29-a487-56c3279198f0)

![Exp5_9](https://github.com/divakar618/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121932143/a84fc854-8a26-4ffe-9e96-799cd4765938)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

