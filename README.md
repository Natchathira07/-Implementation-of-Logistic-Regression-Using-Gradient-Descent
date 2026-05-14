# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries.

2.Load the placement dataset.

3.Remove unnecessary columns like sl_no and salary.

4.Convert categorical columns into category type.

5.Encode categorical values into numbers.

6.Separate input features (x) and output label (y).

7.Initialize random weights (theta).

8.Define sigmoid function to map values between 0 and 1.

9.Define loss function to measure prediction error.

10.Apply gradient descent to update weights.

11.Predict placement results using trained weights.

12.Calculate accuracy by comparing predictions with actual values.

13.Test the model with new input data.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: VD Natchathira
RegisterNumber:212224230178
*/
import pandas as pd
import numpy as np

data = pd.read_csv("Placement_Data.csv")
data1 = data.drop(['sl_no', 'salary'], axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

X = data1.iloc[:, :-1].values  
Y = data1["status"].values  

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

theta = np.random.randn(X.shape[1])  
alpha = 0.01  # Learning rate
num_iterations = 1000  
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15)) / len(y)

def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, Y, alpha, num_iterations)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    return np.where(h >= 0.5, 1, 0)

y_pred = predict(theta, X)
accuracy = np.mean(y_pred == Y)

print("Accuracy:", accuracy)
print("\nPredicted:\n", y_pred)
print("\nActual:\n", Y)

xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])  # Example input
xnew = scaler.transform(xnew)  # Apply same scaling as training data
y_prednew = predict(theta, xnew)
print("\nPredicted Result:", y_prednew)
```

## Output:
<img width="901" height="486" alt="Screenshot 2026-05-14 094545" src="https://github.com/user-attachments/assets/07e2e5b9-9811-4102-b9e4-73e88948a7d1" />

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
