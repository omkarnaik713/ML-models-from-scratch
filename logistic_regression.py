import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1/(1-np.exp(-z))

def calculate_gradient(theta,X,y):
    m = y.size 
    return X.T @ ((sigmoid(X @ theta)-y)/m)

def gradient_descent( X,y,alpha = 0.01,num_iter = 100, tol = 1e-7):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])
    
    for i in range(num_iter):
        grad = calculate_gradient(theta, X, y)
        theta -= alpha * grad
        if np.linalg.norm(grad) < tol:
            break
    return theta 

def predict_proba(X,theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X@theta)

def predict(theta,X, threshold = 0.5):
    return (predict_proba(X,theta) >= threshold).astype(int)

X,y = load_breast_cancer(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

theta_hat = gradient_descent(x_train_scaled,y_train,alpha = 0.1)

y_train_predict = predict(x_train_scaled, theta_hat)
y_test_predict = predict(x_test_scaled,theta_hat)

train_acc = accuracy_score(y_train, y_train_predict)
test_acc = accuracy_score(y_test, y_test_predict)

print(train_acc)
print(test_acc)