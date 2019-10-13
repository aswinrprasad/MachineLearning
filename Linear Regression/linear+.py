
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('weather.csv')
df.head()
new_data = df[['MaxTemp', 'Rainfall','Humidity3pm','Cloud9am','RISK_MM']]
new_data.head()

from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
data=new_data.values

def norm(x):
    min_max_scaler=MinMaxScaler()
    X_scaled=min_max_scaler.fit_transform(x)
    return X_scaled

x=data[:, 0:4]
y=data[:,4]
y=pd.Series(np.where(y == 'Yes', 1, 0),y)

x = norm(x)
y = norm(y.reshape(-1,1))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)


def cost(x,theta,y):
    temp1 = (np.dot(x,theta)-y).reshape(1,len(x[:,0]))
    cost = []
    for p in range(len(theta)):
        tempx = x[:,p]
        temp2 = np.dot(temp1,tempx)
        tempsum = temp2.sum()
        cost.append(tempsum/(len(y)))
    return np.array(cost)


def gradient_descent(alpha, x,y,norma,max_iter=1500):
    if(norma==True):
        x = norm(x.reshape(-1,1))
        y = norm(y.reshape(-1,1))
    theta = np.random.rand(x.shape[1]+1,1)
    temp = np.ones(len(x))
    y = np.array(y)
    x = np.vstack((temp.T,x.T)).T
    y = y.reshape(len(y),1)
    for i in range(max_iter):
        costval = cost(x,theta,y)
        for j in range(x.shape[1]):
            theta[j] -=alpha*costval[j]
    return theta


theta = gradient_descent(0.001, x_train, y_train,False)

temp = np.ones(len(x_test))
x_test = np.vstack((temp.T, x_test.T))
y_pred = np.dot(x_test.T, theta)
mse = (1/(len(x_test))) * np.square(y_pred-y_test).sum()
print("Mean squared error",mse)


