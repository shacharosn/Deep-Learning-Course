import csv
import numpy as np
import pandas as pd
from numpy import genfromtxt


my_data = genfromtxt('childrenDataset.csv', delimiter=',',dtype='unicode')
data_y=np.array(my_data[1:238,0])
data_x=np.array(my_data[1:238,1:4])


for i in range(0,237):
 if data_y[i]=='f':
     data_y[i]=0
     data_y[i] = float(data_y[i])

 if data_y[i]=='m':
     data_y[i]=1
     data_y[i] = float(data_y[i])


data_x=data_x.astype(np.double)
data_y=data_y.astype(np.double)
data_x_training= np.array(data_x[:200,])
data_x_test= np.array(data_x[200:,])
data_y_training= np.array(data_y[:200,])
data_y_test= np.array(data_y[200:,])

def h(x,w,b):
    return 1 / (1+np.exp(-(np.dot(x,w) + b)))

w = np.array([0,0,0])
w=w.astype(np.double)
b = 0
alpha = 0.00001
for iteration in range(10000):
    gradient_b = np.mean(1*(data_y_training-(h(data_x_training,w,b))))
    gradient_w = np.dot((data_y_training-h(data_x_training,w,b)), data_x_training)*1/len(data_y_training)
    b += alpha*gradient_b
    w += alpha*gradient_w
    if iteration % 10 == 0:
     print(np.sum(np.mean((h(data_x_training, w, b)-data_y_training))))


sum=0
for i in range(len(data_y_test)):
    Age = data_x_test[i][0]
    Weight= data_x_test[i][1]
    Height= data_x_test[i][2]
    Gender= data_y_test[i]
    Estimated_Gender= h(np.array([[Age, Weight, Height]]),w,b)
    g= (100*abs(Gender-Estimated_Gender))/Gender
    sum=sum+g
    print('Estimated prize:',Estimated_Gender, '|       real prize:' ,Gender, '|      deviation',g,'%')
print()
print('loss:',np.sum(np.mean((h(data_x_training, w, b) - data_y_training)))*10,'%')
print()

print('The average deviation is:', sum/len(data_y_test),'%')




