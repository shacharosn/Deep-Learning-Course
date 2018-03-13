
import numpy as np
import tensorflow as tf
from numpy import genfromtxt


my_data = genfromtxt('childrenDataset.csv', delimiter=',',dtype='unicode')
data_y=np.array(my_data[1:238,2])
data_x=np.array(my_data[1:238,0:3])
data_x2=np.array(my_data[1:238,3])

for i in range(0,237):
 a = data_x2[i]
 data_x[i][2]=a
 if data_x[i][0]=='f':
     data_x[i][0]=-1
     data_x[i][0] =float(data_x[i][0])

 if data_x[i][0]=='m':
     data_x[i][0]=1
     data_x[i][0] = float(data_x[i][0])


data_x=data_x.astype(np.double)
data_y=data_y.astype(np.double)
data_x_training= np.array(data_x[:200,])
data_x_test= np.array(data_x[200:,])
data_y_training= np.array(data_y[:200,])
data_y_test= np.array(data_y[200:,])

w = np.array([0,0,0])
w=w.astype(np.double)
b = 0
alpha = 0.000005
for iteration in range(10000):
    gradient_b = np.mean(1*(data_y_training-(np.dot(data_x_training,w)+b)))
    gradient_w = 1.0/len(data_y_training) * np.dot((data_y_training-(np.dot(data_x_training,w)+b)), data_x_training)
    b += alpha*gradient_b
    w += alpha*gradient_w

sum=0
for i in range(len(data_y_test)):
    Gender= data_x_test[i][0]
    Age= data_x_test[i][1]
    Height= data_x_test[i][2]
    Weight= data_y_test[i]
    Estimated_Weight= np.dot(np.array([Gender, Age, Height]).astype(np.double), w) + b
    g= (100*abs(Weight-Estimated_Weight))/Weight
    sum=sum+g
    print('Estimated Weight:',Estimated_Weight, '|       real Weight:' ,Weight, '|      deviation',g,'%')
print()
print('The average deviation is:', sum/len(data_y_test),'%')











