
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import tensorflow as tf
import csv
import numpy as np
import pandas as pd
from numpy import genfromtxt

train = pd.read_table("C:\\Users\\shachar\\Desktop\\trainprod.tsv")
#train= train[0:140,1:6]
print(555)

train.category_name.fillna(value="missing", inplace=True)
train.brand_name.fillna(value="missing", inplace=True)
train.name.fillna(value="missing", inplace=True)
train.item_condition_id.fillna(value="missing", inplace=True)
train.shipping.fillna(value="missing", inplace=True)




le = LabelEncoder()
le.fit(np.hstack([train.category_name]))
train.category_name = le.transform(train.category_name)
le.fit(np.hstack([train.brand_name]))
train.brand_name = le.transform(train.brand_name)
le.fit(np.hstack([train.name]))
train.name = le.transform(train.name)



#data_xx = train.loc[:, 2, 4, 6]
data_xx = np.array(train[['shipping','item_condition_id','brand_name']])
#dx = data_xx.to_records(index=False)
#ss=np.array(dx)
#ss2=np.vstack( ss )

data_yy = np.array(train[['price']])
#dy = data_yy.to_records(index=False)
#np.asarray(dy)

print(data_xx)
print(data_xx.size)


features = 3
(hidden1_size, hidden2_size) = (100, 50)

x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([features,1]))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(x,W) + b
loss = tf.reduce_mean(tf.pow(y - y_, 2))
update = tf.train.GradientDescentOptimizer(0.000001).minimize(loss)
data_x = data_xx
data_y= data_yy
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0,100000):
    sess.run(update, feed_dict = {x:data_x, y_:data_y})
    print('Iteration:' , i , ' W:' , sess.run(W) , ' b:' , sess.run(b), ' loss:', loss.eval(session=sess, feed_dict = {x:data_x, y_:data_y}))

print('prediction: ', y.eval(session=sess, feed_dict = 	{x:[[0,153,164], [0,178,164], [0,142,139] ]}))


