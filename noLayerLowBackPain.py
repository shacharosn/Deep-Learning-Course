import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.metrics import accuracy_score

# read data into dataset variable
data = pd.read_csv("C:\\Users\\shachar\\Desktop\\Dataset_spine.csv")


# Drop the unnamed column in place (not a copy of the original)#
data.drop('Unnamed: 13', axis=1, inplace=True)

# Concatenate the original df with the dummy variables
data = pd.concat([data, pd.get_dummies(data['Class_att'])], axis=1)

# Drop unnecessary label column in place.
data.drop(['Class_att','Normal'], axis=1, inplace=True)

data.columns = ['Pelvic Incidence','Pelvic Tilt','Lumbar Lordosis Angle','Sacral Slope','Pelvic Radius',
                'Spondylolisthesis Degree', 'Pelvic Slope', 'Direct Tilt', 'Thoracic Slope',
                'Cervical Tilt','Sacrum Angle', 'Scoliosis Slope','Outcome']
print(data)
data=data.sample(frac=1)
print(data.head())


#   Create the training dataset
training = data.drop('Outcome', axis=1)
testing = data['Outcome']

#   Split into training/testing datasets using Train_test_split
X_train, X_test, y_train, y_test = train_test_split(training, testing, test_size=0.33, random_state=22, stratify=testing)

#print(y_train)
# convert to numpy.ndarray and dtype=float64 for optimal
array_train = np.asarray(training)
array_test = np.asarray(testing)


#   Convert each pandas DataFrame object into a numpy array object.
array_XTrain, array_XTest, array_ytrain, array_ytest = np.asarray(X_train),\
                                                       np.asarray(X_test), np.asarray(y_train), np.asarray(y_test)


array_yytrain = np.array(array_ytrain[:,None])
array_yytest = np.array(array_ytest[:,None])
features = 12

eps = 1e-12
x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.zeros([features,1]))
b = tf.Variable(tf.zeros([1]))
y = 1 / (1.0 + tf.exp(-(tf.matmul(x,W) + b)))
t = 1-y
loss1 = -(y_ * tf.log(y + eps) + (1 - y_) * tf.log(t + eps))
loss = tf.reduce_mean(loss1)
update = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)


scores1 = []
scores2 = []

data_x = array_XTrain
data_y= array_yytrain
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(0,10000):
    sess.run(update, feed_dict = {x:data_x, y_:data_y})
    scores1.append(loss.eval(session=sess, feed_dict = {x:data_x, y_:data_y}))
    scores2.append(loss.eval(session=sess, feed_dict = {x:array_XTest, y_:array_yytest}))
    if i %1000==0:
     print('Iteration:' , i , ' W3:' , sess.run(W) , ' b3:' , sess.run(b), ' loss:', loss.eval(session=sess, feed_dict = {x:data_x, y_:data_y}))
print(y.shape)
print('prediction: ', y.eval(session=sess, feed_dict = 	{x:array_XTest}))

temp_y= y.eval(session=sess, feed_dict = 	{x:array_XTest})
temp_y = temp_y > 0.5
temp_y = temp_y.astype(int)
temp_y= temp_y.flatten()

print(temp_y)
print(' loss:', loss.eval(session=sess, feed_dict={x: array_XTest, y_: array_yytest}))

svmscore = accuracy_score(array_ytest, temp_y)
print("Support Vector Machines are ", svmscore*100, "accurate")
plt.plot(scores1)
plt.plot(scores2)
plt.title('No hiden layer')
plt.legend(['Train error', 'Test error'], loc='upper right')
plt.show()


