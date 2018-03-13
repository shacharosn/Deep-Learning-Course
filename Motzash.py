import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tensorflow as tf
from random import shuffle
from tqdm import tqdm
from random import randint
import cv2
import glob
import cv2
import glob
import numpy as np
import tensorflow as tf
import datetime
import random

path_to_train = "C://Users//shachar//Downloads//train (1)//train"
DOG = 1
CAT = 0
IMAGE_SIZE = 256
size_1 = int(IMAGE_SIZE / 2)
size_2 = int(IMAGE_SIZE / 4)
size_3 = int(IMAGE_SIZE / 8)
FCL_1_number_of_nodes = 1000
FCL_2_number_of_nodes = 500


def getBatch(step, input_size, amount):
    count = step * 25
    labels = np.zeros((amount*2, 2))
    data = np.zeros((amount*2, input_size, input_size, 1))
    position = 0
    for i in range(count, count + amount):
        cat_file = cv2.imread(path_to_train + "cat." + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)

       # cat_file = cv2.resize(cat_file, (256, 256))
        data[position] = cat_file
        labels[position][CAT] = 1
        position = position + 1

        dog_file = cv2.imread(path_to_train + "dog." + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
       # dog_file = cv2.resize(dog_file, (IMAGE_SIZE, IMAGE_SIZE))
        data[position] = dog_file
        labels[position][DOG] = 1
        position = position + 1
    return data, labels

x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 1], name="Input_images")
y_ = tf.placeholder(tf.float32, [None, 2], name="Truth_labels")

W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]))
x_image = tf.reshape(x, [-1,256,256,1]) #if we had RGB, we would have 3 channels

h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

h_pool2_flat = tf.reshape(h_pool2, [-1, 64*64*64])
W_fc1 = tf.Variable(tf.truncated_normal([64*64 * 64, FCL_1_number_of_nodes], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[FCL_1_number_of_nodes]))

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([FCL_1_number_of_nodes, 2], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[2]))

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) #uses moving averages momentum

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(470):
  data, labels = getBatch(i, IMAGE_SIZE, 25)
  train_accuracy = accuracy.eval(feed_dict={x:data, y_: labels, keep_prob: 1.0})
  print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: data, y_: labels, keep_prob: 0.5})
data2, labels2 = getBatch(470, IMAGE_SIZE, 29)
print("test accuracy %g"%accuracy.eval(feed_dict={x: data2, y_: labels2, keep_prob: 1.0}))
