import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tensorflow as tf
from random import shuffle
from tqdm import tqdm
from random import randint



TRAIN_DIR = "C://Users//shachar//Downloads//train (1)//train"
IMG_SIZE = 40
size= 100


def label_img(img):
    word_label = img.split('.')[-3]

    if word_label == 'cat': return [1,0]

    elif word_label == 'dog': return [0,1]

def batch(len):
    begin= randint(0, len)
    if (begin + size)<len:
        end = begin + size
    else:end= len-1

    return begin, end



training_data = []
for img in tqdm(os.listdir(TRAIN_DIR)):
    label = label_img(img)
    path = os.path.join(TRAIN_DIR, img)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    training_data.append([np.array(img), np.array(label)])
shuffle(training_data)
#print(training_data)

train = training_data[:-500]
test = training_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,1600)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,1600)
test_y = [i[1] for i in test]


x = tf.placeholder(tf.float32, [None, 1600])
y_ = tf.placeholder(tf.float32, [None, 2])
W = tf.Variable(tf.random_normal([1600, 2], stddev=0.00001))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

scores1 = []
scores2 = []

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

#print(train)

for i in range(10000):
  start, end= batch(len(train))
  temp_train_x = X[start:end]
  sess.run(train_step, feed_dict={x: X, y_: Y})
  if i % 100 == 0:
      print('Iteration:', i, ' W3:', sess.run(W), ' b3:', sess.run(b), ' loss:',
            cross_entropy.eval(session=sess, feed_dict={x: X, y_: Y}))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_x, y_: test_y}))