#!/usr/bin/env python
#
# Si Zuo, Aalto University
# si.zuo@aalto.fi
# Content based on sources listed below
#
##############################################################################
#  resource for tensorflow
##############################################################################
#  tensorflow: https://www.tensorflow.org/overview/
#              https://github.com/tensorflow/tensorflow
#  youtube channel: https://www.youtube.com/channel/UC0rqucBdTuFTjJiefW5t-IQ

#%%
##############################################################################
#   core components of tensorflow-1.x
##############################################################################
#  1. tensor: a N-dimensional vector (data)
#  2. computational graphs (flow):
#         a. each node of the graphs represents an operation
#         b. the input and the output of the node are tensors

# more in detail: https://towardsdatascience.com/a-beginner-introduction-to-tensorflow-part-1-6d139e038278

#%%
##############################################################################
#   core modules of tensorflow-1.x for neural network
##############################################################################
#  1. tf.nn: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn
#  2. tf.layers: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers
#  3. tf.contrib: https://www.tensorflow.org/versions/r1.10/api_docs/python/tf/contrib

#%%
##############################################################################
#   keras in tensorflow
##############################################################################
#  tf.keras: https://www.tensorflow.org/guide/keras/
#            https://www.tensorflow.org/api_docs/python/tf/keras/applications (pretrained model)

#%%
##############################################################################
# an example of how tensorflow-1.x works
##############################################################################
import tensorflow as tf
print(tf.__version__)

# create two constant tensor
node1 = tf.constant(3.0, dtype=tf.float32)  # with data type
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1: ",node1)
print("node2: ",node2)
print("node3: ",node3)

# create a session to execute the graph
sess = tf.Session()
print("sess.run(node1): ", sess.run(node1))
print("sess.run(node2): ", sess.run(node2))
print("sess.run(node3): ", sess.run(node3))

# %%
##############################################################################


##############################################################################
# an example of creating graph with placeholder
##############################################################################

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # equal to tf.add(a, b)

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))

# %%
##############################################################################


##############################################################################
# an example of fitting a line using tensorflow
##############################################################################
import numpy as np
import matplotlib.pyplot as plt

# create data
x = np.random.rand(1000).astype(np.float32)
weight_gt = 0.5
bias_gt = 0.8
y_gt = x * weight_gt + bias_gt

# create module
Weights = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights*x + biases

# compute loss
loss = tf.reduce_mean(tf.square(y-y_gt))

# optimization
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# initialize all the variables
init = tf.global_variables_initializer()   # !if variables are defined, initializing all the variable is needed

# create session for training
sess = tf.Session()
sess.run(init)          # start initialization

for step in range(1000):  # set number of iteration
    sess.run(train)
    if step % 20 == 0:
        print("step: ", step, sess.run([Weights, biases, loss]))
    if step == 999:
        print("step: ", step, sess.run([Weights, biases, loss]))
        weight_trained = sess.run(Weights)
        bias_trained = sess.run(biases)
        print("final step: ")
        print("weight_gt: {} vs weight_trained: {}".format(weight_gt, weight_trained))
        print("bias_gt: {} vs bias_trained: {}".format(bias_gt, bias_trained))

plt.plot(x, x * weight_gt + bias_gt, 'ro', label ='Original data')
plt.plot(x, x * weight_trained[0] + bias_trained[0], label ='Fitted line')
plt.title('Linear Regression Result')
plt.legend()
plt.show()

# %%
##############################################################################


##############################################################################
# an example of creating neural network using tensorflow
##############################################################################
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# initialize weight variable
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# initialize bias variable
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# create convolutional operation
# x is a 4d tensor with shape [batch,height,width,channels]
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# create pooling operation
# x is a 4d tensor with shape [batch,height,width,channels]
# pooling window size 2x2
# stride = 2
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding="SAME")


# create placeholder for input and groundtruth
x = tf.placeholder("float", shape=[None, 784])
y = tf.placeholder("float", shape=[None, 10])

# weight and bias for layer_1
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# reshape x from a 2d tensor([-1, 784]) into a 4d tensor([-1,28,28,1])
x_image = tf.reshape(x, [-1, 28, 28, 1])

## layer_1
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # conv + relu
h_pool1 = max_pool_2x2(h_conv1)

# weight and bias for layer_2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = weight_variable([64])

## layer_2
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# weight and bias for layer_3
# a fully-connect layer with 1024
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

## layer_3
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout layer
# avoid overfitting
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer with softmax
# the softmax layer turn the network output into probability
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

print("y_conv: ", y_conv)
print("y: ", y)
# compute the cross entropy between gt and network output
cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))

# optimization
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# model evaluation
correct_predict = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))

init = tf.initialize_all_variables()

# create session
sess = tf.Session()

# initialize variables
sess.run(init)

# start training
# import MNIST data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# train on batch
for i in range(1000):    # 1000 iterations
    batch = mnist.train.next_batch(50) # 50 samples for each iteration training
    if i%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y: batch[1], keep_prob:  0.5})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})

# test network performance
print("test accuracy %g" % sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

# %%
##############################################################################


##############################################################################
# an example of implementing neural network via tf.keras
##############################################################################
import tensorflow as tf
import matplotlib.pyplot as plt

# load data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# show data
print("shape of x_train: ", x_train.shape)
print("shape of x_test: ", x_test.shape)
plt.imshow(x_train[45], cmap=plt.cm.binary)
plt.show()

x_train, x_test = x_train / 255.0, x_test / 255.0

# construct neural network
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),   # output = activation(tf.matmul(input, kernel) + bias)
  tf.keras.layers.Dropout(0.2),                    # randomly setting a fraction rate of input units to 0
  tf.keras.layers.Dense(10, activation='softmax')
])
# configures the model for training
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
history  = model.fit(x_train, y_train, validation_split=0.30, epochs=5)
print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()
model.evaluate(x_test,  y_test, verbose=1)

# %%
##############################################################################


##############################################################################
# an example of how tensorflow 2.0 works
##############################################################################
# create two constant tensor
node1 = tf.constant(3.0, dtype=tf.float32)  # with data type
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1: ",node1)
print("node2: ",node2)
print("node3: ",node3)

# %%
##############################################################################


##############################################################################
# an example of implementing neural network via tf.keras in tensorflow 1.x and 2.0
##############################################################################
import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)

# load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

# show sample data
plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()


x_train = x_train.reshape((-1,28,28,1))
x_test = x_test.reshape((-1,28,28,1))

# build model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]),
                        filters=32, kernel_size=(3,3), strides=(1,1), padding='valid',
                        activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.30)
model.evaluate(x_test, y_test, verbose=2)

# %%
##############################################################################

##############################################################################
#  import pretrained model from keras in tensorflow 1.x and 2.0
##############################################################################
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

model = ResNet50(weights='imagenet')

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
print('Predicted:', decode_predictions(preds, top=3)[0])
# %%
##############################################################################

##############################################################################
##############################################################################
#  exercise:
#        try to improve the classification accuracy of the model on MNIST data
#        (for example:
#                  more iterations for training
#                  different network structures
#                  different loss function)
##############################################################################
##############################################################################
