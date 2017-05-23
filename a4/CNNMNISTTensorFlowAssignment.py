# CNN on MNIST
import tensorflow as tf

# Start sesssion
sess = tf.Session()

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('../../data/', one_hot=True)



# Define input and output placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Define model
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # restrict to +/- 2*stddev
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# x: [batch, in_height, in_width, in_channels]
# W: [filter_height, filter_width, in_channels, out_channels]
# strides: sliding window for each dimension of x
# padding='SAME': zero padding, output dimension = input dimension / stride
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# x: [batch, height, width, channels]
# ksize: window size for each dimension of x
# strides: sliding window for each dimension of x
# padding='SAME': zero padding, output dimension = input dimension / stride
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# Reshape flat input to 2D image with single channel, [number of images, x, y, number of channels]
x_image = tf.reshape(x, [-1,28,28,1])

# First convolutional layer with pooling, 32 output maps
# W: [filter_height, filter_width, in_channels, out_channels]
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # convolve, add bias, apply ReLU
h_pool1 = max_pool_2x2(h_conv1) # max pooling, resulting feature maps are 14x14

# Second convolutional layer with pooling, map the 32 output channels from previous layer to 64 output channels
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # max pooling, resulting feature maps are 7x7

# Fully connected layer
# W_fc1: [inputs, number of neurons]
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64]) # flatten all 64 output 7x7 maps
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout (optional)
keep_prob = tf.placeholder(tf.float32) # probability that each element is kept
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Final readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2 # output of readout layer, logits/not normalized

# Training
# Loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
my_opt = tf.train.AdamOptimizer(learning_rate=0.001, 
                                beta1=0.9, 
                                beta2=0.999, 
                                epsilon=1e-08)
                                
train_step = my_opt.minimize(cross_entropy)
# Adam optimizer, default parameters learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08

# 0-1 loss
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # second argmax argument specifies axis
# Average 0-1 loss
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Run
sess.run(tf.global_variables_initializer()) # initialize variables
with sess.as_default():
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

