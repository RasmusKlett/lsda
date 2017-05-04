# Logistic Regression

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Flags
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summary_dir', '/tmp/logLog', 'directory to put the summary data')
flags.DEFINE_string('data_dir', '../data', 'directory with data')
flags.DEFINE_integer('maxIter', 10000, 'number of iterations')
flags.DEFINE_float('learning_rate',0.1,'learning rate')


# Read data
dataTrain = tf.contrib.learn.datasets.base.load_csv_without_header(filename=FLAGS.data_dir + '/LSDA2017WeedCropTrain.csv', target_dtype=np.int, features_dtype=np.float32, target_column=-1)
dataTest = tf.contrib.learn.datasets.base.load_csv_without_header(filename=FLAGS.data_dir + '/LSDA2017WeedCropTest.csv', target_dtype=np.int, features_dtype=np.float32, target_column=-1)

# Input dimension
inDim = dataTrain.data.shape[1]

# Create graph
sess = tf.Session()

# Initialize placeholders
x_data = tf.placeholder(shape=[None, inDim], dtype=tf.float32, name='input')
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='target')

# TensorBoard will collapse the following nodes 
with tf.name_scope('model') as scope:
    # Create variables for logistic regression
    A = tf.Variable(tf.random_normal(shape=[inDim,1]))
    b = tf.Variable(tf.random_normal(shape=[1,1]))

    # Declare model operations
    model_output = tf.add(tf.matmul(x_data, A), b)

# Declare loss function 
with tf.name_scope('loss') as scope:
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))
    tf.summary.scalar('cross-entropy', loss)

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
train_step = my_opt.minimize(loss)

# Map model output to binary predictions
with tf.name_scope('binary_prediction') as scope:
    prediction = tf.round(tf.sigmoid(model_output))
with tf.name_scope('0-1-loss') as scope:
    predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
    accuracy = tf.reduce_mean(predictions_correct)
    tf.summary.scalar('accuracy', accuracy)

# Logging
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train')
test_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/test')

# Initialize variables
init = tf.global_variables_initializer()
sess.run(init)

# Training loop
for i in range(FLAGS.maxIter):
    sess.run(train_step, feed_dict={x_data: dataTrain.data, y_target: np.transpose([dataTrain.target])})
    summary = sess.run(merged, feed_dict={x_data: dataTrain.data, y_target: np.transpose([dataTrain.target])})
    train_writer.add_summary(summary, i)
    summary = sess.run(merged, feed_dict={x_data: dataTest.data, y_target: np.transpose([dataTest.target])})
    test_writer.add_summary(summary, i)

print("final training accuracy:", sess.run(accuracy, feed_dict={x_data: dataTrain.data, y_target: np.transpose([dataTrain.target])}), "final test accuracy: ", sess.run(accuracy, feed_dict={x_data: dataTest.data, y_target: np.transpose([dataTest.target])}))
