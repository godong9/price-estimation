#!/usr/bin/python

import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

tf.set_random_seed(777)  # reproducibility

global_step = tf.Variable(0, trainable=False, name='global_step')

stock = sys.argv[1]
print("Stock:", stock)

def MinMaxScaler(data):
    ''' Min Max Normalization
    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]
    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]
    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

# step debug Parameters
debug_step = 100

# -1:Close, -2:Volume, -3:Low, -4:High, -5:Open
prediction_label = -1

# train Parameters
seq_length = 7
data_dim = 5
hidden_dim = 10
output_dim = 1
learning_rate = 0.01
iterations = 2000
LSTM_stack = 2
output_keep_prob = 1.0

# Date, Open, High, Low, Close, Adj Close, Volume
df = pd.read_csv("stock/" + stock + "_stock.csv")
df = df.drop(columns=["Date", "Adj Close"])
cols = df.columns.tolist()
cols = cols[:3] + cols[-1:] + cols[3:4]
df = df[cols]

# Open, High, Low, Volume, Close
xy = df.values

# train/test split
train_size = int(len(xy) * 0.7)
train_set = xy[0:train_size]
test_set = xy[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence

test_min = np.min(test_set, 0)
test_max = np.max(test_set, 0)
test_denominator = test_max - test_min

# Scale each
train_set = MinMaxScaler(train_set)
test_set = MinMaxScaler(test_set)
prediction_set = test_set[-seq_length:]

# build datasets
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length, :]
        _y = time_series[i + seq_length, [prediction_label]]  # Next prediction_label
        # print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, seq_length)
testX, testY = build_dataset(test_set, seq_length)
predictionX = [prediction_set]

# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
    return cell

multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(LSTM_stack)], state_is_tuple=True)

outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss, global_step=global_step)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())

    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
        iterations = 0
    else:
        sess.run(tf.global_variables_initializer())

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
            X: trainX, Y: trainY})
        global_step_val = sess.run(global_step)
        if global_step_val % debug_step == 0:
            print("[step: {}] loss: {}".format(global_step_val, step_loss))

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
        targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    saver.save(sess, './model/estimate.ckpt', global_step=global_step)

    real_prediction = sess.run(Y_pred, feed_dict={X: predictionX})

    print("Today Prediction:", test_predict[-1] * (test_denominator[prediction_label] + 1e-7) + test_min[prediction_label])
    print("Today Real:", testY[-1] * (test_denominator[prediction_label] + 1e-7) + test_min[prediction_label])
    print("Prediction:", real_prediction[0] * (test_denominator[prediction_label] + 1e-7) + test_min[prediction_label])

    # Plot predictions
    plt.figure(1)
    plt.plot(testY, label="Real")
    plt.plot(test_predict, label="Prediction")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()

    # Plot small predictions
    plt.figure(2)
    plt.plot(testY[-100:], label="Real")
    plt.plot(test_predict[-100:], label="Prediction")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()