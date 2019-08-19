import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
import sys


tf.app.flags.DEFINE_integer('batch_size', 32,
                            'Number of samples per batch.')

tf.app.flags.DEFINE_integer('num_steps', 5000,
                            'Number of steps for training.')

tf.app.flags.DEFINE_boolean('is_evaluation', True,
                            'Whether or not the model should be evaluated.')

tf.app.flags.DEFINE_float(
    'C_param', 0.1,
    'penalty parameter of the error term.')

tf.app.flags.DEFINE_float(
    'Reg_param', 1.0,
    'penalty parameter of the error term.')

tf.app.flags.DEFINE_float(
    'delta', 1.0,
    'The parameter set for margin.')

tf.app.flags.DEFINE_float(
    'initial_learning_rate', 0.01,
    'The initial learning rate for optimization.')

FLAGS = tf.app.flags.FLAGS


def inference_fn(x_data, y_target):
    prediction = tf.sign(tf.add(tf.matmul(x_data, W), b))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
    return accuracy


def loss_fn(x_data,y_target):
    logits = tf.add(tf.matmul(x_data, W),b)
    norm_term = tf.divide(tf.reduce_sum(tf.multiply(tf.transpose(W),W)),2)
    classification_loss = tf.reduce_mean(tf.maximum(0., tf.subtract(1.0, tf.multiply(logits, y_target))))
    total_loss = tf.add(tf.multiply(1.0,classification_loss), tf.multiply(FLAGS.Reg_param, norm_term))
    return total_loss

def next_batch_fn(x_train,y_train,num_samples=FLAGS.batch_size):
    index = np.random.choice(len(x_train), size=num_samples)
    X_batch = x_train[index]
    y_batch = np.transpose([y_train[index]])
    return X_batch, y_batch

# 定义优化函数（训练函数）
def train(loss):
    return tf.train.GradientDescentOptimizer(FLAGS.initial_learning_rate).minimize(loss)

iris = datasets.load_iris()
X = iris.data[:, :2]
y = np.array([1 if label == 0 else -1 for label in iris.target])
my_randoms = np.random.choice(X.shape[0], X.shape[0], replace=False)
train_indices = my_randoms[0:int(0.5 * X.shape[0])]
test_indices = my_randoms[int(0.5 * X.shape[0]):]
x_train = X[train_indices]
y_train = y[train_indices]
x_test = X[test_indices]
y_test = y[test_indices]


with tf.Session() as sess:
    x_data = tf.placeholder(shape=[None, X.shape[1]], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
    W = tf.Variable(tf.random_normal(shape=[X.shape[1], 1]), name="weight")
    b = tf.Variable(tf.random_normal(shape=[1, 1]), name="bias")

    accuracy = inference_fn(x_data, y_target)
    total_loss = loss_fn(x_data, y_target)
    train_op = train(total_loss)

    sess.run(tf.global_variables_initializer())
    for step_idx in range(FLAGS.num_steps):
        # Get the batch of data.
        X_batch, y_batch = next_batch_fn(x_train, y_train, num_samples=FLAGS.batch_size)
        sess.run(train_op, feed_dict={x_data: X_batch, y_target: y_batch})

        loss_step = sess.run(total_loss, feed_dict={x_data: X_batch, y_target: y_batch})
        train_acc_step = sess.run(accuracy, feed_dict={x_data: x_train, y_target: np.transpose([y_train])})
        test_acc_step = sess.run(accuracy, feed_dict={x_data: x_test, y_target: np.transpose([y_test])})
        if step_idx % 500 == 0:
            print('Step #%d, training accuracy= %% %.2f, testing accuracy= %% %.2f ' % (
                step_idx, float(100 * train_acc_step), float(100 * test_acc_step)))
            print(loss_step)

    if FLAGS.is_evaluation:
        [[w1], [w2]] = sess.run(W)
        [[bias]] = sess.run(b)
        x_line = [data[1] for data in X]

        # Find the separator line.
        line = []
        line = [-w2 / w1 * i - bias / w1 for i in x_line]

        # coor_pos_list = [positive_X, positive_y]
        # coor_neg_list = [negative_X, negative_y]

        for index, data in enumerate(X):
            if y[index] == 1:
                positive_X = data[1]
                positive_y = data[0]
                plt.plot(positive_X, positive_y, '+')
            elif y[index] == -1:
                negative_X = data[1]
                negative_y = data[0]
                plt.plot(negative_X, negative_y, 'o')
            else:
                sys.exit("Invalid label!")

        plt.plot(x_line, line, 'r-', label='Separator', linewidth=3)
        plt.legend(loc='best')
        plt.title('Linear SVM')
        plt.show()