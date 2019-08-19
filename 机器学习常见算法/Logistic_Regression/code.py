import numpy as np
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data

###### 定义Flags
tf.app.flags.DEFINE_string(
    'train_path', os.path.dirname(os.path.abspath(__file__)) + '/train_logs',
    'Directory where event logs are written to.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', os.path.dirname(os.path.abspath(__file__)) + '/checkpoints',
    'Directory where checkpoints are written to.')
tf.app.flags.DEFINE_integer('max_num_checkpoint', 10,
                            'Maximum number of checkpoints that TensorFlow will keep.')
tf.app.flags.DEFINE_integer('num_classes', 2,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_integer('batch_size', 512,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_integer('num_epochs', 10,
                            'Number of epochs for training.')
# 变速学习率参数FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 1, 'Number of epoch pass to decay learning rate.')

# 状态参数
tf.app.flags.DEFINE_boolean('is_training', False, 'Training/Testing.')
tf.app.flags.DEFINE_boolean('fine_tuning', False, 'Fine tuning is desired or not?.')
tf.app.flags.DEFINE_boolean('online_test', True, 'Fine tuning is desired or not?.')

tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Automatically put the variables on CPU if there is no GPU support.')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Demonstrate which variables are on what device.')

FLAGS = tf.app.flags.FLAGS

## 检验错误
if not os.path.isabs(FLAGS.train_path):
    raise ValueError('You must assign absolute path for --train_path')
if not os.path.isabs(FLAGS.checkpoint_path):
    raise ValueError('You must assign absolute path for --checkpoint_path')

# 下载数据
mnist = input_data.read_data_sets("MNIST_data/", reshape=True, one_hot=False)

data={}
data['train/image'] = mnist.train.images
data['train/label'] = mnist.train.labels
data['test/image'] = mnist.test.images
data['test/label'] = mnist.test.labels

## 只取标签为0和1的数据集
def extract_samples_Fn(data):
    index_list = []
    for sample_index in range(data.shape[0]):
        label = data[sample_index]
        if label == 1 or label == 0:
            index_list.append(sample_index)
    return index_list

index_list_train = extract_samples_Fn(data['train/label'])
index_list_test = extract_samples_Fn(data['test/label'])
data['train/image'] = mnist.train.images[index_list_train]
data['train/label'] = mnist.train.labels[index_list_train]
data['test/image'] = mnist.test.images[index_list_test]
data['test/label'] = mnist.test.labels[index_list_test]

# 记录数据的大小
dimensionality_train = data['train/image'].shape
num_train_samples = dimensionality_train[0]
num_features = dimensionality_train[1]


# 定义图
graph = tf.Graph()
with graph.as_default():
    # 将定义的图变为tf默认的图

    # 生成衰减的学习律
    global_step = tf.Variable(0, name="global_step", trainable=False)
    decay_steps = int(num_train_samples / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               FLAGS.learning_rate_decay_factor,
                                               staircase=True,
                                               name='exponential_decay_learning_rate')
    # 定义占位符，
    image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
    label_place = tf.placeholder(tf.int32, shape=([None, ]), name='gt')
    label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
    dropout_param = tf.placeholder(tf.float32)

    # 定义模型，这里是简单的全连接层
    logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs=FLAGS.num_classes, scope='fc')
    # 定义损失函数
    with tf.name_scope('loss'):
        loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))

    # 计算预测精确度
    prediction_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    with tf.name_scope('train_op'):
        gradients_and_variables = optimizer.compute_gradients(loss_tensor)
        train_op = optimizer.apply_gradients(gradients_and_variables, global_step=global_step)
        # minimize()中global_step=global_step能够提供global_step自动加一的操作。

    # 开始运行训练
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(graph=graph, config=session_conf) # 设置Session的参数
    with sess.as_default():
        # The saver op.
        saver = tf.train.Saver()

        # 初始化所有变量
        sess.run(tf.global_variables_initializer())

        checkpoint_prefix = 'model'

        if FLAGS.fine_tuning:
            saver.restore(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
            print("Model restored for fine-tuning...")
            # go through the batches

        test_accuracy = 0
        for epoch in range(FLAGS.num_epochs):
            total_batch_training = int(data['train/image'].shape[0] / FLAGS.batch_size)

            for batch_num in range(total_batch_training):
                start_idx = batch_num * FLAGS.batch_size
                end_idx = (batch_num + 1) * FLAGS.batch_size
                train_batch_data, train_batch_label = data['train/image'][start_idx:end_idx], \
                                                      data['train/label'][start_idx:end_idx]

                batch_loss, _, training_step = sess.run(
                    [loss_tensor, train_op,
                     global_step],
                    feed_dict={image_place: train_batch_data,
                               label_place: train_batch_label,
                               dropout_param: 0.5})
            print("Epoch " + str(epoch + 1) + ", Training Loss= " + \
                  "{:.5f}".format(batch_loss))

        if not os.path.exists(FLAGS.checkpoint_path):
            os.makedirs(FLAGS.checkpoint_path)
        save_path = saver.save(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
        print("Model saved in file: %s" % save_path)

        # The prefix for checkpoint files
        checkpoint_prefix = 'model'
        # Restoring the saved weights.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
        print("Model restored...")

        # Evaluation of the model
        test_accuracy = 100 * sess.run(accuracy, feed_dict={
            image_place: data['test/image'],
            label_place: data['test/label'],
            dropout_param: 1.})

        print("Final Test Accuracy is %% %.2f" % test_accuracy)