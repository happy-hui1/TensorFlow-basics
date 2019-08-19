# # 在python2的环境下，超前使用python3的print函数。
# from __future__ import print_function
import tensorflow as tf
import os

# os.path.dirname(__file__)返回脚本的路径
log_dir = os.path.dirname(os.path.abspath(__file__)) + '/logs'

# Defining some sentence!
welcome = tf.constant('Welcome to TensorFlow world!')
# Run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(log_dir), sess.graph)
    # tf.summary.FileWriter(path, sess.graph)指定一个文件用来保存图
    print("output: ", sess.run(welcome))

# Closing the writer.
writer.close()
sess.close()

# class SquareTest(tf.test.TestCase):
#
#   def testSquare(self):
#     with self.test_session():
#       x = tf.square([2, 3])
#       self.assertAllEqual(x.eval(), [4, 9])
#
# if __name__ == '__main__':
#     tf.test.main()