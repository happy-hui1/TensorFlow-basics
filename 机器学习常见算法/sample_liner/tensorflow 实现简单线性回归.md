#### tensorflow 实现简单线性回归

* 数据获取

  ```
  import numpy as np
  import tensorflow as tf
  import matplotlib.pyplot as plt
  import os
  from sklearn.utils import check_random_state
  
  # 人工生成数据
  n = 50
  XX = np.arange(n)
  rs = check_random_state(0)
  YY = rs.randint(-10, 10, size=(n,)) + 2.0*XX
  data = np.stack([XX, YY], axis=1)
  ```

* FLAGS

  ```
  # 定义 flags
  tf.app.flags.DEFINE_integer('num_epochs', 50, 'The number of training the model. Default=50')
  # 存储 所有变量在FLAGS
  FLAGS = tf.app.flags.FLAGS
  ```

* 定义变量（也就是参数）

  ```
  # 创建变量W, b
  W = tf.Variable(0.0, name='weights')
  b = tf.Variable(0.0, name='bias')
  ```

* 定义占位符

  ```
  # 创建占位符
  def inputs():
      X = tf.placeholder(tf.float32, name="X")
      Y = tf.placeholder(tf.float32, name="Y")
      return X, Y
  ```

* 定义模型

  ```
  # 定义模型
  def inference(X):
  
      return X*W +b
  
  ```

  

* 定义损失函数

  ```
  # 定义损失函数
  def loss(X, Y):
      Y_predicted = inference(X)
      return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))/(2*data.shape[0])
  
  ```

  

* 定义优化函数

  ```
  # 定义优化函数（训练函数）
  def train(loss):
      learning_rate = 0.0001
      return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
  ```

  

* 训练模型

  ```
  # 运行
  with tf.Session() as sess:
      # 初始化变量
      sess.run(tf.global_variables_initializer())
  
      X, Y = inputs()
      train_loss = loss(X, Y)
      train_op = train(train_loss)
  
     # 训练模型
      for epoch_num in range(FLAGS.num_epochs):
          loss_value, _ = sess.run([train_loss, train_op],
                                   feed_dict={X: data[:, 0], Y: data[:, 1]})
          # save the values of weight and bias
          wcoeff, bias = sess.run([W, b])
          print('epoch %d, loss=%f, weight=%f, bais=%f' %(epoch_num+1, loss_value, wcoeff, bias))
  
  ```

  

* 结果展示

```
Input_values = data[:,0]
Labels = data[:,1]
Prediction_values = data[:,0] * wcoeff + bias

# uncomment if plotting is desired!
plt.plot(Input_values, Labels, 'ro', label='main')
plt.plot(Input_values, Prediction_values, label='Predicted')
plt.show()

# # Saving the result.
# plt.legend()
# plt.savefig('plot.png')
# plt.close()
```

