#### 1-basic

##### basic_math_operation

* 代码

  ```
  import tensorflow as tf
  import os
  
  # The default path for saving event files is the same folder of this python file.
  tf.app.flags.DEFINE_string(
      'log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
      'Directory where event logs are written to.')
  
  # Store all elemnts in FLAG structure!
  FLAGS = tf.app.flags.FLAGS
  
  if not os.path.isabs(os.path.expanduser(FLAGS.log_dir)):
      raise ValueError('You must assign absolute path for --log_dir')
  # os.path.expanduser(path)  #把path中包含的"~"和"~user"转换成用户目录
  # os.path.isabs(path)  #判断是否为绝对路径
  # 祥见https://www.cnblogs.com/xupeizhi/archive/2013/02/20/2918243.html
  
  # Defining some constant values
  a = tf.constant(5.0, name="a")
  b = tf.constant(10.0, name="b")
  
  # Some basic operations
  x = tf.add(a, b, name="add")
  y = tf.div(a, b, name="divide")
  
  # Run the session
  with tf.Session() as sess:
      writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
      print("a =", sess.run(a))
      print("b =", sess.run(b))
      print("a + b =", sess.run(x))
      print("a/b =", sess.run(y))
  
  # Closing the writer.
  writer.close()
  sess.close()
  ```

* 代码解释补充

  * tf 中定义了 tf.app.flags.FLAGS ，用于接受从终端传入的命令行参数，相当于对Python中的命令行参数模块optpars做了一层封装。

    

  optpars中的参数类型是通过参数 “type=xxx” 定义的，tf中每个合法类型都有对应的 “DEFINE_xxx”函数。常用：
  tf.app.flags.DEFINE_string() ：定义一个用于接收 string 类型数值的变量;
  tf.app.flags.DEFINE_integer() : 定义一个用于接收 int 类型数值的变量;
  tf.app.flags.DEFINE_float() ： 定义一个用于接收 float 类型数值的变量;
  tf.app.flags.DEFINE_boolean() : 定义一个用于接收 bool 类型数值的变量;

  “DEFINE_xxx”函数带3个参数，分别是变量名称，默认值，用法描述，例如

```
tf.app.flags.DEFINE_string('ckpt_path', 'model/model.ckpt-100000', '''Checkpoint directory to restore''')
```

​       定义一个名称是 "ckpt_path" 的变量，默认值是 ckpt_path = 'model/model.ckpt-100000'，描述信息表明这是一个用于保存节点信息的路径

