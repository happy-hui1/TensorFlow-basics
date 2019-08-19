### TensorFlow 问题汇总

#### 1.变速学习率

在Tensorflow中，为解决设定学习率(learning rate)问题，提供了指数衰减法来解决。
通过tf.train.exponential_decay函数实现指数衰减学习率。
1.首先使用较大学习率(目的：为快速得到一个比较优的解);
2.然后通过迭代逐步减小学习率(目的：为使模型在训练后期更加稳定);
decayed_learning_rate=learining_rate*decay_rate^(global_step/decay_steps)
learning_rate为事先设定的初始学习率；
decay_rate为衰减系数；
decay_steps为衰减速度。

而tf.train.exponential_decay函数则可以通过staircase(默认值为False,当为True时，
（global_step/decay_steps）则被转化为整数) ,选择不同的衰减方式。

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)     #生成学习率
learning_rate：0.1；staircase=True;则每100轮训练后要乘以0.96.
通常初始学习率，衰减系数，衰减速度的设定具有主观性(即经验设置)，而损失函数下降的速度与迭代结束之后损失的大小没有必然联系，
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(....., global_step=global_step)  #使用指数衰减学习率

#### tf.contrib.layers.fully_connected

```
tf.contrib.layers.fully_connected(
    inputs,
    num_outputs,
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None
)
```

`默认创建了变量W, b

fully_connected`创建一个名为的变量`weights`，表示一个完全连接的权重矩阵，乘以它`inputs`产生一个`Tensor`隐藏单位。如果`normalizer_fn`提供了a （例如 `batch_norm`），则应用它。否则，如果`normalizer_fn`为None且`biases_initializer`提供了a，`biases`则将创建变量并添加隐藏单位。最后，如果`activation_fn`不是`None`，它也会应用于隐藏单位。

**注意：**如果`inputs`等级大于2，则`inputs`在初始矩阵乘以之前展平`weights`。

##### 参数：

- **inputs**：至少等级2的张量和最后一个维度的静态值; 即`[batch_size, depth]`，`[None, None, None, channels]`。

- **num_outputs**：整数或长整数，图层中的输出单位数。

- **activation_fn**：激活功能。默认值是ReLU功能。将其明确设置为“无”以跳过它并保持线性激活。

- **normalizer_fn**：使用标准化功能代替`biases`。如果 `normalizer_fn`提供`biases_initializer`，`biases_regularizer`则忽略并且`biases`不创建也不添加。没有规范化器功能，默认设置为“无”

- **normalizer_params**：规范化函数参数。

- **weights_initializer**：权重的初始化程序。

- **weights_regularizer**：可选的权重正则化器。

- **biases_initializer**：偏见的初始化程序。如果没有跳过偏见。

- **biases_regularizer**：偏见的可选正则化器。

- **reuse**：是否应重用图层及其变量。必须给出能够重用层范围的能力。

- **variables_collections**：所有变量的集合的可选列表或包含每个变量的不同集合列表的字典。

- **outputs_collections**：用于添加输出的集合。

- **trainable**：如果`True`还将变量添加到图表集合中 `GraphKeys.TRAINABLE_VARIABLES`（请参阅tf.Variable）。

- **scope**：variable_scope的可选范围

  参考：<https://tensorflow.google.cn/api_docs/python/tf/contrib/layers/fully_connected>

#### tensorflow里面name_scope, variable_scope等如何理解？

#### tf. get_variable()

希望一些变量重用的，所以就用到了get_variable()。它会去搜索变量名，然后没有就新建，有就直接用。

name_scope 作用于操作，variable_scope 可以通过设置reuse 标志以及初始化方式来影响域下的变量。（也就是变量名前面加一个域名，就不会出现同名错误，命名困难了）

参考：<https://www.zhihu.com/question/54513728>



##### 损失函数：tf.nn.softmax_cross_entropy_with_logits

tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
除去name参数用以指定该操作的name，与方法有关的一共两个参数：
第一个参数logits：就是神经网络最后一层的输出，如果有batch的话，它的大小就是[batchsize，num_classes]，单样本的话，大小就是num_classes

第二个参数labels：实际的标签，大小同上


具体的执行流程大概分为两步：

第一步是先对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率，对于单样本而言，输出就是一个num_classes大小的向量（[Y1，Y2,Y3...]其中Y1，Y2，Y3...分别代表了是属于该类的概率）

softmax的公式是：![img](https://img-blog.csdn.net/20161128203449282)

第二步是softmax的输出向量[Y1，Y2,Y3...]和样本的实际标签做一个交叉熵，公式如下：
![img](https://img-blog.csdn.net/20161128203840317)
其中![img](https://img-blog.csdn.net/20161128204121097)指代实际的标签中第i个的值，![img](https://img-blog.csdn.net/20161128204500600)就是`softmax的输出向量[Y1，Y2,Y3...]`中，第i个元素的值。

显而易见，预测越准确，结果的值越小（别忘了前面还有负号），最后求一个平均，得到我们想要的loss

注意！！！这个函数的返回值并不是一个数，而是一个向量，如果要求交叉熵，我们要再做一步tf.reduce_sum操作,就是对向量里面所有元素求和，最后才得到，如果求loss，则要做一步tf.reduce_mean操作，对向量求均值！

