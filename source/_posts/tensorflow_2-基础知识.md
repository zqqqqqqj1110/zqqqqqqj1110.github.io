---
title: "tensorflow_2:基础知识"
date: 2023-09-22 15:43:25 +08:00
updated: 2023-09-22 16:53:21 +08:00
categories:
  - 机器学习
---

# 数据类型

## 数值类型

· 标量，例如1，2，3，4，shape=\[\]

· 向量，例如\[1.2\], \[1,2,3.4\],维度为1，shape=\[n\]

· 矩阵，shape=\[n,m\]

· 张量，dim\>2

### **1. 张量**

所有维度数dim \> 2的数组统称为张量。张量的每个维度也作轴(Axis)， 一般维度代表了具体的物理含义，比如Shape 为\[2,32,32,3\]的张量共有 4 维，如果表示图片数据的话，每个维度/轴代表的含义分别是图片数量、图片高度、图片宽度、图片通道数，其中 2 代表了 2 张图片，32 代表了高、宽均为 32，3 代表了 RGB 共 3 个通道。张量的维度数以及每个维度所代表的具体物理含义需要由用户自行定义。在 TensorFlow 中间，为了表达方便，一般把标量、向量、矩阵也统称为张量，不作区分，需要根据张量的维度数或形状自行判断

``` wp-block-code
x = tf.constant(1.2)
x - tf.constant([1, 2., 3.3])
```

代表含义：

``` wp-block-code
<tf.Tensor: id=165, shape=(3,), dtype=float32, numpy=array([1. , 2. , 3.3], dtype=float32)>
```

其中，shape 表示张量的形状，dtype 表示张量的数 值精度，张量 numpy()方法可以返回 Numpy.array 类型的数据 导为numpy数组格式：`x.nmupy()`

### **2. 向量**

创建向量

``` wp-block-code
a = tf.constant([[1, 2], [3, 4]])
```

输出形式

``` wp-block-code
tf.Tensor(
[[1 2]
[3 4]], shape=(2, 2), dtype=int32)
```

### **3. 字符串**

创建字符串

``` wp-block-code
a = tf.constant('hello,world')
```

小写化字符串

``` wp-block-code
a = tf.constant('hello,world')tf.strings.lower(a)
```

### **4. 布尔类型**

普通布尔类型

``` wp-block-code
a = tf.constant(True)
```

向量型布尔

``` wp-block-code
a = tf.constant([True, False])
```

------------------------------------------------------------------------

*注意：Tensorflow与python的布尔类型不等价,例如*

``` wp-block-code
a = tf.constant(True) # 创建 TF 布尔张量
a is True # TF 布尔类型张量与 python 布尔类型比较
a == True # 仅数值比较
```

*此时输出为False*

## 数值精度

### **1. 定义精度**

使用dtype表示，例如

``` wp-block-code
tf.constant(12345.6789, dtype=tf.xxx)    
```

此处，xxx为： tf.int16、tf.int32、tf.int64、tf.float16、tf.float32、 tf.float64 等，其中 tf.float64 即为 tf.double

### **2. 读取与转换精度**

着重使用函数`tf.cast()` 举例：

``` wp-block-code
print('before:',a.dtype)             # 读取原有张量的数值精度
if a.dtype != tf.float32:             # 如果精度不符合要求，则进行转换
    a = tf.cast(a,tf.float32)             # tf.cast 函数可以完成精度转换
print('after :',a.dtype)             # 打印转换后的精度
```

------------------------------------------------------------------------

*注意！* 高精度转为低精度时，可能会发生数据溢出。bool与整形转化与python中一样，即： True = 1； False = 0

## 待优化张量

由于梯度运算需要大量计算资源与自动更新参数，因此该类型张量用于查询梯度信息，函数为tf.Variable 通过该函数可以将普通张量转换为待优化张量

``` wp-block-code
a = tf.constant([-1, 0, 1, 2])
aa = tf.Variable(a)
aa.name, aa.trainable
```

也可以直接使用该函数进行创建

``` wp-block-code
a = tf.Variable([1, 2], [3, 4])
```

普通张量也可以通过`GradientTape.watch()`进行跟踪

## 创建张量

### **1. 从数组，列表对象创建**

使用tf.convert_to_tensor 函数可以创建新 Tensor，并保存在 Python List 对象或者Numpy Array 对象中的数据导入到新 Tensor 中

``` wp-block-code
tf.convert_to_tensor([1, 2.])                       # 从列表创建张量
tf.convert_to_tensor(np.array([[1,2.],[3,4]]))     # 从数组中创建张量
```

### **2. 创建全1或0张量**

``` wp-block-code
主要函数为tf.zeros()和tf.ones(),可以指定维度tf.zeros([])
tf.ones([])
tf.ones([3, 2])     # 3行2列，zeros同理
```

也可以通过 tf.zeros_like, tf.ones_like 可以方便地新建与某个张量shape一致且内容为全0或全1的张量。

``` wp-block-code
a = tf.ones([2, 3])
tf.zeros_like(a)   # ones同理
```

### **3. 自定义数值张量**

有时需要全部初始化为某个自定义数值的张量，比如将张量的数值全部初始化为−1等。此时可以使用tf.fill(shape, value).例如：

``` wp-block-code
tf.fill([2, 2], -1)        # 创建-1 的标量,为2行2列
```

### **4. 创建已知分布的张量**

正态分布(Normal Distribution，或 Gaussian Distribution)和均匀分布(UniformDistribution)是最常见的分布之一，创建采样自这 2 种分布的张量非常有用。以下是一些常见分布。具体可查tensorflow的操作文档

``` wp-block-code
tf.random.normal([2,2]) # 创建标准正态分布的张量
tf.random.normal([2,2], mean=1,stddev=2) # 创建正态分布的张量,均值为1，标准差为2
tf.random.uniform([2,2]) # 创建采样自[0,1)均匀分布的矩阵
tf.random.uniform([2,2],maxval=10) # 创建采样自[0,10)均匀分布的矩阵
tf.random.uniform([2,2],maxval=100,dtype=tf.int32) # 创建采样自[0,100)均匀分布的整型矩阵
```

### **5. 创建序列**

使用tf.range()实现创建一段连续的整形序列。如

``` wp-block-code
tf.range(10)         # 0~10，不包含 10
tf.range(10,delta=2) # 创建 0~10，步长为 2 的整形序列
```

一般来说，tf.range(start, limit, delta=1)可以创建\[start,limit)，步长为 delta 的序列

## 张量的典型应用（稍微了解一下就好）

### **1. 标量**

一般用于误差值的表示，指标的表示。如准确度(Accuracy，简称 acc)，精度(Precision)和召回率(Recall)等

### **2. 向量**

例如偏置张量b（bias)

![](/wp-content/uploads/2023/09/p1-1.png)

### **3. 矩阵**

全连接层的批量输入张量?的形状为\[?, ?in\]，其中?表示输入样本的个数，即 Batch Size，?in表示输入特征的长度。例如特征长度为 4，一共包含 2 个样本的输入可以表示为矩阵

``` wp-block-code
x = tf.random.normal([2,4]) # 2 个样本，特征长度为 4 的张量
```

### **4. 三维张量**

一种典型应用为序列信号。格式为： X = \[?, sequence len, feature len\] 其中，?表示序列信号的数量，sequence len 表示序列信号在时间维度上的采样点数或步数，feature len 表示每个点的特征长度
