---
title: "tensorflow_4:合并，分割等基础操作"
date: 2023-09-22 16:11:28 +08:00
updated: 2023-09-22 16:53:07 +08:00
categories:
  - 深度学习
tags:
  - TensorFlow
---

### **拼接**

合并是指将多个张量在某个维度上合并为一个张量。以某学校班级成绩册数据为例，设张量?保存了某学校 1~4 号班级的成绩册，每个班级 35 个学生，共 8 门科目成绩，则张量?的 shape 为：\[4,35,8\]；同样的方式，张量?保存了其它 6 个班级的成绩册，shape \[6,35,8\]。通过合并这 2 份成绩册，便可得到学校所有班级的成绩册，记为张量?，shape 应为\[10,35,8\]，其中，10 代表 10 个班级，35 代表 35 个学生，8 代表 8 门科目 合并代码如下：

``` wp-block-code
a = tf.random.normal([4, 35, 8])
b = tf.random.normal([6, 35, 8])
tf.concat([a, b], axis=0)           # 第一块为指定的张量合并，axis为指定第n维进行合并
```

### **堆叠**

?保存了某个班级的成绩册，shape 为\[35,8\]，张量?保存了另一个班级的成绩册，shape 为\[35,8\]。合并这 2 个班级的数据时，则需要创建一个新维度，定义为班级维度，新维度可以选择放置在任意位置，一般根据大小维度的经验法则，将较大概念的班级维度放置在学生维度之前，则合并后的张量的新 shape 应为

``` wp-block-code
[2,35,8]a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])
tf.stack([a, b], axis=0)
```

*不能创建维度，只能在已有的维度上进行堆叠*

### **分割**

得到整个学校的成绩册张量，shape 为\[10,35,8\]，现在需要将数据在班级维度切割为10个张量，每个张量保存了对应班级的成绩册数据

``` wp-block-code
x = tf.random.normal([10, 35, 8])
result = tf.split(x, num_or_size_splits=10, axis=0)
len(result)
for i in range(len(result)):           # 打印出10份分割后的张量
    print(result[i])
```

## 数据统计

### **向量范数**

向量范数(Vector Norm)是表征向量“长度”的一种度量方法，它可以推广到张量上。在神经网络中，常用来表示张量的权值大小，梯度大小等

- L1范数，定义为向量?的所有元素绝对值之和`tf.norm(x, ord=1)`

![](/wp-content/uploads/2023/09/p1-2.png)

L2 范数，定义为向量?的所有元素的平方和，再开根号`tf.norm(x, ord=2)`

![](/wp-content/uploads/2023/09/p2-2.png)

∞ −范数，定义为向量?的所有元素绝对值的最大值`tf.norm(x, ord=np.inf)`

![](/wp-content/uploads/2023/09/p3-1.png)

### **最值，均值，和**

敲个reduce\_，什么都出来了。具体看自己想要什么

![](/wp-content/uploads/2023/09/p4-1.png)

例子：

``` wp-block-code
x = tf.random.normal([4, 10])
tf.reduce_max(x, axis=1)           # 在指定的维度上，看4的最大值，所以说一共有四个
```

## 张量比较

将预测结果和真实标签比较，统计比较结果中正确的数量来计算准确率 常用函数比较：

![](/wp-content/uploads/2023/09/p5-1-1024x592.png)

## 数据限幅

通过函数实现数据的下限幅与上限幅

``` wp-block-code
x = tf.range(9)
tf.maximum(x, 2)   # 下限幅到2
```

![](/wp-content/uploads/2023/09/p6.png)

使用函数tf.clip_by_value实现上下限幅

``` wp-block-code
x = tf.range(9)
tf.clip_by_value(x,2,7) # 限幅为 2~7
```
