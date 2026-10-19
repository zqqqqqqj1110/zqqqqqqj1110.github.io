---
title: "tensorflow_5:高级操作"
date: 2023-09-22 16:21:49 +08:00
updated: 2023-09-22 16:53:00 +08:00
categories:
  - 深度学习
tags:
  - TensorFlow
---

**常用， 但是稍复杂的功能函数**

### 1. tf.gather

该函数可以根据索引号收集数据 假设共有4个班级，每个班级35个学生，8门科目，保存成绩册的张量shape为\[4,35,8\].

``` wp-block-code
x = tf.random.uniform([4, 35, 8], maxval=100)
```

收集第 1~2 个班级的成绩册，可以给定需要收集班级的索引号：\[0,1\]，并指定班级的维度 axis=0，通过 tf.gather 函数收集数据

``` wp-block-code
tf.gather(x, [0, 1], axis=0)
```

抽查所有班级的第 1、4、9、12、13、27 号同学的成绩数据

``` wp-block-code
tf.gather(x,[0,3,8,11,12,26],axis=1)
```

要收集所有同学的第3和第5门科目的成绩:

``` wp-block-code
tf.gather(x,[2,4],axis=2)    # 第3,5科目的成绩
```

### 2. tf.gather_nd

该函数可以实现采样多个点（多维） 抽查第2个班级的第2个同学的所有科目，第3个班级的第3个同学的所有科目，第4个班级的第4个同学的所有科目

``` wp-block-code
tf.gather_nd(x,[[1,1],[2,2],[3,3]])
```

### 3. tf.boolean_mask(感觉可能没啥用?)

以掩码方式进行数据提取,采样第 1 和第 4 个班级的数据

``` wp-block-code
tf.boolean_mask(x, mask=[True, False, True, False], axis=0)
```

### 4. tf.where(说白了就是筛选，和sql一个东西)

通过tf.where(cond, a, b)操作可以根据cond条件的真假从参数?或?中读取数据，条件判定规则如下

![](/wp-content/uploads/2023/09/p1-3.png)

![](/wp-content/uploads/2023/09/p2-3-1024x283.png)

*当参数a=b=None时，即a和b参数不指定，tf.where会返回cond张量中所有True的元素的索引坐标*

### 5. scatter_nd

该函数可以高效地刷新张量的部分数据，但是这个函数只能在**全0的白板张量上面执行刷新操作**

原理：

![](/wp-content/uploads/2023/09/p3-2-1024x223.png)

参数为：`tf.scatter_nd(indices, updates, shape)`

需要刷新的数据索引号通过indices表示，新数据为updates。根据indices给出的索引位置将updates中新的数据依次写入白板中，并返回更新后的结果张量 实例：

indices = tf.constant(\[\[4\], \[3\], \[1\], \[7\]\])     \# 构造需要刷新数据的位置参数,即为4、3、1和7号位置  
print(indices)  
updates = tf.constant(\[4.4, 3.3, 1.1, 7.7\])     \# 构造需要写入的数据,4号位写入 4.4, 3号位写入3.3...以此类推  
print(updates)  
tf.scatter_nd(indices, updates, \[8\])           \# 在长度为8的全0向量上根据indices写入updates数据

3D张量跟新示意图

![](/wp-content/uploads/2023/09/p4-2-1024x379.png)

实例：

``` wp-block-code
indices = tf.constant([[1], [3]])
print(indices)
updates = tf.constant([
  [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
  [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]
])
print(updates)
tf.scatter_nd(indices, updates, [4, 4, 4])
```

### 6. meshgrid(画图的，你懂的必须学)

生成二维网格的采样点坐标，方便可视化等应用场合。考虑2个自变量x和y的Sinc函数表达式为

![](/wp-content/uploads/2023/09/p5-2.png)

``` wp-block-code
x = tf.linspace(-8., 8, 100)       # 设置 x 轴的采样点
y = tf.linspace(-8., 8, 100)       # 设置 y 轴的采样点
x, y = tf.meshgrid(x, y)           # 生成网格点，并内部拆分后返回
x.shape, y.shape                   # 打印拆分后的所有点的 x,y 坐标张量 shape
z = tf.sqrt(x**2+y**2)
z = tf.sin(z)/z                     # sinc 函数实现
​
import matplotlib
from matplotlib import pyplot as plt
# 导入 3D 坐标轴支持
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)                         # 设置 3D 坐标轴
# 根据网格点绘制 sinc 函数 3D 曲面
ax.contour3D(x.numpy(), y.numpy(), z.numpy(), 50)
plt.show()
```

### 批训练（高贵的gpu）

``` wp-block-code
train_db = train_db.batch(128) # 设置批训练，batch size 为 128
```

### 预处理（高贵的自动化）

``` wp-block-code
train_db = train_db.map(preprocess)
```
