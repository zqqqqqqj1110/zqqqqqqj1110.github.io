---
title: "DL之PINN"
date: 2023-11-05 16:44:13 +08:00
updated: 2023-11-05 23:03:01 +08:00
categories:
  - 机器学习
---

## PINN是什么？

本质上依然是神经网络，但是又把**物理方程**作为限制加入神经网络中使训练的结果满足物理规律。通过把物理方程的迭代前后的差值加到神经网络的**损失函数**里面去，让物理方程也“参与”到了训练过程。这样，神经网络在训练迭代时候优化的不仅仅是网络自己的损失函数，还有物理方程每次迭代的差，使得最后训练出来的结果就满足物理规律。相对于其他普通的神经网络，PINN的形状为：

![](/wp-content/uploads/2023/11/image-1024x736.png)

前部分和一般的神经网络相同，在输出层输出u之后还需要对u进行求偏导（按照物理公式改变）。PINN的损失函数是由n个等式构成的（各物理公式转化为损失值，再进行相加）。

## 代码部分

PINN的大体框架代码为：

- 导入需要的包，这里使用tensorflow2.0作为例子（pytorch也可以，只是代码会有一些不同而已）

``` wp-block-code
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

- 定义神经网络层（我的理解是，和普通的神经网络类似）

``` wp-block-code
# 定义 PINN 模型
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(50, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
```

- 定义损失函数，这里的损失函数就是方程或公式，这里以Laplace方程为例

``` wp-block-code
# 定义损失函数
def loss_fn(model, boundary_data, generated_data):
    boundary_loss = tf.reduce_mean(tf.square(model(boundary_data) - boundary_condition(boundary_data)))
    generated_loss = tf.reduce_mean(tf.square(model(generated_data)))
    return boundary_loss + generated_loss
```

- 生成数据，由于PINN是没有训练数据的，因此在此生成数据（按照xy的范围生成）

``` wp-block-code
# 生成边界数据
boundary_data = np.random.rand(100, 3)  # 100个随机边界点
boundary_data = tf.constant(boundary_data, dtype=tf.float32)

# 生成自适应数据
generated_data = np.random.rand(200, 3)  # 200个随机数据点
generated_data = tf.constant(generated_data, dtype=tf.float32)
```

- 做完这些操作之后，接下来就是训练模型，这里就不过多解释了

``` wp-block-code
# 定义优化器
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# 训练 PINN 模型
epochs = 5000

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, boundary_data, generated_data)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")
```

- 最后将其可视化，查看结果

``` wp-block-code
# 三维可视化
x = np.linspace(0, 1, 50)
y = np.linspace(0, 1, 50)
z = np.linspace(0, 1, 50)
X, Y, Z = np.meshgrid(x, y, z)
input_data = np.array(list(zip(X.ravel(), Y.ravel(), Z.ravel())))
input_data = tf.constant(input_data, dtype=tf.float32)
output = model(input_data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c=output.numpy().reshape(X.shape), cmap='viridis')
plt.show()
```

## 总结：

PINN的本质其实就是神经网络，不同点就是将物理公式内嵌入了该网络之中。具体表现就是自定义损失函数，使其与公式关联（再次可以通过2范数等进行约束）
