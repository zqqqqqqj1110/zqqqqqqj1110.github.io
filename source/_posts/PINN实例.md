---
title: "PINN实例"
date: 2023-11-05 22:59:22 +08:00
updated: 2023-11-05 23:00:18 +08:00
categories:
  - 机器学习
---

## 题目

![](/wp-content/uploads/2023/11/8dade9c57f80f461a40bb2834f0774e-1024x576.png)

## 求解过程

首先观察该式子，是一个偏微分方程再加上4个边界条件。由此应该会有五个损失函数。本题的解析解为**u=sin(pix)sin(piy)**，我们也可以将PINN求解出的u与解析解进行比较（训练的时候不要放进去，不然产生逻辑错误）~~你都有解析解了还要求来干啥QAQ~~

## 代码部分

- 导入该有的包（其实一个pytorch就够了，画图是为了显示出差异）

``` wp-block-code
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

- 自定义种子，由于神经网络是随机设置初始解，这是为了使输出的每次结果都固定

``` wp-block-code
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(888888)
```

- 设置基础参数，包括网格中点的数量，边界点数量，内点数量等等

``` wp-block-code
# 基础参数
epochs = 10000    # 训练代数
h = 100    # 画图网格密度
N = 1000    # 内点配置点数
N1 = 100    # 边界点配置点数
N2 = 1000    # PDE数据点
```

- 设置边界点和偏微分方程右边的值

``` wp-block-code
def interior(n=N):
    # 内点
    x = torch.rand(n, 1)
    y = torch.rand(n, 1)
    cond = 2 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    return x.requires_grad_(True), y.requires_grad_(True), cond
# requires_grad=True 的作用是让 backward 可以追踪这个参数并且计算它的梯度。
# 最开始定义你的输入是 requires_grad=True ，那么后续对应的输出也自动具有 requires_grad=True
#  ，如代码中的 y 和 z ，而与 z 对 x 求导无关联的 a ，其 requires_grad 仍等于 False。



def down(n=N1):
    # 边界 u(x,0)=0
    x = torch.rand(n, 1)
    y = torch.zeros_like(x)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def up(n=N1):
    # 边界 u(x,1)=0
    x = torch.rand(n, 1)
    y = torch.ones_like(x)
    cond = torch.zeros_like(x)
    return x.requires_grad_(True), y.requires_grad_(True), cond


def left(n=N1):
    # 边界 u(0,y)=0
    y = torch.rand(n, 1)
    x = torch.zeros_like(y)
    cond = torch.zeros_like(y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

def right(n=N1):
    # 边界 u(1,y)=0
    y = torch.rand(n, 1)
    x = torch.ones_like(y)
    cond = torch.zeros_like(y)
    return x.requires_grad_(True), y.requires_grad_(True), cond

def data_interior(n=N2):
    # 内点
    x = torch.rand(n, 1)
    y = torch.rand(n, 1)
    cond = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    return x.requires_grad_(True), y.requires_grad_(True), cond
```

**ps：data_interior是解析解的真实值，不要带入到模型中（损失函数）训练哦，requires_grad\_(True)是非常有必要的，不然会疯狂报错。**

- 定义神经网络层，我在此定义了一个输入层（x和y输入），三个全连接隐藏层，一个输出层（u）

``` wp-block-code
class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
```

- 建立损失函数，如同前文所述，**损失函数的数量=偏微分方程+边界条件**。只要把x，y导入到神经网络中后计算并且与前面设置的右边的值做2范数即可

``` wp-block-code
# Loss
loss = torch.nn.MSELoss()

# 递归求导
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

# 以下4个损失是PDE损失

def l_interior(u):
  # pde
  x, y, cond = interior()
  uxy = u(torch.cat([x, y], dim=1))
  return loss(-gradients(uxy, x, 2) - gradients(uxy, y, 2), cond)

def l_down(u):
    # 损失函数L4
    x, y, cond = down()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)

def l_up(u):
    # 损失函数L5
    x, y, cond = up()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)


def l_left(u):
    # 损失函数L6
    x, y, cond = left()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)

def l_right(u):
    # 损失函数L7
    x, y, cond = right()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)

# 构造数据损失
def l_data(u):
    # 损失函数L8
    x, y, cond = data_interior()
    uxy = u(torch.cat([x, y], dim=1))
    return loss(uxy, cond)
```

- 模型训练，这个就不多说了

``` wp-block-code
# Training
u = MLP()
opt = torch.optim.Adam(params=u.parameters())
# 初始化模型的参数，并将它们传入Adam函数构造出一个Adam优化器
# 这里可以通过设定 lr的数值来给定学习率
for i in range(epochs):
    opt.zero_grad()
    #  将这一轮的梯度清零，防止其影响下一轮的更新
    l = l_interior(u) + l_up(u) + l_down(u) + l_left(u)  + l_right(u)
    l.backward()
    # 反向计算出各参数的梯度
    opt.step()
    # 更新网络中的参数
    if i % 100 == 0:
        print(i)
```

- 结果对比（无可视化）

``` wp-block-code
xc = torch.linspace(0, 1, h)
xm, ym = torch.meshgrid(xc, xc)
xx = xm.reshape(-1, 1)
yy = ym.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
u_pred = u(xy)
u_real = torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)
u_error = torch.abs(u_pred-u_real)
u_pred_fig = u_pred.reshape(h,h)
u_real_fig = u_real.reshape(h,h)
u_error_fig = u_error.reshape(h,h)
print("Max abs error is: ", float(torch.max(torch.abs(u_pred - torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)))))
```

- 可视化输出结果

``` wp-block-code
# 作PINN数值解图
fig = plt.figure(1, figsize=(12, 5))  # 调整图像大小
ax = fig.add_subplot(131, projection='3d')  # 使用子图
ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_pred_fig.detach().numpy())
ax.text2D(0.5, 0.9, "PINN Solution", transform=ax.transAxes)

# 作真实解图
ax = fig.add_subplot(132, projection='3d')  # 使用子图
ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_real_fig.detach().numpy())
ax.text2D(0.5, 0.9, "Real Solution", transform=ax.transAxes)

# 绘制误差图
fig = plt.figure(2, figsize=(8, 6))  # 调整误差图的大小
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xm.detach().numpy(), ym.detach().numpy(), u_error_fig.detach().numpy())
ax.text2D(0.5, 0.9, "Absolute Error", transform=ax.transAxes)

plt.show()
```

## 最终结果

看起来还不错，误差精度到了e-2.我觉得多迭代几次应该还能更好，剩下的就摁调参了呗（吧？）==

![](/wp-content/uploads/2023/11/image-2.png)
