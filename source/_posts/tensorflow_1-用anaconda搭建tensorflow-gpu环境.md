---
title: "tensorflow_1:用anaconda搭建tensorflow-gpu环境"
date: 2023-09-22 15:34:02 +08:00
updated: 2023-09-22 16:53:26 +08:00
categories:
  - 机器学习
---

## anaconda下载

懒的说了。。。网上一大堆的教程。我自己用的时候有个问题。分享一下

### anaconda navigator无法打开

1.在Anaconda Prompt中输入`anaconda-navigator `发现问题

![](/wp-content/uploads/2023/09/p1-1024x347.png)

在903行发现函数中，是url跳转位置不对。

### 解决办法

1.  改变url的值 在函数`vscode_application_data()`中添加url:`url='https://update.code.visualstudio.com/api/update/win32-x64-user/stable/version'`（1060行）
2.  给versions变量写一个静态值 打开原url链接`'https://update.code.visualstudio.com/api/update/win32-x64-user/stable/version'`，接着找到“productVersion”。最后修改903行。添加`version=['xxx']`,xxx为版本号

## 搭建虚拟环境(x为自定义)

首先查看官方文档。查看自己的显卡能搭配怎样的配置

（我自己的是3070。30系以上可通用）

- CUDA Toolkit 11.5.0
- cuDNN 8.1.x（与所选TensorFlow-GPU版本兼容）
- TensorFlow-GPU 2.5.x或更高版本

众所周知，用python必须需要一个虚拟环境。不然电脑就会变成依托答辩。以下是步骤： 首先打开Anaconda Prompt，输入以下指令：

``` wp-block-code
conda create -n tensorflow-gpu python=3.8.5   # 创建虚拟环境
activate tensorflow-gpu                       # 启动环境

conda config --add channels nvidia            # 添加软件源
conda config --add channels conda-forge       # 添加软件源

conda install cudatoolkit=x.x                 # 安装cuda
conda install cudnn=x.x.x                     # 安装cudnn
pip install tensorflow-gpu==x.x.x             # 安装gpu版本的tsflow

# 可选项
conda install scipy pillow        # 装包
conda install jupyterlab          # 装包
conda install scikit-learn        # 装包
conda install matplotlib (pip install matplotlib)       # 装包
conda install pandas（conda install openpyxl）          # 装包
conda install -c conda-forge opencv       # 装包
conda install -c conda-forge lightgbm     # 装包
pip install xgboost -i https://pypi.tuna.tsinghua.edu.cn/simple/          # 装包
pip install SimpleITK -i https://pypi.tuna.tsinghua.edu.cn/simple         # 装包
pip install pyradiomics -i https://pypi.tuna.tsinghua.edu.cn/simple/      # 装包
```

**检测是否安装成功**

``` wp-block-code
python                     # 进入python环境
import tensorflow as tf     # 导入
tf.test.is_gpu_available()  # 检验
```

当图片为下所示时，说明安装完成

![](/wp-content/uploads/2023/09/p2-1024x547.png)

1.  显卡型号
2.  显卡算力
3.  成功（bool值）

## 解决已经安装了，但是仍然显示'no module found'的error

我用的是pycharm。整了半天发现是环境没用上。。。

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

## 附：anaconda常用命令

### 本体

**1.查看conda版本**

`conda --version `

**2.查看conda的环境配置**

`conda config -- show`

**3.设置镜像**

`#设置清华镜像`  
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/`  
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/`  
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/`  
`conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/`  
`#设置bioconda`  
`conda config --add channels bioconda`  
`conda config --add channels conda-forge`  
`#设置搜索时显示通道地址`  
`conda config --set show_channel_urls yes`

**更新conda** conda update conda

**更新Anaconda**` conda update Anaconda`

### 虚拟环境(xxx为环境名字)

**创建虚拟环境** `conda create -n xxx python=3.8`

**查看有哪些环境**` conda env list`

**激活虚拟环境** `conda activate xxx`

**退出虚拟环境** `conda activate `

**删除虚拟环境**

`conda remove --name xxx --all                 # 虚拟环境及其中所安装的包都删除`  
`conda remove --name env_name package_name         # 只删除虚拟环境中的某个或者某些包`

### 包管理

**查询全部的包** `conda list`

**包的安装和更新** `conda install package_name`

**卸载包** `conda uninstall package_name`
