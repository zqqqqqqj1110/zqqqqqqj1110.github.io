---
title: "GitHub基础操作"
date: 2023-11-17 18:58:16 +08:00
updated: 2023-11-17 18:59:13 +08:00
categories:
  - 工具与技巧
tags:
  - GitHub
---

## 初始化

话不多说上代码：

``` wp-block-code

git version # 你的名字
git config --global user.name "你的名字" # 初始化你的名字
git config --global user.email "你的邮箱" # 初始化你的邮箱
git init # 初始化
```

在执行git init后，该文件中会出现一个".git"名字的隐藏文件，该文件是你历史版本的记录。

![](/wp-content/uploads/2023/11/p1-1.png)

## 提交

先给代码再讲解

``` wp-block-code
git add . # 提交文件夹内所有的文件
git add xxx.xxx # 提交单个文件，注意要全称（名字.后缀）
```

这两个add操作结束后，文件此时处于暂存区中，还需要一行代码来进行正式上传，接下来有两种办法，都普及一下把?

1.  **直接使用代码（最常用的办法）**

``` wp-block-code
git commit -m "注释"
```

注意：注释最好稍微规范一点，比如可以这样写

``` wp-block-code
git commit -m "fix(test): change content"
```

test为模块或文件，后面的表示你修改了什么内容

2.  **进入到vim进行注释**

这个办法其实很阴间，因为vim使用起来个人感觉是有一点麻烦的，首先输入git commit,接着是vim内的操作：

· a 进入编辑模式

· i 进入编辑模式

· esc 退出xx模式

输入esc后，再输入:wq即可退出vim,如下图所示：

![](/wp-content/uploads/2023/11/p2-1-667x1024.png)

此时输入git log,即可看到自己是否提交成功了，如下图所示：

![](/wp-content/uploads/2023/11/p3-1-1024x559.png)

## 回滚

就一行代码：

``` wp-block-code
git reset --hard xxxxxxxxxxxxxxxxxxxxxxx #xxx为log后的版本id，上图中黄色的那些东西
```

回滚指令不止有hard，还有soft，默认的mixed模型等等，具体自己百度一下

## 分支（团队操作的精髓）

有一个大问题，当我回滚之后，我想保存为副本怎么办呢？ 答：**创建分支，之后可以切换分支查看文件**

``` wp-block-code
git branch x1 #x1为名字
git branch x2
git checkout x1 # 切换到x1分支
git merge x1 # 将x1这个分支与主合并
```

网上找了张图，差不多是这个意思

![](/wp-content/uploads/2023/11/p4-1-1024x750.png)

## 建立仓库

只有建立了GitHub仓库，才能知道到底应该传到哪。当新建一个仓库后，里面都会写出具体命令，如下图所示：

![](/wp-content/uploads/2023/11/p5-1-1024x544.png)

大致解释一下把，省的百度垃圾时间了?‍?️?‍?️?‍?️

``` wp-block-code
git branch -M main # 创建main分支，并把主分支切换为main
git remote add origin https://github.com/zqqqqqqj1110/23333333333.git # 仓库地址，连了才能知道往哪里传
git push -u origin main # 推送
```

接下来可以需要输入邮箱密码，输就完事了，懒的演示了

## 参与他人项目

1.  **点左上角的fork**

![](/wp-content/uploads/2023/11/p6-1-1024x342.png)

1.  **回到自己的仓库并将其down下来**

![](/wp-content/uploads/2023/11/p7-1.png)

接着在终端中使用代码：git clone ctrl+v \#复制粘贴地址即可

3.  **添加上游代码：**

``` wp-block-code
git remote add upstream ctrl+v # 此处为他人仓库的地址
```

4.  **添加功能**

``` wp-block-code
git checkout -b xxx # xxx是你的分支的名字（结束后合并分支，你懂的）
```

5.  **上传**

``` wp-block-code
git add
git commit -m ""
git push # 第一次push有可能失败，如果失败用下面的
git push -u origin xxx # -u为参数，接下来分别代表源和分支
```

最后点击到别人的GitHub仓库中，点击最上方的pull request(简称pr)就ok了

## merge失败问题

有可能是上游代码发生改变了，此时需要在自己终端中跟新一下最新代码，如下：

``` wp-block-code
git fetch upstream # 从上游跟新代码
git merge upstream/main # 将代码合并到自己这
```
