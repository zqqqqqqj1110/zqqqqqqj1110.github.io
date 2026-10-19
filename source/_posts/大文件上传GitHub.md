---
title: "大文件上传GitHub"
date: 2023-09-23 16:14:59 +08:00
updated: 2025-03-22 15:24:19 +08:00
categories:
  - 工具与技巧
tags:
  - GitHub
---

<p>实例：上传游戏</p>
<h2 class="wp-block-heading">步骤如下：</h2>
<p>1.路径</p>
<pre class="wp-block-code"><code>.\Git\bin</code></pre>
<p>2.上传过程</p>
<pre class="wp-block-code"><code>git init  # 初始化
git lfs install   # 安装大文件上传应用
git lfs track *   # 追踪要上传的大文件，*表示该文件内全选，也可以"文件名"或"文件.格式"
git add .gitattributes  # 添加上传文件的属性文件
git commit -m "xxx"   #属性上传文件的说明
git remote add orogin xxx   # 建立本地与GitHub仓库的链接
git push origin master    # 上传属性文件
git add *   # 添加要上传的大文件
git commit -m "xxx"   # 添加大文件的说明
git push origin master    #最后一步上传！！！</code></pre>
<p>3.（有可能没有？？？谁知道呢.jpg） 网络可能会出现问题，导致上传失败，如下图所示：</p>
<figure class="wp-block-image size-large"><img src="/wp-content/uploads/2023/09/fail_picture-1024x172.png" alt="" class="wp-image-95"/></figure>
<p>那该咋办呢？原因是网络问题 从这重新开始再来一遍</p>
<pre class="wp-block-code"><code>git add *<br>git commit -m "xxx"</code></pre>
## 非大文件上传方法
```
git clone 仓库http连接
cd PINNs  //可选，主要是看想上传什么文件
git add .    //全选当前路径下的所有文件
git add name    //指定文件
git commit -m &quot;备注&quot;
git push origin main
```
