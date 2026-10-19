---
title: "nlp预处理常用函数"
date: 2024-11-28 18:43:48 +08:00
categories:
  - 机器学习
---

1. 删除html标签
```
from bs4 import BeautifulSoup
a = BeautifulSoup()
a.get_text()
```
2. 删除数字与标点符号
```
import re
letter_only = re.sub(&quot;[^a-zA-Z]&quot;,
                     &quot; &quot;,
                     example.get_text() )
```
3. 转化大小写
```
lower_case = letter_only.lower()
words = lower_case.split()
```
4. 删除常用词
```
import nltk
nltk.download(&quot;stopwords&quot;)  # 按需下载字库
from nltk.corpus import stopwords
words = [w for w in words if not w in stopwords.words(&quot;english&quot;)]
```
5. 转化为字符串形式
```
word = &quot; &quot;.join()
```
