---
title: "Kaggle比赛：Titanic - Machine Learning from Disaster"
date: 2024-04-21 18:13:49 +08:00
updated: 2024-08-19 14:27:42 +08:00
categories:
  - 数学建模与竞赛
tags:
  - Kaggle
---

# Kaggle比赛：Titanic - Machine Learning from Disaster

链接：https://www.kaggle.com/competitions/titanic

## 成绩758/15641（2024.4.21）：

![](/wp-content/uploads/2024/04/1-7.png)

还算不错吧，虽然是学习赛但好歹也是个**5%**嘻嘻

## 代码分析及思路

### 导入包与数据集

所有的包都放在一个代码块了，个人习惯而已

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from scipy.stats import pointbiserialr
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
```

```python
data_train = pd.read_csv(&#039;train.csv&#039;)
data_test = pd.read_csv(&#039;test.csv&#039;)

data_train.info()
data_test.info()
```

导入数据，并通过info查看基本信息，describe也可以，还是个人习惯，主要是int和float不太适用于后续的树模型（xgboost）

![](/wp-content/uploads/2024/04/2-6.png)

### 数据探索

这一块主要是为了做描述性统计的，做一些基础的分析

```python
# 连续性变量Age与Fare的箱线图
fig, axes = plt.subplots(1, 2, figsize=(4, 5))

data = data_train

sns.boxplot(data[&#039;Age&#039;], orient=&#039;v&#039;, width=0.5, ax=axes[0])
axes[0].set_title(&#039;Boxplot of Age&#039;)
axes[0].set_ylabel(&#039;Age&#039;)

sns.boxplot(data[&#039;Fare&#039;], orient=&#039;v&#039;, width=0.5, ax=axes[1])
axes[1].set_title(&#039;Boxplot of Fare&#039;)
axes[1].set_ylabel(&#039;Fare&#039;)

plt.tight_layout()


# 连续性变量Age与Fare的KDE图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.kdeplot(data=data[data[&#039;Survived&#039;] == 0][&#039;Age&#039;], label=&#039;Not Survived&#039;, shade=True)
sns.kdeplot(data=data[data[&#039;Survived&#039;] == 1][&#039;Age&#039;], label=&#039;Survived&#039;, shade=True)
plt.xlabel(&#039;Age&#039;)
plt.ylabel(&#039;Density&#039;)
plt.title(&#039;Relationship between Age and Survived&#039;)
plt.legend()

plt.subplot(1, 2, 2)
sns.kdeplot(data=data[data[&#039;Survived&#039;] == 0][&#039;Fare&#039;], label=&#039;Not Survived&#039;, shade=True)
sns.kdeplot(data=data[data[&#039;Survived&#039;] == 1][&#039;Fare&#039;], label=&#039;Survived&#039;, shade=True)
plt.xlabel(&#039;Fare&#039;)
plt.ylabel(&#039;Density&#039;)
plt.title(&#039;Density Plot of Fare by Survival&#039;)
plt.legend()

plt.tight_layout()


# 分类变量与Survived的柱状图
plt.figure(figsize=(10, 5))

plt.subplot(2, 3, 1)
sns.countplot(x=&#039;Pclass&#039;, hue=&#039;Survived&#039;, data=data)
plt.xlabel(&#039;Pclass&#039;)
plt.ylabel(&#039;Count&#039;)
plt.title(&#039;Survival by Pclass&#039;)

plt.subplot(2, 3, 2)
sns.countplot(x=&#039;Sex&#039;, hue=&#039;Survived&#039;, data=data)
plt.xlabel(&#039;Sex&#039;)
plt.ylabel(&#039;Count&#039;)
plt.title(&#039;Survival by Sex&#039;)

plt.subplot(2, 3, 3)
sns.countplot(x=&#039;Embarked&#039;, hue=&#039;Survived&#039;, data=data)
plt.xlabel(&#039;Embarked&#039;)
plt.ylabel(&#039;Count&#039;)
plt.title(&#039;Survival by Embarked&#039;)

plt.subplot(2, 3, 4)
sns.countplot(x=&#039;SibSp&#039;, hue=&#039;Survived&#039;, data=data)
plt.xlabel(&#039;SibSp&#039;)
plt.ylabel(&#039;Count&#039;)
plt.title(&#039;Survival by SibSp&#039;)

plt.subplot(2, 3, 5)
sns.countplot(x=&#039;Parch&#039;, hue=&#039;Survived&#039;, data=data)
plt.xlabel(&#039;Parch&#039;)
plt.ylabel(&#039;Count&#039;)
plt.title(&#039;Survival by Parch&#039;)

plt.subplot(2, 3, 6)
sns.countplot(x=&#039;Cabin&#039;, hue=&#039;Survived&#039;, data=data)
plt.xlabel(&#039;Cabin&#039;)
plt.ylabel(&#039;Count&#039;)
plt.title(&#039;Survival by Cabin&#039;)

plt.tight_layout()


plt.show()
```

Survived为二分类变量。需要注意的是，数值型变量分布可以有箱线图表出；对于分类变量可以有KDE表出，如下：

![](/wp-content/uploads/2024/04/3-3.png)
![](/wp-content/uploads/2024/04/4-2.png)
![](/wp-content/uploads/2024/04/5-1.png)

结论可以由上看出，不多说了

最后再查看样本分布数量

```python
# 查看样本量是否平衡
survived_counts = data[&#039;Survived&#039;].value_counts()
print(survived_counts)
```

![](/wp-content/uploads/2024/04/6.png)

其实是不怎么平衡的，最好在这里使用SMOKE或上采样等方法，这应该是一个上分点

### 数据挖掘

#### 数据预处理

**1. 缺失值处理**

train中缺少Age，Cabin与Embraked，test缺少Age，Cabin与Fare。Age，Cabin是大头，其他两个只缺了5个以下所以直接用众数填补(也可以删，但是test需要提交所以不能删)

在此使用模型预测Age（单一决策树）

```python
# 定义一个函数来填补缺失值
def fill_missing_values(df):
    # 用平均值填补Fare的缺失值
    fare_imputer = SimpleImputer(strategy=&#039;mean&#039;)
    df[&#039;Fare&#039;] = fare_imputer.fit_transform(df[[&#039;Fare&#039;]])
    
    # 用众数填补Embarked的缺失值
    embarked_imputer = SimpleImputer(strategy=&#039;most_frequent&#039;)
    df[&#039;Embarked&#039;] = embarked_imputer.fit_transform(df[[&#039;Embarked&#039;]])
    
    # 提取有年龄信息和缺失年龄信息的数据
    known_age = df[df[&#039;Age&#039;].notna()]
    unknown_age = df[df[&#039;Age&#039;].isna()]
    
    # 特征和目标变量
    X_train = known_age.drop([&#039;Age&#039;], axis=1)
    y_train = known_age[&#039;Age&#039;]
    X_test = unknown_age.drop([&#039;Age&#039;], axis=1)
    
    # 训练决策树模型
    dt_regressor = DecisionTreeRegressor(random_state=42)
    dt_regressor.fit(X_train, y_train)
    
    # 预测缺失值
    predicted_age = dt_regressor.predict(X_test)
    
    # 创建一个新的DataFrame来保存填充后的数据
    filled_df = df.copy()
    
    # 将预测值填充到新的DataFrame中
    filled_df.loc[filled_df[&#039;Age&#039;].isna(), &#039;Age&#039;] = predicted_age
    
    return filled_df
```

如果使用xgboost的话，是不怎么需要处理异常值与缺失值的，但是后文还有其他工程，因此先处理。这种办法的问题就在于需要让自己模型预测值的准确率高，不然就是瞎填补

对于Cabin来说，由于缺失值太多，这里处理为：有值的为1，没值的为0

```python
data[&#039;Cabin&#039;] = data[&#039;Cabin&#039;].notna().astype(int)
test_data[&#039;Cabin&#039;] = test_data[&#039;Cabin&#039;].notna().astype(int)
```

注意一个点，**修改的时候必须训练集与测试集一起修改**。不然模型是预测不了特征个数不对的对应文件的。后文不再赘述

**2. 异常值处理**

懒的搞了，相信xgboost

**3. SibSp与Parch**

在此选择相加，但是普通的相加减在树模型中是无效的（加不加对模型的精度都没有提升）。

```python
data[&#039;Numbers&#039;] = data[&#039;SibSp&#039;] + data[&#039;Parch&#039;]
data = data.drop([&#039;SibSp&#039;, &#039;Parch&#039;], axis=1)

test_data[&#039;Numbers&#039;] = test_data[&#039;SibSp&#039;] + test_data[&#039;Parch&#039;]
test_data = test_data.drop([&#039;SibSp&#039;, &#039;Parch&#039;], axis=1)
```

**4. Embarked**

在此将其转为1，2，3变量

```python
data[&#039;Embarked&#039;] = data[&#039;Embarked&#039;].replace({&#039;S&#039;:1, &#039;C&#039;:2, &#039;Q&#039;:3})

test_data[&#039;Embarked&#039;] = test_data[&#039;Embarked&#039;].replace({&#039;S&#039;:1, &#039;C&#039;:2, &#039;Q&#039;:3})
```

**5. Ticket**

我不会处理，对于文本类的真的不是很熟悉，在此我直接删掉这个特征了（别学我）

```python
data = data_train.drop([&#039;Ticket&#039;], axis=1)

test_data = data_test.drop([&#039;Ticket&#039;], axis=1)
```



#### 特征工程

**1. 对Name处理**

在此借用讨论区大神的办法，我直接删掉，但是大神说通过前缀可以看出这个人的社会地位，因此做整合

```python
# 训练集
data[&#039;Title&#039;] = data[&#039;Name&#039;].apply(lambda x:x.split(&#039;,&#039;)[1].split(&#039;.&#039;)[0].strip())
data[&#039;Title&#039;].replace([&#039;Capt&#039;, &#039;Col&#039;, &#039;Major&#039;, &#039;Dr&#039;, &#039;Rev&#039;],&#039;Officer&#039;, inplace=True)
data[&#039;Title&#039;].replace([&#039;Don&#039;, &#039;Sir&#039;, &#039;the Countess&#039;, &#039;Dona&#039;, &#039;Lady&#039;], &#039;Royalty&#039;, inplace=True)
data[&#039;Title&#039;].replace([&#039;Mme&#039;, &#039;Ms&#039;, &#039;Mrs&#039;],&#039;Mrs&#039;, inplace=True)
data[&#039;Title&#039;].replace([&#039;Mlle&#039;, &#039;Miss&#039;], &#039;Miss&#039;, inplace=True)
data[&#039;Title&#039;].replace([&#039;Master&#039;,&#039;Jonkheer&#039;],&#039;Master&#039;, inplace=True)
data[&#039;Title&#039;].replace([&#039;Mr&#039;], &#039;Mr&#039;, inplace=True)
data = pd.get_dummies(data, columns=[&#039;Title&#039;], prefix=&#039;Title&#039;)
data = data.drop([&#039;Name&#039;], axis=1)

# 测试集
test_data[&#039;Title&#039;] = test_data[&#039;Name&#039;].apply(lambda x:x.split(&#039;,&#039;)[1].split(&#039;.&#039;)[0].strip())
test_data[&#039;Title&#039;].replace([&#039;Capt&#039;, &#039;Col&#039;, &#039;Major&#039;, &#039;Dr&#039;, &#039;Rev&#039;],&#039;Officer&#039;, inplace=True)
test_data[&#039;Title&#039;].replace([&#039;Don&#039;, &#039;Sir&#039;, &#039;the Countess&#039;, &#039;Dona&#039;, &#039;Lady&#039;], &#039;Royalty&#039;, inplace=True)
test_data[&#039;Title&#039;].replace([&#039;Mme&#039;, &#039;Ms&#039;, &#039;Mrs&#039;],&#039;Mrs&#039;, inplace=True)
test_data[&#039;Title&#039;].replace([&#039;Mlle&#039;, &#039;Miss&#039;], &#039;Miss&#039;, inplace=True)
test_data[&#039;Title&#039;].replace([&#039;Master&#039;,&#039;Jonkheer&#039;],&#039;Master&#039;, inplace=True)
test_data[&#039;Title&#039;].replace([&#039;Mr&#039;], &#039;Mr&#039;, inplace=True)
test_data = pd.get_dummies(test_data, columns=[&#039;Title&#039;], prefix=&#039;Title&#039;)
test_data = test_data.drop([&#039;Name&#039;], axis=1)
```

**2. 分箱操作**

将Age和Numbers进行分箱，模型可以捕捉其中的非线性关系（需要先画qq图观察）；Age的异常值过多，分区间后可以使数据更加稠密，降低异常值的影响

首先观察原始分布，为自定义边界做准备

```python
plt.figure(figsize=(8, 6))
sns.kdeplot(data=data[data[&#039;Survived&#039;] == 0][&#039;Age&#039;], color=&#039;blue&#039;, label=&#039;Not Survived&#039;, shade=True)
sns.kdeplot(data=data[data[&#039;Survived&#039;] == 1][&#039;Age&#039;], color=&#039;orange&#039;, label=&#039;Survived&#039;, shade=True)
plt.title(&#039;Density Distribution of Fare by Survival&#039;)
plt.xlabel(&#039;Age&#039;)
plt.ylabel(&#039;Density&#039;)
plt.legend()
plt.grid(True)
plt.show()


cross_tab = pd.crosstab(data[&#039;Age&#039;], data[&#039;Survived&#039;])

cross_tab
```

如下图所示结果，Numbers也是一样的，不做演示了

![](/wp-content/uploads/2024/04/7.png)

![](/wp-content/uploads/2024/04/8.png)

这样就可以明确分组边界了

```python
# 定义分组边界
bins = [0, 5, 18, 30, 40, 50, 80]
labels = [&#039;Age_0_5&#039;, &#039;Age_5_18&#039;, &#039;Age_18_30&#039;, &#039;Age_30_40&#039;, &#039;Age_40_50&#039;, &#039;Age_50_80&#039;]

# 对年龄进行分组
data[&#039;Age_Group&#039;] = pd.cut(data[&#039;Age&#039;], bins=bins, labels=labels)
data = pd.get_dummies(data, columns=[&#039;Age_Group&#039;], prefix=&#039;Age_Group&#039;)
data = data.drop([&#039;Age&#039;], axis=1)

test_data[&#039;Age_Group&#039;] = pd.cut(test_data[&#039;Age&#039;], bins=bins, labels=labels)
test_data = pd.get_dummies(test_data, columns=[&#039;Age_Group&#039;], prefix=&#039;Age_Group&#039;)
test_data = test_data.drop([&#039;Age&#039;], axis=1)

# 定义分组边界
bins = [0, 1, 2, 3, 10]
labels = [&#039;Age_0_1&#039;, &#039;Age_1_2&#039;, &#039;Age_2_3&#039;, &#039;Age_3_10&#039;]

# 对人数进行分组
data[&#039;Numbers_Group&#039;] = pd.cut(data[&#039;Numbers&#039;], bins=bins, labels=labels)
data = pd.get_dummies(data, columns=[&#039;Numbers_Group&#039;], prefix=&#039;Numbers&#039;)
data = data.drop([&#039;Numbers&#039;], axis=1)
test_data[&#039;Numbers_Group&#039;] = pd.cut(test_data[&#039;Numbers&#039;], bins=bins, labels=labels)
test_data = pd.get_dummies(test_data, columns=[&#039;Numbers_Group&#039;], prefix=&#039;Numbers&#039;)
test_data = test_data.drop([&#039;Numbers&#039;], axis=1)
```



分组边界都是自己画描述性统计自己分的。其实还有一个Fare作为连续性变量也需要分，但是我分出来之后发现精度反而变滴了，期待有人可以帮我解决疑惑

**3. 哑变量处理**

将Sex，Embarked分类特征转化为哑变量，有一个疑问是：Embarked按理说应该做哑变量处理，但是我转为1，2，3这样int型反而精更好。不明白为啥

```python
data = pd.get_dummies(data, columns=[&#039;Sex&#039;], prefix=&#039;Sex&#039;)
data[&#039;Embarked&#039;] = data[&#039;Embarked&#039;].replace({&#039;S&#039;:1, &#039;C&#039;:2, &#039;Q&#039;:3})

test_data = pd.get_dummies(test_data, columns=[&#039;Sex&#039;], prefix=&#039;Sex&#039;)
test_data[&#039;Embarked&#039;] = test_data[&#039;Embarked&#039;].replace({&#039;S&#039;:1, &#039;C&#039;:2, &#039;Q&#039;:3})
```

### 模型

本文使用XGBoost，这个世界上最tmd好用的树模型！！！

**1. 预测与保存**

```python
# 将数据拆分为特征和目标变量
X = data.drop([&#039;Survived&#039;], axis=1)
y = data[&#039;Survived&#039;]

# 将数据拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.dtypes)

# 训练XGBoost模型
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 使用训练好的模型进行预测
test_predictions = model.predict(test_data)

# 创建包含PassengerId和Survived字段的DataFrame
results = pd.DataFrame({&#039;PassengerId&#039;: test_data[&#039;PassengerId&#039;], &#039;Survived&#039;: test_predictions})

# 将结果保存为CSV文件
results.to_csv(&#039;result.csv&#039;, index=False)
```

其实还有模型调参的部分，但是我调参之后精度反而下降了不少，不懂这是啥原因，可能是做的过于简单了

**2. 模型评估**

F1，精确率，准确率，召回率和混淆矩阵进行评估

```python
# 评估模型
accuracy = accuracy_score(y_test, y_pred)
print(&quot;准确率:&quot;, accuracy)
conf_matrix = confusion_matrix(y_test, y_pred)
print(&quot;混淆矩阵:&quot;)
print(conf_matrix)
precision = precision_score(y_test, y_pred)
print(&quot;精确率:&quot;, precision)
recall = recall_score(y_test, y_pred)
print(&quot;召回率:&quot;, recall)
f1 = f1_score(y_test, y_pred)
print(&quot;F1指标:&quot;, f1)
```

还有一个点，针对树模型可以画出基尼指数做判断，代码与运行结果如下：

```python
# 获取特征重要性得分
feature_importance = model.feature_importances_

# 创建特征重要性的DataFrame
feature_importance_df = pd.DataFrame({&#039;Feature&#039;: X_train.columns, &#039;Importance&#039;: feature_importance})

# 按重要性降序排序
feature_importance_df = feature_importance_df.sort_values(by=&#039;Importance&#039;, ascending=False)

# 可视化特征重要性
plt.figure(figsize=(10, 6))
sns.barplot(x=&#039;Importance&#039;, y=&#039;Feature&#039;, data=feature_importance_df)
plt.title(&#039;Feature Importance&#039;)
plt.xlabel(&#039;Importance Score&#039;)
plt.ylabel(&#039;Features&#039;)
plt.show()
```

![](/wp-content/uploads/2024/04/9.png)

### 交叉验证

我只能说，你永远可以相信你在本地交叉验证的结果！！！

我自己的精确度为0.83240，提交后的为0.80143，有点过拟合吧？但是交叉验证出来的值非常对，达到了惊人的0.80035。

```python
from sklearn.model_selection import cross_val_score

# 将数据拆分为特征和目标变量
X = data.drop([&#039;Survived&#039;], axis=1)
y = data[&#039;Survived&#039;]

# 训练XGBoost模型并进行交叉验证
model = xgb.XGBClassifier()
scores = cross_val_score(model, X, y, cv=5)  # 使用5折交叉验证

# 输出交叉验证得分
print(&quot;交叉验证得分:&quot;, scores)
print(&quot;平均交叉验证得分:&quot;, scores.mean())
```

## 总结

1. XGBoost真的很好用
2. 分箱操作很重要
3. 千万不要删特征，要做特征工程的转换
4. 最后一定要用到交叉分析

## 改进与疑惑

### 上分点

1. 模型的调参
2. Fare的分组
3. 对Ticket的处理
4. 异常值分类处理

### 疑惑

1. 按理说Embarked应该转为哑变量啊，但是数值型变量反而提升了进度
2. 模型调参不对，下次再试试看
3. Fare的分箱处理，我没处理好不知道是为啥
4. PassengerID按理说没啥用，但是去掉之后又下降了，不理解
