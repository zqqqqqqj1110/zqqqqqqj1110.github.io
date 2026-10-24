---
title: "Bow&Word2Vec"
date: 2024-12-01 20:13:41
updated: 2024-12-01 21:28:50
categories:
  - 机器学习
tags:
  - NLP
---

在对文本数据集进行预处理之后，后续办法与其他机器学习方法类似，即：特征工程+模型建立求解。本文介绍两种方法
# Bag Of Words（Bow）
所谓BOW，便是先将所有出现过的单词记录下来，接着将句子转化为文本向量，表示盖茨在文本中出现的频率或次数
## EXP
有以下两个句子：
文本1：I love machine learning
文本2：I love programming
构建词汇表：['I', 'love', 'machine', 'learning', 'programming']
然后，两个文本就可以分别表示为：
文本1的向量：[1, 1, 1, 1, 0]
文本2的向量：[1, 1, 0, 0, 1]

变体也有很多，比如说N-gram模型（主要增加语法信息）

BOW最大的特定就是简单，对于词频信息捕捉的很完美，但是缺乏词汇之间的相互作用（向量内的值没有互相联系）

## 流程
1. 数据预处理
```
def review_to_word():
	1. 常用处理
	2. 返回一个string。
```
2. BOW
```
from sklearn.feature_extraction.text import CountVectorizer
# initialize
vectorizer = CountVectorizer(
    analyzer = &#039;word&#039;,
    tokenizer = None,
    preprocessor = None,
    stop_words = None,
    max_features = 5000        # 指定特征个数
train_data_features = vectorizer.fit_transform(clean_train_reviews)
print(train_data_features)
# transfer into np form
train_data_features = train_data_features.toarray()
print(train_data_features)
```
------特征格式未定，期待有更好的结果-----
stopword不需要，因为前文已经处理过的，相较于包我更相信自己手动处理的东西
3. 测试
```
train_data_features.shape    # （总句子数量，特征词汇数量）
# 词汇表
vocab = vectorizer.get_feature_names_out()
print(vocab)
# 词频
dist = np.sum(train_data_features, axis=0)
print(dist)
```
4. 构建模型（RandomForest为例）
```
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit( train_data_features, train[&quot;sentiment&quot;] )
```

# Word2Vec
用于克服文本之间单词相互独立的问题，主要方式有两种：Skip-Gram与CBOW。前者用于预测上下文词汇，后者用于预测目标词。但是主要的问题在于只有固定的窗口（给定的上下文窗口长度）
## 结果
"dog" → [0.23, 0.51, -0.11, ..., 0.75]
"cat" → [0.21, 0.48, -0.13, ..., 0.78]
"king" → [0.45, 0.62, -0.09, ..., 0.50]
里面的数字是隐含特征，可以理解为PCA降维之后的值一样不可解释。但是通过几何意义可以解释，两个单词之间向量的相似度（余弦相似度）

## 流程
1. 数据预处理+句子拆分为段落
```
def review_to_wordlist(review, remove_stopwords = False):
    # 1. HTML
    review_test = BeautifulSoup(review).get_text()
    # 2. non-letteers
    review_only = re.sub(&quot;[^a-zA-Z]&quot;, &quot; &quot;, review_test)
    # 3. lower and split words
    review_lower = review_only.lower().split()
    # 4. Word2vec no need to delete stopword and number
    if remove_stopwords:
        stops = set(stopwords.words(&quot;english&quot;))
        words = [w for w in review_lower if not w in stops]
    # 5. return words list
    return (review_lower)
tokenizer = nltk.data.load(&#039;tokenizers/punkt/english.pickle&#039;)
def review_to_sentences(review, tokenizer, remove_stopwords = False):
    # 1. 段落-&gt;n个句子
    raw_sentences = tokenizer.tokenize(review.strip())
    # 2. 句子-&gt;单词
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) &gt; 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    return sentences
```
结束后的结果应该为list中，每个单词为单独元素
3. 处理数据
```
sentences = []
for review in train[&quot;review&quot;]:
    sentences += review_to_sentences(review, tokenizer)
for review in unlabeled_train[&quot;review&quot;]:
    sentences += review_to_sentences(review, tokenizer)
```
4. Word2Vec
```
import logging
logging.basicConfig(format=&#039;%(asctime)s : %(levelname)s : %(message)s&#039;,\
    level=logging.INFO)
# setting parameters
num_features = 300                    
min_word_count = 40                      
num_workers = 1 
context = 10
downsampling = 1e-3
# Word2Vec
from gensim.models import word2vec
model = word2vec.Word2Vec(sentences, workers=num_workers,
            vector_size=num_features, min_count = min_word_count,
            window = context, sample = downsampling)
model.init_sims(replace=True)
model_name = &quot;300features_40minwords_10context&quot;
model.save(model_name)
```
5. 查看结果(三种方式)
```
model.wv.doesnt_match(&quot;man woman child kitchen&quot;.split())
model.wv.most_similar(&quot;man&quot;)
model.wv.most_similar(&quot;awful&quot;)
```

## 与RandomFoerst相结合
### 1. 向量平均
"dog" 对应的词向量：[1, 2, 3]
"cat" 对应的词向量：[4, 5, 6]
"pet" 对应的词向量：[7, 8, 9]

那么向量的平均计算[4, 5, 6]

接着，将句子中的所以单词取平均后带入到RF中就行了

### 2. 聚类
词汇表 {dog, cat, apple, car, tree, run}
dog → [0.2, 0.4, -0.5, ...]
cat → [0.1, 0.3, -0.4, ...]
apple → [0.5, 0.7, -0.3, ...]
簇1：{dog, cat}
簇2：{apple, car, tree, run}

随后生成一个特征向量，表示文本中每个簇的频率分布，即：
文本1: [0.2, 0.5, 0.3]（20% 的词属于簇1，50% 属于簇2，30% 属于簇3）
文本2: [0.1, 0.6, 0.3]（10% 的词属于簇1，60% 属于簇2，30% 属于簇3）

将文本划为训练集，标签（label）划为结果。带入到随机森林中训练
