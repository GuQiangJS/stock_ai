# 激活层和损失函数的选择

source:[https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/]()

|问题类型|激活函数|损失函数|示例|
|---|---|---|---|
|Binary classification<br/>二元分类任务|sigmoid|binary_crossentropy|[Dog vs cat](#二元分类任务 - Dog VS Cat),Sentiemnt analysis(pos/neg)|
|Multi-class, single-label classification<br/>多级单标签分类|softmax|categorical_crossentropy|[MNIST](#多级单标签分类 - MNIST)|
|Multi-class, multi-label classification<br/>多级，多标签分类|sigmoid|binary_crossentropy|[新闻标签分类](#多级，多标签分类 - 新闻标签分类)|
|Regression to arbitrary values<br/>回归到任意值|None|mse|[Boston住房价格预测](#回归任意值 - Boston住房价格预测)|
|Regression to values between 0 and 1<br/>回归到0~1之间的值|sigmoid|mse or binary_crossentropy|Engine health assessment where 0 is broken, 1 is new|

## 二元分类任务 - Dog VS Cat
This [competition on Kaggle](https://www.kaggle.com/c/dogs-vs-cats) is where you write an algorithm to classify whether images contain either a dog or a cat. It is a binary classification task where the output of the model is a single number range from 0~1 where the lower value indicates the image is more "Cat" like, and higher value if the model thing the image is more "Dog" like.

Here are the code for the last fully connected layer and the loss function used for the model

```python
#Dog VS Cat last Dense layer
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
```

如果您对这个dog vs cat任务的完整源代码感兴趣，请看一下GitHub上的这个 [很棒的教程](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb) on GitHub.

## 多级单标签分类 - MNIST

任务是将手写数字（28像素乘28像素）的灰度图像分类为10个类别（0到9）

最后一层使用“ softmax ”激活，这意味着它将返回10个概率分数的数组（总和为1）。 每个分数将是当前数字图像属于我们的10个数字类别之一的概率。

```python
#MNIST last Dense layer
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

[完整源代码](https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.1-introduction-to-convnets.ipynb) for MNIST classification is provided on GitHub.

## 多级，多标签分类 - 新闻标签分类

[Reuters-21578](https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/) 是一个约20K新闻线的集合，分类为672个标签。 它们分为五大类：

* Topics
* Places
* People
* Organizations
* Exchanges

例如，一个新闻可以有3个标签

* Places: USA, China
* Topics:  trade

```python
# News tags classification last Dense layer
model.add(Dense(num_categories, activation='sigmoid'))
model.compile(loss='binary_crossentropy', 
              optimizer='adam', metrics=['accuracy'])
```

[源代码](https://github.com/Tony607/Text_multi-class_multi-label_Classification) on my GitHub.

I also wrote [another blog for this task in detail](https://www.dlology.com/blog/how-to-do-multi-class-multi-label-classification-for-news-categories/) as well, check out if you are interested.

## 回归任意值 - Boston住房价格预测

目标是使用给定数据预测单个连续值而不是房价的离散标签。

网络以Dense结束而没有任何激活，因为应用任何激活函数（如sigmoid）会将值约束为0~1，我们不希望这种情况发生。

**mse**损失函数，它计算预测和目标之间差异的平方，这是回归任务广泛使用的损失函数。

```python
# predict house price last Dense layer
model.add(layers.Dense(1))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
```

[源代码](http://nbviewer.jupyter.org/github/fchollet/deep-learning-with-python-notebooks/blob/master/3.7-predicting-house-prices.ipynb) can be found in the same GitHub repo.

## 回归到0到1之间的值

对于像评估喷气发动机的健康状况这样的任务，提供多个传感器记录。 我们希望输出是从0到1的连续值，其中0表示需要更换发动机，1表示它处于完美状态，而0和1之间的值可能意味着需要一定程度的维护。
 
与之前的回归问题相比，我们将 **sigmoid** 激活应用于最后一个密集层，以将值约束在0到1之间。

```python
# Jet engine health assessment last Dense layer
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
```