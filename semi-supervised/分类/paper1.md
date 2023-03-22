## FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence(NIPS2020)

1. 摘要
这篇文章同时运用了伪标签以及强弱增强之间的一致性来提高半监督学习的性能。
FixMatch 使用一致正则化(consistency regularization)和伪标签(pseudo-labeling)来产生人造标签。基于**弱增强**(例如，仅仅使用翻转、移动等数据增强操作)的无标签图像经过模型预测得到的伪标签作为相同图像**强增强**视角的标签进行训练。人造标签选择置信概率最高的类别作为标签选择依据。FixMatch 简化了现有的其它方法，使得超参数的数量减少，更利于探究关键因素。

![image](https://user-images.githubusercontent.com/62278179/226886656-1b59c711-7dca-4cd8-8e30-0dd04c188012.png)

2. Method
损失函数主要包括两个部分，即有标签的监督损失和无标签的损失两部分。

![image](https://user-images.githubusercontent.com/62278179/226887790-53ae7a5c-ca82-4a2d-bd87-a170d0fe6dbc.png)





