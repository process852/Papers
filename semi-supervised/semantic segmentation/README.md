## 语义分隔中的半监督学习
`A Survey on Semi-Supervised Semantic Segmentation` 是一篇关于半监督学习在语义分割任务中的综述性文章，其 [下载链接](https://arxiv.org/pdf/2302.09899.pdf)。

#### 引言
语义分割是针对图像像素级别的分类任务，根据每个像素所属的语义类别分配对应的标签值。随着深度学习技术的发展，高质量的数据集标注愈发重要。但是语义分割任务的标签标注工作相较于其它 CV 任务是更加耗时和繁琐的，为此研究者提出了弱监督、半监督、无监督等不同方式来尽可能缓解对于标签数量的要求。而本文主要是集中分析了近几年在半监督语义分割任务上的工作进展。下图展示了近几年在该领域发表的论文和引用趋势。

![image](https://user-images.githubusercontent.com/62278179/222064632-a4972c31-0671-4c00-996f-639c448ec452.png)

#### 背景
假设我们拥有数据集 $X = \{X_L, X_U\}$，其中 $X_L = \{(x_i, y_i)\}_{i = 1}^l$ 表示有标签部分的数据集，$X_U = \{x_i\}_{i = 1}^u$ 表示无标签数据，通常来说 u 远大于 l。半监督语义分割的最终目标是**从有标签和无标签数据中提取知识**获得一个比仅在**有标签数据**训练下的更好模型。
#### 方法
![image](https://user-images.githubusercontent.com/62278179/222067688-0103894c-5f90-463f-bf7d-b108253cf4dd.png)

- Adversarial methods
- Consistency regularization
一致正则化基于光滑假设，也就是一个点附近的点的预测标签应该保持一致。一致正则化的方法基本上都基于 `Mean Teacher` 方法展开，它迫使学生模型与老师模型预测保持一致。基于一致正则化的语义分割主要区别在于如何添加扰动。目前主要有以下 4 个方式的扰动形式：
    - Input perturbations：输入数据扰动主要是采取不同的数据增强策略，使其增强前后的预测结果保持一致。
    - Feature perturbations：主要是针对经过网络处理输出的特征图进行扰动添加
    - Network perturbations：主要是指的是不同网络或者是相同网络不同初始化的形式
    - Combined perturbations：主要是混合以上几种形式的扰动
-  Pseudo-labeling methods
伪标签方法的基本思路利用有标签数据训练的模型生成无标签数据的伪标签。
-  Contrastive learning
