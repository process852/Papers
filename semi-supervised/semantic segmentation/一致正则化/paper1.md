## Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation(Arxiv)

#### 摘要

![image](https://user-images.githubusercontent.com/62278179/222645819-539e005c-b36c-4874-9016-97924b2bc04c.png)

本文作者借鉴了分类任务中的强弱视角进行监督学习，为了提高在语义分割中的扰动空间范围，作者提出了添加一个单独的特征空间扰动流作为补充。另一方面，为了提高图像级别的数据增强，提出了双流扰动技术，即一个弱增强视角同时监督两个不同强增强视角。

#### 引言

作者实验发现图像的数据增强操作使得 `FixMatch` 在语义分割中扮演了重要作用。观察下表可以发现，图像数据增强操作使得性能有了大的提升。

![image](https://user-images.githubusercontent.com/62278179/222647409-a644b644-ee83-4daa-995c-e59c34e9919a.png)

为了本论文继承了 `FixMatch` 的图像增强的优点，同时以两个不同的视角加强了扰动空间的范围。
- expanding a broader perturbation space
分别在图像级和特征级分别对其产生扰动，图像级的扰动主要包含颜色抖动、随机翻转、CutMix等。而特征级的扰动是简单的特征图通道 `Dropout`。与之前的方式不同在于作者将**图像级和特征级扰动分成两个单独的流**进行训练，这样可以减轻学习的负担。
![image](https://user-images.githubusercontent.com/62278179/222648494-4d3c2e5f-b95e-47f4-af4b-cfafdb6e8bb6.png)

- sufficiently harvesting original perturbations
在 FixMatch 中仅仅使用了一个强增强视角，作者为了充分利用预先定义的扰动操作，使用两个独立的生成的强增强视角。增强操作都是随机从同一个扰动池采样得来。

![image](https://user-images.githubusercontent.com/62278179/222649850-787083c1-c505-4e8f-ae38-6ecb495bb1ec.png)


#### 方法

![image](https://user-images.githubusercontent.com/62278179/222651423-52d0bbca-583d-42ab-b729-4a59458508dd.png)

为了构建更加广阔的扰动空间，本文提出在弱增强视角下的特征图上注入扰动信息，同时作者人为图像级别扰动和特征级别扰动分离对于性能有一定提升。

![image](https://user-images.githubusercontent.com/62278179/222652639-37fe8f9b-c125-4b9f-81b0-2da5e9a659dd.png)

