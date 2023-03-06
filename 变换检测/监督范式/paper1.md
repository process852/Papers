## Learning to Measure Changes: Fully Convolutional Siamese Metric Networks for Scene Change Detection

#### 摘要
场景变换检测的难点在于存在大量的噪声变换，例如光照变化、阴影覆盖以及拍摄视角变化等。本文利用比较特征对之间的相似性来检测变化区域。为了学习判别性的特征，利用对比损失去减少未变化特征对之间的距离，拉大变化特征对之间的距离。为了解决大的视角差的问题，还提出了阈值化的对比损失，以更加包含的策略来惩罚噪声变化。实验数据集主要包含 CDnet, PCD2015, VL_CMU_CD。
源代码地址：https://github.com/gmayday1997/ChangeDet

#### 引言
本文将变化视为不相似，如何定义不相似函数去衡量变化是关键。变化通常包含语义变化和噪声变化。借鉴其它视觉任务中的特征学习思想，鼓励减小类内方差，扩大类间差别。利用基于深度学习的方法，可以学习一个隐式的变化度量函数。本文中将变化区域的图像对称为正样本，而未变化区域的图像对称为负样本。

主要贡献：
- 提出了基于度量学习的场景变化检测，首次提出
- 发展了阈值化的对比损失去克服噪声变化
- 在一系列数据集上取得了 SOTA 性能

#### 方法

![image](https://user-images.githubusercontent.com/62278179/223004751-92b1c4f0-8b27-46db-a41d-b604fb9d6cbd.png)

- 对比损失：为了过滤一些可能出现的噪声变化，在对比损失中设置了一个阈值参数。

![image](https://user-images.githubusercontent.com/62278179/223004105-8a020d5c-f905-4945-a631-ce3dcd11121e.png)

- Thresholded Contrastive Loss
作者认为语义相同的地方使其特征完全相似也是不合理，所以设置了一个阈值。

![image](https://user-images.githubusercontent.com/62278179/223004460-c85bce51-040b-471c-b991-1dc4c0dde4ed.png)

- 训练策略
利用多层进行对比损失计算监督

- 数据预处理
  - VL_CMU_CD: 图像大小 resize 为 (512,512)
  - PCD2015: 图像大小使用原大小 (1024, 224)  5折交叉验证
  - CDnet: 91595 图像对，其中训练集为 73276对， 测试集为 18319对，图像缩放到 (512,512)
 







