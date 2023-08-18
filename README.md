[[_TOC_]]

# 一. 人脸识别

人脸识别是一种通过分析人脸图像特征进行身份识别的一种方法。

* 一般流程

人脸检测->人脸对齐->特征提取->人脸分类

## 1.1. 常用人脸识别数据集
| Data Name                                  | Description                                                                                  |
|--------------------------------------------|----------------------------------------------------------------------------------------------|
| Labeled Faces in the Wild（LFW）           | 包含超过1万张人脸图像的数据集，涵盖了不同角度、光照和人种                                    |
| The Extended Yale Face Database B（YaleB） | 包括2414张图像，来自38个不同的人，共有64个不同的光照条件                                     |
| CelebA                                     | 包含超过20万个名人图像的数据集，可以用于人脸识别、人脸属性分析、人脸合成等                   |
| MegaFace                                   | 包含来自690k个不同身份的1000万个人脸图像的数据集，是当前最大的公共人脸识别数据集之一         |
| VGGFace2                                   | 包含超过9000个身份的超过340万个人脸图像的数据集                                              |
| CASIA-WebFace                              | 包含超过5000个身份的超过50万个人脸图像的数据集，适用于人脸识别，尤其是在视角和光照方面的变化 |
| MTFL                                       | 包含 12,995 张人脸图像，用 (1) 五个面部标志，(2) 性别、微笑、戴眼镜和头部姿势的属性进行了注释|
| BioID                                      | 包含了1521幅分辨率为384x286像素的灰度图像。每一幅图像来自于23个不同的测试人员的正面角度的人脸|

## 1.2. 常用模型
| Model    | Publish | Year | Paper                                                                                                           | Code                                                                                                                               |
|----------|---------|------|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| DeepFace | CVPR    | 2014 | [paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf) | [code](https://github.com/serengil/deepface)                                                                                       |
| VGGFace  | ICLR    | 2015 | [paper](https://arxiv.org/pdf/1409.1556v6.pdf)                                                                  | [code1](https://github.com/tensorflow/models/tree/master/research/audioset)/[code2](https://github.com/facebookresearch/detectron) |
| FaceNet  | CVPR    | 2015 | [paper](https://arxiv.org/pdf/1503.03832v3.pdf)                                                                 | [code1](https://github.com/davidsandberg/facenet)/[code2](https://github.com/timesler/facenet-pytorch)                             |
| SphereFace | CVPR    | 2017 | [paper](https://arxiv.org/pdf/1704.08063v4.pdf) | [code1](https://github.com/wy1iu/sphereface)/[code2](https://github.com/clcarwin/sphereface_pytorch) |
| CosFace    | CVPR    | 2018 | [paper](https://arxiv.org/pdf/1801.09414v2.pdf) | [code](https://github.com/cvqluu/Additive-Margin-Softmax-Loss-Pytorch)                               |
| **ArcFace**  | CVPR    | 2019 | [paper](https://arxiv.org/pdf/1801.07698v4.pdf)                                                                 | [code](https://github.com/deepinsight/insightface)                                                                                 |
| BroadFace  | ECCV    | 2020 | [paper](https://arxiv.org/pdf/2008.06674v1.pdf) | [code](https://arxiv.org/pdf/2008.06674v1.pdf)                                                       |
| MagFace    | CVPR    | 2021 | [paper](https://arxiv.org/pdf/2103.06627v4.pdf) | [code](https://github.com/IrvingMeng/MagFace)                                                        |
| SFace    | TIP     | 2021 | [paper](https://arxiv.org/pdf/2205.12010v1.pdf)                                                                 | [code1](https://github.com/zhongyy/SFace)/[code2](https://github.com/serengil/deepface)                                            |
| MagFace    | CVPR    | 2021 | [paper](https://arxiv.org/pdf/2103.06627v4.pdf) | [code](https://github.com/IrvingMeng/MagFace)                                                        |
| AdaFace    | CVPR    | 2022 | [paper](https://arxiv.org/pdf/2204.00964v2.pdf) | [code](https://github.com/mk-minchul/adaface)                                                        |


## 1.3. 常用损失

* 两类：
1. 基于Euclid Distance的损失函数（Contrastive Loss、Triplet Loss、Center Loss等)
2. 基于Angular Margin相关的损失函数（L-Softmax Loss、A-Softmax Loss、CosFace Loss、ArcFace Loss等）。

| Method | Description | Publication |
|--------|-------------|----|
|对比损失| 作者：Yann LeCun，是一种降维学习方法，它可以学习一种映射关系，这种映射关系可以使得在高维空间中，相同类别但距离较远的点，通过函数映射到低维空间后，距离变近，不同类别但距离都较近的点，通过映射后再低维空间变得更远。该损失函数在深度学习中主要是用于降维中，即本来相似的样本，在经过降维（特征提取）后，在特征空间中，两个样本仍旧相似；而原本不相似的样本，在经过降维后，在特征空间中，两个样本仍旧不相似。            |  [paper](https://link.zhihu.com/?target=http%3A//yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf)  |
|**三元组损失**| 单位：Google，有三个样本，anchor表示某个样本，positive表示和anchor相同的样本（代表同一个人），negative表示和anchor不同的样本。然后通过网络学习，让anchor和positive相互靠近，anchor和negative尽量远离。 Triplet loss的目标是：让相同样本在嵌入空间中尽量靠近在一起，不同标签的两个样本的嵌入距离要很远。         |  [paper](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1503.03832)  |
| 中心损失|  作者：陈乔宇，为每一个类别提供一个类别中心，最小化min-batch中每个样本与该类别中心的距离，即缩小类内距离。有效地表征了深度特征的类内距离，提升深度特征的判别能力，在保持不同类别的特征可分离的同时最小化类内距离是关键。          |  [paper](https://link.zhihu.com/?target=https%3A//ydwen.github.io/papers/WenECCV16.pdf)  |
| L-Softmax loss | Softmax Loss并不能够有效地学习得到使得类内较为紧凑、类间较离散的特征，L-Softmax loss希望通过增加一个正整数变量m，从而产生一个决策余量，对softmax进行改进，它学习到的参数可以将不同类样本的类间距离加大，最后学到的特征之间的分离程度比原来的要明显得多    |  [paper](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1612.02295.pdf)  |
|A-Softmax loss  |  A-Softmax（Angular Softmax loss）：在L-Softmax loss的基础上做权重归一化和偏置项归零，使得预测仅取决于权重和偏置之间的角度   |  [paper](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1704.08063)  |
| CosFace Loss |  cosface loss的设计思路是通过在余弦空间上减去一个正值，使得在0-π     |  [paper](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1801.09414)  |   
| ArcFace Loss | arc loss的出发点是，从反余弦空间优化类间距离，通过在夹角上加个m，使得cos值在0-π单调区间上值更小   |  [paper](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1801.07698) |


# 二. 对抗攻防

* 攻击

通过对攻击的目标图像加入肉眼难以察觉的噪声对图像进行各种形式的攻击。

|攻击形式|描述|
|--------|----|
| 数字攻击  | 通过添加数字噪声对图像进行扰动实现攻击，让模型做出错误的预测  |
| 物理攻击 |  通过物理的方式，如贴纸，眼镜，眼影等进行攻击，是的模型做出错误的预测 |
| 白盒攻击 | 神经网络攻击者对要攻击的模型的网络架构，网络权重，防御方法等信息了如指掌  |
| 黑盒攻击 |  黑盒攻击与白盒攻击正好相反，攻击者对模型一无所知，只能通过模型的输入 / 输出反馈有限的获取到模型的内部信息，并利用有限的信息进行攻击 |
| 灰盒攻击 |  攻击者只掌握模型的部分信息，如，只知道模型的网络架构，但并不知道网络权重 |
| 目标攻击 |  攻击者对模型进行定向攻击，使模型进行定向错误预测。如，对原始图像植入攻击扰动(Attack Perturbation)，使模型对任何植入扰动的图像都预测为人(human) |
| 非目标攻击 |  攻击者对模型进行不定向攻击，攻击者只能使模型进行错误预测，而无法使模型进行定向错误预测 |
| 其他划分方式  | 扰动强度：0范数，二范数，无穷范数等；实现方式：基于梯度攻击，基于优化方式攻击，基于决策面攻击等 |

* 防御

1. 主要分为三种：去噪，滤除，检测

2. 主要关注：物理攻击+黑盒攻击+非目标攻击


## 2.1. 主流攻击方法

这里主要列举一些物理攻击方法，如帽子攻击，眼镜攻击，眼影攻击，贴纸攻击等

| Name                                                                                   | Publish  | Year | Paper                                                                        | Code                                                   |
|----------------------------------------------------------------------------------------|----------|------|------------------------------------------------------------------------------|--------------------------------------------------------|
| Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition | CCS      | 2016 | [paper](https://users.cs.northwestern.edu/~srutib/papers/face-rec-ccs16.pdf) | [code](https://github.com/mahmoods01/accessorize-to-a-crime)                                                      |
| AdvHat: Real-world adversarial attack on ArcFace Face ID system                        | ICPR        | 2019 | [paper](https://arxiv.org/pdf/1908.08705v1.pdf)                              | [code](https://github.com/papermsucode/advhat)         |
| Efficient Decision-based Black-box Adversarial Attacks on Face Recognition             | CVPR     | 2019 | [paper](https://arxiv.org/pdf/1904.04433v1.pdf)                              | [code](https://github.com/SCLBD/BlackboxBench)         |
| On adversarial patches: real-world attack on ArcFace-100 face recognition system       | SIBIRCON | 2019 | [paper](https://ieeexplore.ieee.org/document/8958134)                        | \                                                      |
| Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Model    | \        | 2021 | [paper](https://arxiv.org/pdf/2111.10759.pdf)                                | [code](https://github.com/alonzolfi/adversarialmask)                   |
| Adv-Makeup: A New Imperceptible and Transferable Attack on Face Recognition            | IJCAI    | 2021 | [paper](https://arxiv.org/abs/2105.03162)                                    | [code](https://github.com/TencentYoutuResearch/Adv-Makeup)|                                                      |
| FACESEC: A Fine-grained Robustness Evaluation Framework for Face Recognition Systems   | CVPR     | 2021 | [paper](https://arxiv.org/pdf/2104.04107v1.pdf) | [code](https://github.com/KnowledgeDiscovery/FaceSec) |
| Towards face encryption by generating adversarial identity masks                       | ICCV     | 2021 | [paper](https://arxiv.org/pdf/2003.06814v2.pdf) | [code](https://github.com/shawnxyang/tip-im) | 
| Adversarial Sticker: A Stealthy Attack Method in the Physical World                    | TPAMI    | 2022 | [paper](https://ieeexplore.ieee.org/abstract/document/9779913)               | [code](https://github.com/jinyugy21/Adv-Stickers_RHDE) |
| Adv-Attribute: Inconspicuous and Transferable Adversarial Attack on Face Recognition   | NeurIPS  | 2022 | [paper](https://arxiv.org/abs/2210.06871)                                    | \                                                      |
| Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer | CVPR | 2022 | [paper](https://arxiv.org/pdf/2203.03121v2.pdf) | [code](https://github.com/cgcl-codes/amt-gan) |
| Towards Effective Adversarial Textured 3D Meshes on Physical Face Recognition          | CVPR     | 2023 | [paper](https://arxiv.org/pdf/2303.15818v1.pdf)                              | [code](https://github.com/thu-ml/at3d)                 |
| Discrete Point-wise Attack Is Not Enough: Generalized Manifold Adversarial Attack for Face Recognition | CVPR | 2023 | [paper](https://arxiv.org/pdf/2301.06083v2.pdf)                  | [code](https://github.com/tokaka22/gmaa)               |



## 2.2. 主流防御方法

### 2.2.1 通用防御

| Name                                                                                                                  | Publish | Year | Paper                                                                                                                                                                           | Code                                                                                                                        |
|-----------------------------------------------------------------------------------------------------------------------|---------|------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| Defense against Adversarial Attacks Using High-Level Representation Guided Denoiser                                   | CVPR    | 2018 | [paper](https://arxiv.org/pdf/1712.02976v2.pdf)                                                                                                                                 | [code](https://github.com/lfz/Guided-Denoise)                                                                               |
| PixelDefend: Leveraging Generative Models to Understand and Defend against Adversarial Examples                       | ICLR    | 2018 | [paper](https://arxiv.org/pdf/1710.10766v3.pdf)                                                                                                                                 | [code](https://github.com/Microsoft/PixelDefend)                                                                            |
| Towards Deep Learning Models Resistant to Adversarial Attacks                                                         | ICLR    | 2018 | [paper](https://arxiv.org/pdf/1706.06083v4.pdf)                                                                                                                                 | [code1](https://github.com/MadryLab/mnist_challenge)/[code2](https://github.com/locuslab/convex_adversarial)                |
| Feature Squeezing: Detecting Adversarial Examples in Deep Neural Networks                                             | NDSS    | 2018 | [paper](https://arxiv.org/pdf/1704.01155v2.pdf)                                                                                                                                 | [code](https://github.com/mzweilin/EvadeML-Zoo)                                                                             |
| Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples                                     | NeurIPS | 2018 | [paper](https://arxiv.org/pdf/1810.11580v1.pdf)                                                                                                                                 | [code](https://github.com/AmIAttribute/AmI)                                                                                 |
| Ensemble Adversarial Training: Attacks and Defenses                                                                   | ICLR    | 2018 | [paper](https://arxiv.org/pdf/1705.07204v5.pdf)                                                                                                                                 | [code1](https://github.com/tensorflow/models)/[code2](https://github.com/csdongxian/skip-connections-matter)                |
| Countering Adversarial Images using Input Transformations                                                             | ICLR    | 2018 | [paper](https://arxiv.org/pdf/1711.00117v3.pdf)                                                                                                                                 | [code](https://github.com/facebookresearch/adversarial_image_defenses)                                                      |
| Hilbert-Based Generative Defense for Adversarial Examples                                                             | ICCV    | 2019 | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Bai_Hilbert-Based_Generative_Defense_for_Adversarial_Examples_ICCV_2019_paper.pdf)                                | [code]()                                                                                                                    |
| Adversarial Defense by Stratified Convolutional Sparse Coding                                                         | CVPR    | 2019 | [paper](https://arxiv.org/pdf/1812.00037v2.pdf)                                                                                                                                 | [code](https://github.com/gitbosun/advdefense_csc)                                                                          |
| CIIDefence: Defeating Adversarial Attacks by Fusing Class-Specific Image Inpainting and Image Denoising               | ICCV    | 2019 | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Gupta_CIIDefence_Defeating_Adversarial_Attacks_by_Fusing_Class-Specific_Image_Inpainting_and_ICCV_2019_paper.pdf) | [code]()                                                                                                                    |
| Feature Denoising for Improving Adversarial Robustness                                                                | CVPR    | 2019 | [paper](https://arxiv.org/pdf/1812.03411v2.pdf)                                                                                                                                 | [code1](https://github.com/facebookresearch/ImageNet-Adversarial-Training)/[code2](https://github.com/lirundong/quant-pack) |
| Adversarial Defense via Learning to Generate Diverse Attacks                                                          | ICCV    | 2019 | [paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Jang_Adversarial_Defense_via_Learning_to_Generate_Diverse_Attacks_ICCV_2019_paper.pdf)                            | [code](https://github.com/YunseokJANG/l2l-da)                                                                               |
| Parametric Noise Injection: Trainable Randomness to Improve Deep Neural Network Robustness against Adversarial Attack | CVPR    | 2019 | [paper](https://arxiv.org/pdf/1811.09310v1.pdf)                                                                                                                                 | [code](https://github.com/elliothe/CVPR_2019_PNI)                                                                           |
| Improving Adversarial Robustness via Guided Complement Entropy                                                        | ICCV    | 2019 | [paper](https://arxiv.org/pdf/1903.09799v3.pdf)                                                                                                                                 | [code](https://github.com/henry8527/GCE)                                                                                    |
| Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks                                           | ICCV    | 2019 | [paper](https://arxiv.org/pdf/1904.00887v4.pdf)                                                                                                                                 | [code](https://github.com/aamir-mustafa/pcl-adversarial-defense)                                                            |
| Adversarial Learning with Margin-based Triplet Embedding Regularization                                               | ICCV    | 2019 | [paper](https://arxiv.org/pdf/1909.09481v1.pdf)                                                                                                                                 | [code](https://github.com/aamir-mustafa/pcl-adversarial-defense)                                                            |
| Defending Against Physically Realizable Attacks on Image Classification                                               | ICLE    | 2020 | [paper](https://arxiv.org/pdf/1909.09552v2.pdf)                                                                                                                                 | [code](https://github.com/tongwu2020/phattacks)                                                                             |
| Architectural Adversarial Robustness: The Case for Deep Pursuit                                                       | CVPR    | 2021 | [paper](https://arxiv.org/pdf/2011.14427v1.pdf)                                                                                                                                 | \                                                                                                                           |
| Removing Adversarial Noise in Class Activation Feature Space                                                          | CVPR    | 2021 | [paper](https://arxiv.org/pdf/2104.09197v1.pdf)                                                                                                                                 | [code](https://arxiv.org/pdf/2104.09197v1.pdf)                                                                              |
| Person Re-identification Method Based on Color Attack and Joint Defence                                               | CVPR    | 2021 | [paper](https://arxiv.org/pdf/2111.09571v4.pdf)                                                                                                                                 | [code](https://github.com/finger-monkey/ReID_Adversarial_Defense)                                                           |
| How to Robustify Black-Box ML Models? A Zeroth-Order Optimization Perspective                                         | ICLR    | 2022 | [paper](https://openreview.net/pdf?id=W9G_ImpHlQd)                                                                                                                              | [code](https://github.com/damon-demon/black-box-defense)                                                                    |



### 2.2.2 人脸防御
| Name                                                                                        | Publish | Year | Paper                                                          | Code                                        |
|---------------------------------------------------------------------------------------------|---------|------|----------------------------------------------------------------|---------------------------------------------|
| Perturbation Inactivation Based Adversarial Defense for Face Recognition                    | TIFS    | 2022 | [paper](https://ieeexplore.ieee.org/abstract/document/9845464) | [code](https://github.com/renmin1991/perturbation-inactivate) |
| A Random-patch based Defense Strategy Against Physical Attacks for Face Recognition Systems | \       | 2023 | [paper](https://arxiv.org/abs/2304.07822)                      | \                                           |
| Detecting Adversarial Faces Using Only Real Face Self-Perturbations                    | IJCAI    | 2023 | [paper](https://arxiv.org/pdf/2304.11359v2.pdf) | [code](https://github.com/cc13qq/sapd) |


# 三. 攻防方案

* 涵盖内容

1. 领域进展，主流方案，方案水平，Outlook

2. Baseline选取，选取依据，方案水平，

3. Benchmark，训练集，测试集，模型测试方案

工作流规范

## 3.1. 防御方案

主要以人脸防御模型为主：灭火扰动，随机切片，对抗训练，去噪

### 3.1.1. 预选方案
1. 自扰动训练检测
2. 免疫空间灭火扰动
3. 攻击检测

| Name                                                                                        | Publish | Year | Paper                                                          | Code                                        |
|---------------------------------------------------------------------------------------------|---------|------|----------------------------------------------------------------|---------------------------------------------|
| Detecting Adversarial Faces Using Only Real Face Self-Perturbations                    | IJCAI    | 2023 | [paper](https://arxiv.org/pdf/2304.11359v2.pdf) | [code](https://github.com/cc13qq/sapd) |
| Perturbation Inactivation Based Adversarial Defense for Face Recognition                    | TIFS    | 2022 | [paper](https://ieeexplore.ieee.org/abstract/document/9845464) | [code](https://github.com/renmin1991/perturbation-inactivate) |
| Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples           | NeurIPS | 2018 | [paper](https://arxiv.org/pdf/1810.11580v1.pdf) | [code](https://github.com/AmIAttribute/AmI) |

### 3.1.2. 备选方案
1. 随机切片
2. 黑盒攻击防御
3. 特征去噪
3. MagNet

| Name                                                                                        | Publish | Year | Paper                                                          | Code                                        |
|---------------------------------------------------------------------------------------------|---------|------|----------------------------------------------------------------|---------------------------------------------|
| A Random-patch based Defense Strategy Against Physical Attacks for Face Recognition Systems | \       | 2023 | [paper](https://arxiv.org/abs/2304.07822)                      | \                                           |
| How to Robustify Black-Box ML Models? A Zeroth-Order Optimization Perspective                                         | ICLR    | 2022 | [paper](https://openreview.net/pdf?id=W9G_ImpHlQd)                                                                                                                              | [code](https://github.com/damon-demon/black-box-defense)                                                                    |
| Feature Denoising for Improving Adversarial Robustness                                                                | CVPR    | 2019 | [paper](https://arxiv.org/pdf/1812.03411v2.pdf)                                                                                                                                 | [code1](https://github.com/facebookresearch/ImageNet-Adversarial-Training)/[code2](https://github.com/lirundong/quant-pack) |
| MagNet: a Two-Pronged Defense against Adversarial Examples                                  | CCS     | 2017 | [paper](https://arxiv.org/pdf/1705.09064v2.pdf)                | [code1](https://github.com/Trevillie/MagNet)/[code2](https://github.com/GokulKarthik/MagNet.pytorch) |


## 3.2. 测试方案

### 3.2.1. 预选物理攻击

主要使用物理攻击方式：贴纸，帽子攻击，化妆，3D面罩攻击等

| Name                                                                                   | Publish  | Year | Paper                                                                        | Code                                                   |
|----------------------------------------------------------------------------------------|----------|------|------------------------------------------------------------------------------|--------------------------------------------------------|
| AdvHat: Real-world adversarial attack on ArcFace Face ID system                        | ICPR        | 2019 | [paper](https://arxiv.org/pdf/1908.08705v1.pdf)                              | [code](https://github.com/papermsucode/advhat)         |
| Adversarial Sticker: A Stealthy Attack Method in the Physical World                    | TPAMI    | 2022 | [paper](https://ieeexplore.ieee.org/abstract/document/9779913)               | [code](https://github.com/jinyugy21/Adv-Stickers_RHDE) |
| Towards Effective Adversarial Textured 3D Meshes on Physical Face Recognition          | CVPR     | 2023 | [paper](https://arxiv.org/pdf/2303.15818v1.pdf)                              | [code](https://github.com/thu-ml/at3d)                 |
| Adv-Makeup: A New Imperceptible and Transferable Attack on Face Recognition            | IJCAI    | 2021 | [paper](https://arxiv.org/abs/2105.03162)                                    | [code](https://github.com/TencentYoutuResearch/Adv-Makeup)|


### 3.2.2. 预选数字攻击

* 测试模型

1. BIM
2. DIFGSM
3. FGSM
4. MIFGSM
5. PGD
6. RFGSM
7. TIFGSM
8. TIPIM


