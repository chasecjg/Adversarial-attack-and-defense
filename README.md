
人脸识别对抗攻防汇总

# 一. 人脸识别

## 1.1. 常用数据集
| Data Name                                  | Description                                                                                  |
|--------------------------------------------|----------------------------------------------------------------------------------------------|
| Labeled Faces in the Wild（LFW）           | 包含超过1万张人脸图像的数据集，涵盖了不同角度、光照和人种                                    |
| The Extended Yale Face Database B（YaleB） | 包括2414张图像，来自38个不同的人，共有64个不同的光照条件                                     |
| CelebA                                     | 包含超过20万个名人图像的数据集，可以用于人脸识别、人脸属性分析、人脸合成等                   |
| MegaFace                                   | 包含来自690k个不同身份的1000万个人脸图像的数据集，是当前最大的公共人脸识别数据集之一         |
| VGGFace2                                   | 包含超过9000个身份的超过340万个人脸图像的数据集                                              |
| CASIA-WebFace                              | 包含超过5000个身份的超过50万个人脸图像的数据集，适用于人脸识别，尤其是在视角和光照方面的变化 |


## 1.2. 常用网络
| Model    | Publish | Year | Paper                                                                                                           | Code                                                                                                                               |
|----------|---------|------|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| DeepFace | CVPR    | 2014 | [paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf) | [code](https://github.com/serengil/deepface)                                                                                       |
| VGGFace  | ICLR    | 2015 | [paper](https://arxiv.org/pdf/1409.1556v6.pdf)                                                                  | [code1](https://github.com/tensorflow/models/tree/master/research/audioset)/[code2](https://github.com/facebookresearch/detectron) |
| FaceNet  | CVPR    | 2015 | [paper](https://arxiv.org/pdf/1503.03832v3.pdf)                                                                 | [code1](https://github.com/davidsandberg/facenet)/[code2](https://github.com/timesler/facenet-pytorch)                             |
| SphereFace | CVPR    | 2017 | [paper](https://arxiv.org/pdf/1704.08063v4.pdf) | [code1](https://github.com/wy1iu/sphereface)/[code2](https://github.com/clcarwin/sphereface_pytorch) |
| ArcFace  | CVPR    | 2019 | [paper](https://arxiv.org/pdf/1801.07698v4.pdf)                                                                 | [code](https://github.com/deepinsight/insightface)                                                                                 |
| SFace    | TIP     | 2021 | [paper](https://arxiv.org/pdf/2205.12010v1.pdf)                                                                 | [code1](https://github.com/zhongyy/SFace)/[code2](https://github.com/serengil/deepface)                                            |

## 1.3. 常用损失
| Method     | Publish | Year | Paper                                           | Code                                                                                                 |
|------------|---------|------|-------------------------------------------------|------------------------------------------------------------------------------------------------------|
| SphereFace | CVPR    | 2017 | [paper](https://arxiv.org/pdf/1704.08063v4.pdf) | [code1](https://github.com/wy1iu/sphereface)/[code2](https://github.com/clcarwin/sphereface_pytorch) |
| CosFace    | CVPR    | 2018 | [paper](https://arxiv.org/pdf/1801.09414v2.pdf) | [code](https://github.com/cvqluu/Additive-Margin-Softmax-Loss-Pytorch)                               |
| ArcFace    | CVPR    | 2019 | [paper](https://arxiv.org/pdf/1801.07698v4.pdf) | [code](https://github.com/deepinsight/insightface)                                                   |
| BroadFace  | ECCV    | 2020 | [paper](https://arxiv.org/pdf/2008.06674v1.pdf) | [code](https://arxiv.org/pdf/2008.06674v1.pdf)                                                       |
| MagFace    | CVPR    | 2021 | [paper](https://arxiv.org/pdf/2103.06627v4.pdf) | [code](https://github.com/IrvingMeng/MagFace)                                                        |
| AdaFace    | CVPR    | 2022 | [paper](https://arxiv.org/pdf/2204.00964v2.pdf) | [code](https://github.com/mk-minchul/adaface)                                                        |

# 二. 人脸攻防


# 2.1. 常用攻击

这里主要列举一些物理攻击方法，如帽子攻击，眼睛攻击，眼影攻击，贴纸攻击等

| Name                                                                                   | Publish  | Year | Paper                                                                        | Code                                                   |
|----------------------------------------------------------------------------------------|----------|------|------------------------------------------------------------------------------|--------------------------------------------------------|
| Accessorize to a crime: Real and stealthy attacks on state-of-the-art face recognition | CCS      | 2016 | [paper](https://users.cs.northwestern.edu/~srutib/papers/face-rec-ccs16.pdf) | \                                                      |
| AdvHat: Real-world adversarial attack on ArcFace Face ID system                        | \        | 2019 | [paper](https://arxiv.org/pdf/1908.08705v1.pdf)                              | [code](https://github.com/papermsucode/advhat)         |
| Efficient Decision-based Black-box Adversarial Attacks on Face Recognition             | CVPR     | 2019 | [paper](https://arxiv.org/pdf/1904.04433v1.pdf)                              | [code](https://github.com/SCLBD/BlackboxBench)         |
| On adversarial patches: real-world attack on ArcFace-100 face recognition system       | SIBIRCON | 2019 | [paper](https://ieeexplore.ieee.org/document/8958134)                        | \                                                      |
| Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Model    | \        | 2021 | [paper](https://arxiv.org/pdf/2111.10759.pdf)                                | [demo](https://youtu.be/_TXkDO5z11w)                   |
| Adv-Makeup: A New Imperceptible and Transferable Attack on Face Recognition            | IJCAI    | 2021 | [paper](https://arxiv.org/abs/2105.03162)                                    | \                                                      |
| Adversarial Sticker: A Stealthy Attack Method in the Physical World                    | TPAMI    | 2022 | [paper](https://ieeexplore.ieee.org/abstract/document/9779913)               | [code](https://github.com/jinyugy21/Adv-Stickers_RHDE) |
| Adv-Attribute: Inconspicuous and Transferable Adversarial Attack on Face Recognition   | NeurIPS  | 2022 | [paper](https://arxiv.org/abs/2210.06871)                                    | \                                                      |
| Towards Effective Adversarial Textured 3D Meshes on Physical Face Recognition          | CVPR     | 2023 | [paper](https://arxiv.org/pdf/2303.15818v1.pdf)                              | [code](https://github.com/thu-ml/at3d)                 |
| Discrete Point-wise Attack Is Not Enough: Generalized Manifold Adversarial Attack for Face Recognition | CVPR | 2023 | [paper](https://arxiv.org/pdf/2301.06083v2.pdf)                  | [code](https://github.com/tokaka22/gmaa)               |


# 2.2. 常用防御

## 2.2.1 通用防御

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
## 2.2.2 针对人脸防御
| Name                                                                                        | Publish | Year | Paper                                                          | Code                                        |
|---------------------------------------------------------------------------------------------|---------|------|----------------------------------------------------------------|---------------------------------------------|
| Perturbation Inactivation Based Adversarial Defense for Face Recognition                    | TIFS    | 2022 | [paper](https://ieeexplore.ieee.org/abstract/document/9845464) | [code](https://github.com/AmIAttribute/AmI) |
| A Random-patch based Defense Strategy Against Physical Attacks for Face Recognition Systems | \       | 2023 | [paper](https://arxiv.org/abs/2304.07822)                      | \                                           |
| Detecting Adversarial Faces Using Only Real Face Self-Perturbations                         | IJCAI   | 2023 | [paper](https://arxiv.org/pdf/2304.11359v2.pdf)                | [code](https://github.com/cc13qq/sapd)      |



