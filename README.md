# Recommend-System-TF2.0

<p align="left">
  <img src='https://img.shields.io/badge/python-3.6-blue'>
  <img src='https://img.shields.io/badge/tensorflow-2.0-brightgreen'>
  <img src='https://img.shields.io/badge/keras-2.4.3-brightgreen'>
</p>  

<p align="left">
  <a href="https://www.zhihu.com/people/yu-yi-chu-shi"><img src="https://img.shields.io/badge/知乎-予以初始-blue" alt=""></a>
  <a href="https://blog.csdn.net/weixin_45658131?spm=1000.2115.3001.5343"><img src="https://img.shields.io/badge/CSDN-予以初始-red" alt=""></a>
  <a href="https://jc-leehub.github.io/"><img src="https://img.shields.io/badge/HomePage-予以初始-green" alt=""></a>
  <a href="https://github.com/jc-LeeHub/Recommend-System-TF2.0/blob/master/image/IMG_4498(20210107-214348).JPG"><img src="https://img.shields.io/badge/微信号-予以初始-brightgreen" alt=""></a>
</p>

此仓库用于记录在学习推荐系统过程中的知识产出，主要是对经典推荐算法的**原理解析**及**代码实现**。

算法包含但不仅限于下图中的算法，**持续更新中...**

<div align=center><img src='https://cdn.jsdelivr.net/gh/jc-LeeHub/Recommend-System-tf2.0@master/image/%E6%8E%A8%E8%8D%90%E7%AE%97%E6%B3%95.png' width='80%' style="float:center"/></div>

## Models List

|  Model | Paper |                                                                                                                                        
| :----: | :------- | 
|  [FM](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/FM) | [ICDM 2010] [Fast Context-aware Recommendationswith Factorization Machines](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle_et_al2011-Context_Aware.pdf)           |
|  [CCPM](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/CCPM)  | [CIKM 2015] [A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)           |
|  [FFM](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/FFM) | [RecSys 2016] [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)           |
|  [FNN](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/FNN)   | [ECIR 2016] [Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                  |
|  [PNN](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/PNN)   | [ICDM 2016] [Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                 |
|  [Wide & Deep](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/WideDeep) | [DLRS 2016] [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                         |         
|  [Deep Crossing](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/DeepCrossing) | [KDD 2016] [Deep Crossing: Web-Scale Modeling withoutManually Crafted Combinatorial Features](https://www.kdd.org/kdd2016/papers/files/adf0975-shanA.pdf)                                                         |  
|  [DeepFM](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/DeepFM)   | [IJCAI 2017] [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                      |
|  [Deep & Cross Network](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/DCN)    | [ADKDD 2017] [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                               |                           
|  [AFM](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/AFM) | [IJCAI 2017] [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
|  [NFM](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/NFM) | [SIGIR 2017] [Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |
|  Piece-wise Linear Model | [arxiv 2017] [Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)             |                           
|  [xDeepFM](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/xDeepFM) | [KDD 2018] [xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                     |
|  [DIN](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/DIN) | [KDD 2018] [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)  
|  [MMoE](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/MMOE) | [KDD 2018]  [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)              ||
|  FwFM | [WWW 2018] [Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf)               |
|  [AutoInt](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/AutoInt) | [CIKM 2019] [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                           |                        
|  DIEN | [AAAI 2019] [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                           |
|  ONN  | [arxiv 2019] [Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                               |
|  [FGCNN](https://github.com/jc-LeeHub/Recommend-System-tf2.0/tree/master/FGCNN) | [WWW 2019] [Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction ](https://arxiv.org/pdf/1904.04447)                           |
|  DSIN  | [IJCAI 2019] [Deep Session Interest Network for Click-Through Rate Prediction ](https://arxiv.org/abs/1905.06482)    |
|  FiBiNET| [RecSys 2019] [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)|
|  FLEN | [arxiv 2019] [FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/pdf/1911.04690.pdf)                                                         |
|  DCN V2  | [arxiv 2020] [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535)          |

<img src='https://cdn.jsdelivr.net/gh/jc-LeeHub/Recommend-System-tf2.0@master/image/%E5%88%86%E5%89%B2%E7%BA%BF.jpg' width='100%'>

## Introduction

- 原理结合代码食用更佳，掌握算法的最好方式就是用代码撸它

- 原理解析可参考知乎专栏 [推荐算法也可以很简单](https://www.zhihu.com/column/c_1330637706267734016)

- 代码实践参考本仓库即可，每个模型都有对应README.md，对模型原理、代码结构、实验结果进行了介绍

**Tips:** 该仓库使用的代码均为TF2.0，如果你不熟悉该框架，可参考文档[**简单粗暴的Tensorflow2.0**](https://tf.wiki/zh_hans/basic/models.html)

## Citation

- 论文列表引用于浅梦，并作了相应补充. Weichen Shen.(2017). DeepCTR: Easy-to-use,Modular and Extendible package of deep-learning based CTR models. https://github.com/shenweichen/deepctr. 感谢整理！

## About

- 知乎：[予以初始](https://www.zhihu.com/people/yu-yi-chu-shi)

- CSDN: [予以初始](https://blog.csdn.net/weixin_45658131?spm=1000.2115.3001.5343)

- Website: [HomePage](https://jc-leehub.github.io/)

- E-mail: junchaoli@hnu.edu.cn

- wechat ID: Liii00061333

<img src='https://cdn.jsdelivr.net/gh/jc-LeeHub/Recommend-System-tf2.0@master/image/IMG_4498(20210107-214348).JPG' width='20%'>
