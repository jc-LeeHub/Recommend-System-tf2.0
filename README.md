# Recommend-System-TF2.0

<p align="left">
  <img src='https://img.shields.io/badge/python-3.6-blue'>
  <img src='https://img.shields.io/badge/tensorflow-2.0-brightgreen'>
  <img src='https://img.shields.io/badge/keras-2.4.3-brightgreen'>
  <img src='https://img.shields.io/badge/numpy-1.16-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.0.3-brightgreen'>
  <img src='https://img.shields.io/badge/sklearn-0.23.2-brightgreen'>
</p>  

此仓库用于记录在学习推荐系统过程中的知识产出，主要是对基本推荐算法的**原理解析**以及**代码实现**。

算法包含但不仅限于下图中的算法，**持续更新中...**

<img src='https://github.com/jc-LeeHub/Recommend-System-TF2.0/blob/master/image/%E5%88%86%E5%89%B2%E7%BA%BF.jpg' width='100%'>

**原理解析可参考我知乎专栏**

- [推荐算法专栏](https://www.zhihu.com/column/c_1330637706267734016)

- 原理学习可按照下图的发展顺序进行，也可直接参照专栏文章发表顺序

- 文章会以通俗易懂的方式深入浅出每个算法的原理

**代码实现可参考本仓库代码** 

- 原理结合代码食用更佳，掌握算法的最好方式就是用代码撸它

- 每个模型文件夹中的代码都可独立运行，文件之间没有复杂的函数依赖，对新手友好

**Tips:** 该仓库使用的代码均为TF2.0，如果你不熟悉该框架，可参考文档[**简单粗暴Tensorflow2.0**](https://tf.wiki/zh_hans/basic/models.html)

<div>
<img src='https://github.com/jc-LeeHub/Recommend-System-TF2.0/blob/master/image/%E6%8E%A8%E8%8D%90%E7%AE%97%E6%B3%95.png'>
</div>

## Models List

|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  CCPM  | [CIKM 2015][A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)             |
|  FNN   | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                    |
|  PNN   | [ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                   |
|              Wide & Deep               | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                 |
|                 DeepFM                 | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           |
|        Piece-wise Linear Model         | [arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)                                 |
|          Deep & Cross Network          | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |
|   AFM  | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
|   NFM  | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |
|   xDeepFM   | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |
|   DIN  | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)     
|   AutoInt   | [CIKM 2019][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              ||
|   DIEN  | [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                            |
|   FwFM  | [WWW 2018][Field-weighted Factorization Machines for Click-Through Rate Prediction in Display Advertising](https://arxiv.org/pdf/1806.03514.pdf)                |
|   ONN   | [arxiv 2019][Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                |
|   FGCNN | [WWW 2019][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction ](https://arxiv.org/pdf/1904.04447)                             |
|   DSIN  | [IJCAI 2019][Deep Session Interest Network for Click-Through Rate Prediction ](https://arxiv.org/abs/1905.06482)                                                |
|   FiBiNET | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |
|   FLEN    | [arxiv 2019][FLEN: Leveraging Field for Scalable CTR Prediction](https://arxiv.org/pdf/1911.04690.pdf)   |
|   DCN V2  | [arxiv 2020][DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535)   |

## Citation

- 论文列表引用于浅梦 Weichen Shen. (2017). DeepCTR: Easy-to-use,Modular and Extendible package of deep-learning based CTR models. https://github.com/shenweichen/deepctr. 感谢整理！

## About

知乎：[予以初始](https://www.zhihu.com/people/yu-yi-chu-shi)

CSDN: [予以初始](https://blog.csdn.net/weixin_45658131?spm=1000.2115.3001.5343)

Website: [HomePage](https://jc-leehub.github.io/)

wechat ID: Liii00061333

<img src='https://github.com/jc-LeeHub/Recommend-System-TF2.0/blob/master/image/IMG_4498(20210107-214348).JPG' width='20%'>

