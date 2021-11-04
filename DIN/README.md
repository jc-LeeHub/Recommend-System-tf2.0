## DIN （Deep Interest Network for Click-Through Rate Prediction）

![DIN模型结构](https://github.com/jc-LeeHub/Recommend-System-tf2.0/blob/master/image/DIN_model.png)

### 1 原理

可参考我的知乎文章 [推荐算法(九)——阿里经典深度兴趣网络 DIN](https://zhuanlan.zhihu.com/p/429433768)

### 2 代码结构

```shell
├── layer.py  
│   ├── Attention   # 注意力机制的定义
│   ├── Dice        # Dice 激活函数的定义
├── model.py  
│   ├── DIN         # DIN 模型搭建
```

### 3 实验数据

选择 [Criteo](https://github.com/jc-LeeHub/Recommend-System-TF2.0/blob/master/Data/train.txt) 作为实验数据集，这里只使用部分样本(2000个)进行训练。

**样本字段:**

I1~I13：数值特征

C14~C39：类别特征

**预处理：**
1. 对数值特征 I1~I13 的缺失值进行填充, 然后进行归一化处理；
2. 对类别特征 C14~C39 进行 onehot 编码, 转换成稀疏的数值特征；
3. 将数值特征与类别特征用字典保存为 feature_columns；
3. 切分数据集，返回 feature_columns, (train_X, train_y), (test_X, test_y)。

### 4 实验结果

没有找到对应的训练集...
