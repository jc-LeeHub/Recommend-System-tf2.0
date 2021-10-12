## PNN （Product-based neural networks for user response prediction）

![PNN 模型结构](https://cdn.jsdelivr.net/gh/jc-LeeHub/Recommend-System-tf2.0@master/image/PNN.png)

### 1 原理

可参考我的知乎文章 [PNN——显式特征交互模型](https://zhuanlan.zhihu.com/p/420679130)

### 2 代码结构

```shell
├── utils.py   
│   ├── create_criteo_dataset  # Criteo 数据预处理，返回划分好的训练集与验证集,以及记录数值特征与类别特征的字典
├── layer.py  
│   ├── Dense_layer        # 全连接层
│   ├── InnerProductLayer  # 内积交互层
│   ├── OuterProductLayer  # 外积交互层
│   ├── FGCNN_layer        # FGCNN 层(可忽略)
├── model.py  
│   ├── PNN          # PNN 模型搭建
├── train.py 
│   ├── main         # 将处理好的数据输入 PNN 模型进行训练，并评估结果
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

模型准确率： Accuracy: 0.7867

loss下降曲线：

<div align=center><img src="https://cdn.jsdelivr.net/gh/jc-LeeHub/Recommend-System-tf2.0@master/image/PNN_loss.png" width="50%;" style="float:center"/></div>
