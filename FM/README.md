## FM （Factorization Machines）

### 1 原理

可参考我的知乎文章 [推荐算法(一)——FM因式分解机](https://zhuanlan.zhihu.com/p/342803984)

### 2 代码结构

```shell
├── utils.py   
│   ├── create_criteo_dataset  # Criteo 数据预处理，返回划分好的训练集与验证集
├── model.py  
│   ├── FM_layer # FM 层的定义
│   ├── FM_layer # FM 模型的搭建
├── train.py 
│   ├── main     # 将处理好的数据输入 FM 模型进行训练，并评估结果
```

### 3 实验数据

选择 [Criteo](https://github.com/jc-LeeHub/Recommend-System-TF2.0/blob/master/Data/train.txt) 作为实验数据集，这里只使用部分样本(2000个)进行训练。

**样本字段:**

I1~I13：数值特征

C14~C39：类别特征

**预处理：**
1. 对数值特征 I1~I13 的缺失值进行填充, 然后进行归一化处理；
2. 对类别特征 C14~C39 进行 onehot 编码, 转换成稀疏的数值特征；
5. 切分数据集，返回 (train_X, train_y), (test_X, test_y)。

### 4 实验结果

模型准确率： Accuracy: 0.772

loss下降曲线：

<div align=center><img src="https://cdn.jsdelivr.net/gh/jc-LeeHub/Recommend-System-tf2.0@master/image/fm%E7%BB%93%E6%9E%9C.jpg" width="50%;" style="float:center"/></div>
