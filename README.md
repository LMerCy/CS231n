
# cs231n
## 整理了cs231n作业
## 作业１内容：
设计一个线性分类器基类，完成knn、svm、softmax继承该基类。基类中包含通用的训练、预测函数，损失函数的计算由各个线性分类器重载
　

##  1.1 KNN:
    1. 完成“训练”（存储）模块；
    2. 用三种不同的计算方式方式实现预测：两次循环，一次循环，向量化;
    3. 利用交叉验证的方式去调整Ｋ值的选择。
##  1.2 SVM
    1. 实现前向计算loss和grad（循环法和向量化）;
    2. 在基类中实现SGD，SVM类继承该基类；
    3. 在基类中完成预测功能。
## 1.3 Softmax
    1. 实现前向计算loss（softmax+交叉熵）和grad(向量化)；
    2. 交叉验证调参：正则项的权重和学习速率；
    3. 分别可视化SVM和Softmax的权重，来直观对比两者权重的不同。
## 1.4 两层FC神经网络（FC-Relu-FC-Softmax）
    1. 实现两层FC神经网络的前向（loss计算、梯度计算）、训练（SGD）和预测。
## 1.5 图像特征提取
    1. 结合Hog和color histogram提取图像特征，再通过SVM/NN来对图像进行训练、预测。

#### 总结：重点在于理解整个代码的结构，各个模型的梯度推导和代码实现、以及通过拓展阅读对各个细节的深入理解（Softmax的含义，l1和l2的区别，传统线性分类器的权重的可视化结果...）。

## 作业2内容：
作业2内容很多，主要内容是模块化构建任意深度的神经网络（包括CNN），并在过程中了解CNN的各个细节。	
##   2.1 模块化构建神经网络
	1. 和1.4相同搭建一个FC神经网络,但是此处要通过模块化（计算图）的方式去构建，即分别构建各种功能的层(FC,Relu,Softmanx,SVM),包括前向计算loss和后向计算梯度，然后通过累加各层来实现构建NN。
	2. 通过通用的Solver来训练1中的神经网络。
##   2.2 构建任意深度神经网络
	网络的结构：{affine - relu} x (L - 1) - affine - softmax，L是构建网络时的参数，循环体内各隐层的节点数亦可设置。 

##   2.3 实现优化算法
    实现SGD+Momentum,Adgrad,RMSProp和Adam四种优化算法。
##   2.4 实现Batch Normalization和Layer Normalization层
    实现BN的前向和后向计算，并将其并添加到之前的网络中：
    新的网络结构为：{affine - [batch/layer norm] - relu} x (L - 1) - affine - softmax
##   2.5 实现inverted Dropout
	1. 实现Dropout的前向和后向计算，并添加到之前的网络中：
	   新的网络结构为：{affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax。
	2. 设计实验验证不同P值时的模型防止过拟合的效果。
##  2.6 实现CNN
	1. 实现卷积层前向loss计算和后向梯度计算；
	2. 通过卷积层实现图片的灰度转化，和边缘提取；
	3. 实现Maxpooling层的前向计算和后向计算；
        4. 利用已有的计算层构建CNN;
	5. 实现spatial batch normalization前向计算和后向计算;
	6. 实现Group Normalization前向计算和后向计算。
#### 总结：作业2难度最大的地方在于如何模块化构建神经网络，但实际上作业本身已经为我们构建好总体的框架，所以仔细阅读这个框架也会有很大的收获。主要的工作在于理解各个层的作用以及优化算法的原理。推荐论文：

[1][Sergey Ioffe and Christian Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", ICML 2015](https://arxiv.org/abs/1502.03167)

[2][Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton. "Layer Normalization." stat 1050 (2016): 21](https://arxiv.org/pdf/1607.06450.pdf)

[3][Wu, Yuxin, and Kaiming He. "Group Normalization." arXiv preprint arXiv:1803.08494 (2018)](https://arxiv.org/abs/1803.08494)

[4][N. Dalal and B. Triggs. Histograms of oriented gradients for human detection. In Computer Vision and Pattern Recognition (CVPR), 2005](https://ieeexplore.ieee.org/abstract/document/1467360/)

[5]Tijmen Tieleman and Geoffrey Hinton. "Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude." COURSERA: Neural Networks for Machine Learning 4 (2012)

[6][Diederik Kingma and Jimmy Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015](https://arxiv.org/abs/1412.6980)

            
