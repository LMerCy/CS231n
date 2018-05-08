# cs231n
整理了cs231n作业
作业１内容：
  0.设计一个线性分类器基类，完成KNN、SVM、Softmax继承该基类。基类中包含通用的训练、预测函数，损失函数的计算由各个线性分类器重载
　1.KNN:
	  完成“训练”（存储）模块；
	  用三种不同的计算方式方式实现预测：两次循环，一次循环，向量化；
      利用交叉验证的方式去调整Ｋ值的选择。
  2.SVM
	   实现前向计算loss和grad（循环法和向量化）;
	   在基类中实现SGD，SVM类继承该基类；
	   在基类中完成预测功能。
  3.Softmax
       实现前向计算loss（softmax+交叉熵）和grad(向量化)；
	   交叉验证调参：正则项的权重和学习速率；
	   分别可视化SVM和Softmax的权重，来直观对比两者权重的不同。
  4.两层FC神经网络（FC-Relu-FC-Softmax）
       实现两层FC神经网络的前向（loss计算、梯度计算）、训练（SGD）和预测。
  5.图像特征提取
       结合Hog和color histogram提取图像特征，在通过SVM/NN来对图像进行训练、预测。
总结：重点在于理解整个代码的结构，各个模型的梯度推导和代码实现、以及通过拓展阅读对各个细节的深入理解（Softmax的含义，l1和l2的区别，传统线性分类器的权重的可视化结果...）。
作业2内容：
   作业2内容很多，主要目的是解决如何构建一个任意深度CNN的问题，并在过程中了解CNN的各个细节。	
   1.和作业1的4一样构建一个NN,但是此处要通过模块化（计算图）的方式去构建，即分别构建各种功能的层,包括Fc、Relu、
   （分别构建前向计算loss和后向计算梯度），然后通过累加各层来实现构建NN。
   2.通过通用的Solver来训练1中的NN。
   3.构建一个任意深度的神经网络，网络的结构：{affine - relu} x (L - 1) - affine - softmax，
      实现任意构建任意深度的该循环体，以及循环体内各隐层的节点数。 
   4.实现SGD+Momentum,Adgrad,RMSProp和Adam四种优化算法。
   5.实现Batch Normalization和Layer Normalization，并添加到之前的网络中：{affine - [batch/layer norm] - relu} x (L - 1) - affine - softmax
   6.实现inverted Dropout，并添加到之前的网络中：{affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax。设计实验验证不同P值时的模型防止过拟合的效果。
  !7.实现CNN
	    利用循环的方法实现卷积层前向（loss）和后向的计算；
		通过卷积层实现将彩色图片转化为灰度图片，和边缘提取；
		实现Maxpooling层；
		利用已有的计算层构建CNN;
		实现spatial batch normalization;
		实现Group Normalization。
 总结：
            
