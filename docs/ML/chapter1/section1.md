# 1 简介(Introduction)
## 1.1.1 概览 (Overview)
在Machine Learning中让机器进行学习的方法主要分为两大类，监督学习(Supervised Learning)和无监督学习(Unsupervised Learning)。监督学习的意思是建立一个数学模型，让模型(或称算法)通过学习输入数据集中的特征集(Feature)与之所对应标签(Label)的关系，然后在面对只有特征集的新数据时模型能够根据新数据的特征集准确判断出它们的标签。简而言之就是需要找到一个方程，让方程经过学习训练后能够准确地对新数据进行预测或者分类。

监督学习的任务主要可以归纳为两种，回归(Regression)与分类(Classification)。例如，通过房子的地段、面积、层数等特征对房价（标签）进行预测，就是一种回归任务，如图1.1.1。而根据商品的价格、评价等特征来判断商品是否应该推荐给用户，就是分类任务，如图1.1.2。

<center>
<div >
    <div style="float:left; width:40%; padding-left:100px">
        <img src="ML\ML_figure\regression_case.png" width="100%">
        <div style="margin-bottom: 50px; margin-top: 20px>
        <font size="3">图1.1.1 Regression Task</font>
        </div>
    </div>
    <div style="float:right; width:57%; padding-right:100px">
        <img src="ML\ML_figure\classification_case.png" width="100%">
        <div style="margin-bottom: 50px; margin-top: 20px>
        <font size="3">图1.2 Classification Task</font>
        </div>
    </div> 
</div>
</center>


## 1.1.2 监督学习的流程 (Procedure of Supervised Learning)

监督学习算法的流程如图1.1.3所示，流程主要分为三大部分：
1. 首先需要将原始数据集进行一系列的预处理工作(Data Pre-processing)，目的是为了让原始数据的格式变成模型所需要的输入格式。然后将预处理好的数据集分割成训练集(Training Set)和测试集(Test Set)。训练集的作用是用于训练模型，测试集的作用是用于测试已经训练好的模型是否能对新的数据集进行较好的拟合，检查模型是否存在过拟合(Overfitting)、欠拟合(Underfitting)的现象。
2. 把数据集切分为训练集和测试集之后，将训练集输入进模型进行训练，模型的输出$f(x)$与标签$y$经过损失函数(Loss Function)的计算求出误差(Error)，然后通过如梯度下降(Gradient Descent)等优化算法减小模型输出与标签的误差。由于损失函数是模型误差关于权重参数的方程，因此在减小误差的过程中，模型中的权重参数也会同时进行优化调整，参数优化的过程持续到模型误差收敛。
3. 将测试集输入进训练好的模型，并且得到最终的预测（分类）结果。

<center>
    <img  src="ML\ML_figure\supervised_learning.png" width="60%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.1.3 Classification Task</font>
    </div>
</center>