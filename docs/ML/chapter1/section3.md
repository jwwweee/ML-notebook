# 1.3 线性回归 (Linear Regression)

## 1.3.1 一维线性回归 (Simple Liner Regression)

线性回归(Linear Regression)是监督学习中最基础的模型，用于解决回归任务。该模型通过找出数据集特征$x$与标签$y$之间的线性关系，拟合出一个方程来对新数据进行预测。假设一个单变量数据集的特征为$x$，标签为$y$，那么一个关于单变量$x$的一维线性回归(Simple Linear Regression)的方程$f(x)$可以定义为：

$$
f(x) = wx+b
\tag{1.3.1}
$$

其中$w$为线性回归方程的权重参数(Paramater)，$b$为偏置(Bias)。偏置的作用是为了更好的拟合数据以及调整模型的位置。

## 1.3.2 多元线性回归 (Multivariate Linear Regression)

若一组数据集为多特征（多变量）数据集，那么就可以将一维线性回归推广成多元线性回归(Multivariate Linear Regression)。假设一个多维特征数据集为$D=\{(x_1, x_2, x_3,\dots,x_m), y\}$，$y$为这些特征的标签，那么多元线性回归定义为：

$$
\begin{aligned}
f(x_i) &= {w_1}{x_1}+{w_2}{x_2}+{w_3}{x_3}+\dots+{w_n}{x_n}+b \\ &= \sum_{j=1}^n{{x_jw_j}+b}
\end{aligned}
\tag{1.3.2}
$$




其中一组特征$(x_1,x_2,\dots,x_n)$与之对应的标签$y$的组合称为一个样本。然而在实际的任务中，一个完整的数据集会有若干个样本。例如，房价预测中会有很多不同的房子样本与其对应的价格，在时间序列任务中也会有多个不同时间点记录的数据。为了更好的表示这些数据，这里可以用矩阵$X$来表示每个特征$x_{i,j}$，其中$i=1,2,\dots,m$为样本个数，$j=1,2,\dots,n$为特征个数（数据维度）。对于每个样本的特征集$(x_{i,1},x_{i,2},\dots,x_{i,n})$有着对于的标签$y_i$，标签向量可以写作$\boldsymbol{y}=(y_1, y_2,\dots,y_m)$。最终一个完整的数据集$D$可以表示为$D=\{X,\boldsymbol{y}\}$。其中$X$的展开为：

$$
X={
\begin{pmatrix}
 x_{1,1} &  x_{1,2} &  x_{1,3} & \dots &  x_{1,n} \\ 
 x_{2,1} &  x_{2,2} &  x_{2,3} & \dots &  x_{2,n} \\ 
 \vdots & \vdots & \vdots & \ddots & \vdots \\
 x_{m,1} &  x_{m,2} &  x_{m,3} & \dots &  x_{m,n} \\ 
 \end{pmatrix}}
\tag{1.3.3}
$$

与之对应地，权重$w_j$也可以写成向量形式$\boldsymbol{w}$（式1.3.4），权重矩阵的最后一项是偏置向量$b$。为了让输入$X$与包含偏置的权重参数集$\boldsymbol{w}$相乘，因此需要在$X$的每一行的最后加多一项“$1$”，如式1.3.5。

$$
\boldsymbol{w}={
\begin{pmatrix}
 w_1  \\ 
 w_2  \\ 
 \vdots  \\
 w_n  \\ 
 b    \\
 \end{pmatrix}}
\tag{1.3.4}
$$

$$
X={
\begin{pmatrix}
 x_{1,1} &  x_{1,2} &  x_{1,3} & \dots &  x_{1,n} & 1 \\ 
 x_{2,1} &  x_{2,2} &  x_{2,3} & \dots &  x_{2,n} & 1 \\ 
 \vdots & \vdots & \vdots & \ddots & \vdots &  \vdots  \\
 x_{m,1} &  x_{m,2} &  x_{m,3} & \dots &  x_{m,n} & 1 \\ 
 \end{pmatrix}}
\tag{1.3.5}
$$

最后线性模型的矩阵形式的方程为：

$$
f(X)=X\boldsymbol{w}
\tag{1.3.6}
$$

## 1.3.3 损失函数 (Loss Function)

为了让模型去尽可能地拟合实际数据，那么就要找到合适的参数$w$与输入$X$相乘，从而让模型输出值$f(x_o)$与实际标签$y_o$的误差尽可能最小，即$|f(x_o)-y_o|$，但由于绝对值在定义域上不是全程可微的，因此我们需要将误差函数改写成$(f(x_o)-y_o)^2$。类似这样计算模型输出值$f(x_o)$与实际标签$y_o$误差的函数被称为损失函数(Loss Function)，回归任务中使用最广泛的损失函数为均方误差(Mean Square Error)，可以写作：

$$
\mathrm{lost}(w,b) = \frac{1}{2m}\sum_{i=1}^m({f(x_i)}-y_i)^2
\tag{1.3.7}
$$

其中$\frac{1}{2}$项是为了方便之后消去平方求导之后的2，$\frac{1}{m}$项是为了计算误差的全样本平均。

在一维线性回归里，不同的权重参数$w$(也是斜率)可以拟合出不同的直线。关于权重参数$w$的损失函数$\mathrm{lost}(w)$的图像如图1.3.1所示（若推广到二维特征的时候，图像会变成一个碗状。），求解损失函数的最小值也就是让点$(w_o,\mathrm{lost}(w_o))$去到图1.3.1中抛物线的最底端。

求解$\mathrm{lost}(w)$的最小值实际上是一个优化问题，这个优化问题通过均方误差方程最小化求解模型参数$w$，该方法也叫“最小二乘法”(Least Square Method)。优化的过程所使用的算法通常有两种：数值微分法、梯度下降法。

<center>
    <img  src="ML\ML_figure\loss_function.png" width="35%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.3.1 MSE Loss Function</font>
    </div>
</center>

## 1.3.4 数值微分法 (Numerical Differential Method)

数值微分法(Numerical Differential Method)是直接通过计算矩阵伪逆来得到权重参数$W$的直接方法。我们的优化目标是最小化损失函数$\min \mathrm{lost}(w,b)$，由于$\mathrm{lost}(w,b)$为凸函数(Convex)，因此当$\frac{\partial \mathrm{lost}(w,b)}{\partial w}=0$时，$\mathrm{lost}(w)$最小（求开口向上的二次函数求最小值方法是让其导数为0）。

当特征矩阵为$X$，权重向量为$\boldsymbol{w}$，标签向量为$\boldsymbol{y}$时，矩阵形式下的损失函数可以改写成以下形式：
\begin{equation}
\mathrm{lost}(\boldsymbol{w})=(X\boldsymbol{w}-\boldsymbol{y})^{T}(X\boldsymbol{w}-\boldsymbol{y})
\end{equation}

接下来对矩阵形式的损失函数$\mathrm{lost}(w,b)$进行求导。损失函数$\mathrm{lost}(\boldsymbol{w})$是一个关于$\boldsymbol{w}$的复合函数，其求导过程如下：首先设$z=(X \boldsymbol{w}-y)$，那么就有$\mathrm{lost}(\boldsymbol{w})=z^{T}z$。接着假设我们最后的求导结果$\frac{\partial \mathrm{lost}(\boldsymbol{w})}{\partial \boldsymbol{w}}$的大小为$m \times 1$的向量，$\frac{\partial \mathrm{lost}(\boldsymbol{w})}{\partial z}$的大小为$n \times m$，$\frac{\partial z}{\partial \boldsymbol{w}}$的大小为$1 \times nwe$。由于是矩阵相乘，矩阵大小$(n,m) \times (n,1)$并不能得到$(m,1)$的矩阵，因此需要让$\frac{\partial z}{\partial \boldsymbol{w}}$项转置变成${\frac{\partial z}{\partial \boldsymbol{w}}}^T$，其大小改变为$(1,m)$的矩阵，并且将它移动到$\frac{\partial \mathrm{lost}(\boldsymbol{w})}{\partial z}$项的前面，便可进行相乘，则有：

$$
\begin{aligned}
	\frac{\partial \mathrm{lost}(\boldsymbol{w})}{\partial \boldsymbol{w}}&={\frac{\partial z}{\partial \boldsymbol{w}}}^T \cdot \frac{\partial \mathrm{lost}(\boldsymbol{w})}{\partial z}\\
	&=X^{T}(2(X\boldsymbol{w}-y))\\
	&=2X^{T}(X\boldsymbol{w}-y)
\end{aligned}
\tag{1.3.8}
$$

然后令$\frac{\partial \mathrm{lost}(\boldsymbol{w},b)}{\partial \boldsymbol{w}}=0$，则$\boldsymbol{w}$的求解过程为：

$$
\begin{aligned}
2X^{T}(X\boldsymbol{w}-y)&=0 \\
2X^{T}X\boldsymbol{w}-2X^{T}y&=0 \\
2X^{T}X\boldsymbol{w}&=2X^{T}y \\
(X^{T}X)^{-1}X^{T}X\boldsymbol{w}&=(X^{T}X)^{-1}X^{T}y\\
\boldsymbol{w}&=(X^{T}X)^{-1}X^{T}y
\end{aligned}
\tag{1.3.9}
$$

假如当矩阵不存在逆矩阵即非满秩的时候可以求广义逆（伪逆）,但是数值微分法有个缺点，当特征数量非常多、样本量非常巨大的时候，矩阵求伪逆的运算过程会相当复杂，并且会消耗过多的运算资源。因此，在求解高复杂度模型的时候通常会使用梯度下降法。

## 1.3.5 梯度下降 (Gradient Descent)

梯度下降(Gradient Descent)是另一种更广泛使用的模型参数优化方法，该方法是利用迭代更新的方法对各项参数进行优化。假设模型中的其中一项权重参数为$w_i$、偏置为$b$，那么损失函数$\mathrm{lost}(w,b)$关于权重参数$w_i$与偏置$b$的更新公式就可以写作：

$$
\begin{aligned}
w&\gets w-\eta\frac{\partial \mathrm{lost}(w,b)}{\partial w}\\
b&\gets b-\eta\frac{\partial \mathrm{lost}(w,b)}{\partial b}
\end{aligned}
\tag{1.3.10}
$$

其中“$\gets$”是赋值符号。$\eta$是学习率(Learning Rate)，是一种需要人为设置的超参数(Hyperparameter)，它的作用是调整梯度下降时的步幅，通常会被设置为$0.1$到$1\times 10^-n$之间。另一方面，偏导项决定了权重参数更改的方向，因此偏导项和学习率共同决定了下降的步幅。如图1.3.2所示，设置不同的学习率会导致梯度下降的过程不同：
\begin{itemize}
	\item [1.] 当学习率$\eta$很小：梯度下降的过程会很慢。
	\item [2.] 当学习率$\eta$很大：梯度下降越过最低点，会导致模型不收敛甚至发散。
	\item [3.] 当学习率$\eta$大小合适时，梯度下降时导数会越来越小（切线越来越平缓），因此步幅越来越小。
\end{itemize}

<center>
    <img  src="ML\ML_figure\learning_rate.png" width="65%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.3.2 Different Learning Rates</font>
    </div>
</center>

梯度下降的更新公式反映的是权重参数$w$、偏置$b$和损失函数的值在梯度下降时变化的过程。更新公式的实质是利用权重参数本身减去学习率$\eta$与损失函数偏导的乘积项，权重参数更新与偏置更新的两个更新公式是同时进行的。

梯度的定义是方向导数汇总的向量。权重参数$w$始终指向最低点，而梯度下降是指函数沿着梯度方向变小（在多变量模型中则是多个维度的偏导数指向的方向）。在图像中梯度下降的具体过程如图1.3.3所示。假如损失函数的初始值在抛物线的左侧，这意味着学习率偏导乘积项恒负。那么当$w$减去恒负的学习率偏导乘积项，则$w$会持续增大。因此，图像中损失函数的值无限接近$0$的时候，$w$会一直向右移动。反之，当损失函数的初始值在抛物线的右侧，那么$w$就要一直减去一个恒正项，则会让$w$一直减小。最终图像中损失函数的值无限接近$0$的时候，$w$会一直向左移动。因此，无论$w$一开始在抛物线的左边还是右边，它最终都会往中间（抛物线最低点）的方向走。

<center>
    <img  src="ML\ML_figure\gradient_descent.png" width="50%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.3.3 Gradient Descent</font>
    </div>
</center>