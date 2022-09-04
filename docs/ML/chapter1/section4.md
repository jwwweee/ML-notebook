# 1.4 对数几率回归 (Logistic Regression) 

## 1.4.1 线性回归的分类任务(Classification Task with Linear Regression)

利用线性回归与“最小二乘法”能解决一些简单的回归问题。对于分类问题，很自然的就能联想到其中一种方法，就是直接把就是把分类的类别作为回归任务中的标签并且对其进行预测。比如，一个二维特征的二分类任务我们就可以分别把两个类别(Class)标记成“$1$”和“$-1$”，然后让模型去找到一条理想的分界线。如图1.4.1所示。


<center>
    <img  src="ML\ML_figure\linear_classify.png" width="40%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.4.1 Binary Classification by Linear Regression</font>
    </div>
</center>


如果能够根据数据集拟合出一条完美的分界线，那么就会像图中所示得到一条为$w_1 x_{1}+w_2 x_2 +b =0$的直线。在直线的左下上方为类别$y=-1$的输入，在绿线的右下方为类别$y=1$的输入。具体做法是如果一个新的数据点的$y$接近$-1$那么就把这个数据点预测为类别“$-1$”，反之就预测为“类别1”。这样似乎看起来很合理，那么如果“类别1”的输入数据分布变成了图1.4.2所示的样子。

<center>
    <img  src="ML\ML_figure\linear_classify2.png" width="37%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.4.2 Binary Classification by Linear Regression Case 2</font>
    </div>
</center>

从图中可以发现，假如“类别1”的输入数据的分布出现了图中右下角的那些值，虽然说这些值非常偏远，但是它们仍然符合在$y>0$的范畴内。假如出现了这些偏远值，那么在模型拟合的时候就会被这些值影响到，进而出现图中灰线转变为黑线的趋势，而这样的模型拟合出来的分割线并不是我们最终想要的。

由于分类任务的标签都是离散值，而回归任务的标签是连续值，直接用线性回归解决分类任务显然是不合适的。解决分类任务的一个常规做法，是将数据的“类别”转换成“概率”。更具体解释是，我们不在专注于去识别一个数据样本的类别是什么，而是根据一个数据样本的特征$X$去求解这一样本会是某一类别的概率$P(C)$是多少。因此，我们需要使用概率模型去解决分类问题。

## 1.4.2 概率模型：抽小球 (Probability Model: A simple case)

为了更好的理解概率模型解决分类问题的过程，下面将从一个从不同盒子选取小球的例子进行说明。图1.4.3描述的是从两个盒子里拿小球的场景：

<center>
    <img  src="ML\ML_figure\pick_ball.png" width="70%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.4.3 Picking Balls</font>
    </div>
</center>

假设我们人为规定选中蓝色盒子的概率是$\frac{2}{3}$，选中绿色盒子的概率是$\frac{1}{3}$；然后蓝盒子里面的蓝球占了$\frac{4}{5}$，红球占了$\frac{1}{5}$；在红盒子里的蓝球占了$\frac{2}{5}$，红球占了$\frac{3}{5}$。现在问题是：一个蓝色的球是从蓝盒子选出来的，它的概率是多少？

因为我们已知选出来的是蓝色小球，而现在又问他从蓝盒子抽出来的概率，所以这是一个已知结果反推这个事情发生原因（条件）的“后验概率”，因此需要用到贝叶斯公式去计算已知我们拿的小球是蓝色的，且蓝色小球是从红盒子拿出来的概率是多少：

$$
\mathrm{P}\left(\mathrm{B}_{1} \mid \text { Blue }\right)=\frac{P\left(\text { Blue } \mid B_{1}\right) P\left(B_{1}\right)}{P\left(\text { Blue } \mid B_{1}\right) P\left(B_{1}\right)+P\left(\text { Blue } \mid B_{2}\right) P\left(B_{2}\right)}
\tag{1.4.1}
$$

## 1.4.3 Sigmoid函数与对数几率回归 (Sigmoid Function and Logistic Regression)

假设数据集中某一样本为$(x,y)$，现在把盒子换成“类别”(Class)，即$y$，小球换成是模型的输入数据，即$x$。那么已知输入数据$x$的特征（蓝球），求解是“类别1”（蓝盒子）$y=1$的概率问题，就类似于抽取小球概率问题。那么求解类别后验概率的贝叶斯公式就可以改写成：

$$
P\left(y={1} \mid x\right)=\frac{P\left(x \mid y={1}\right) P\left(y={1}\right)}{P\left(x \mid y={1}\right) P\left(y={1}\right)+P\left(x \mid y={0}\right) P\left(y={0}\right)}
\tag{1.4.2}
$$
其中$y=0$是“类别2”（小球从红盒子拿出来的事件）。

接着式1.4.2在分号上下同时除以分子$P\left(x \mid y={1}\right) P\left(y={1}\right)$，再令$z=\mathrm{ln}\frac{P\left(x \mid y={1}\right) P\left(y={1}\right)}{P\left(x \mid y={0}\right) P\left(y={0}\right)}$（注意对数分数项$z$在下式的第二步中为它的倒数，即$\frac{1}{z}$），然后有：

$$
\begin{aligned}
	P\left(y={1} \mid x\right)&=\frac{P\left(x \mid y={1}\right) P\left(y={1}\right)}{P\left(x \mid y={1}\right) P\left(y={1}\right)+P\left(x \mid y={0}\right) P\left(y={0}\right)}\\
	&=\frac{1}{1+\frac{P\left(x \mid y={0}\right) P\left(y={0}\right)}{P\left(x \mid y={1}\right) P\left(y={1}\right)}}\\
	&=\frac{1}{1+e^{-z}}
\end{aligned}
\tag{1.4.3}
$$
最后式1.4.3变形后的结果被称为Logistic函数，也被称为Sigmoid函数，该函数的图像如图1.4.4所示。

<center>
    <img  src="ML\ML_figure\sigmoid.png" width="40%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.4.4 Sigmoid Function</font>
    </div>
</center>

如果$P\left(y={1} \mid x\right)>0.5$则被规定为输出是“类别1”，反之则为“类别2”。对数几率回归的初衷是要让线性回归变成一个输出概率的模型，即令$z=\sum_{i=1}^m{{x_iw}+b}$。最终，对数几率回归模型可以定义为：

$$
\begin{aligned}
h(z)&=\frac{1}{1+e^{-z}}\\
h(x)&=\frac{1}{1+e^{-(\sum_{i=1}^m{{x_iw}+b})}}
\end{aligned}
\tag{1.4.4}
$$

相比于普通线性回归模型，在线性回归的基础上再外嵌一个Sigmoid函数之后，模型的最终输出范围是$(0,1)$的值，也就是我们所要的概率。另外，对数几率回归虽然虽然在名字上有“回归”两个字，但是我们可以看到对数几率回归其实是一种用来做分类的算法，对数几率回归的概率模型最后求解出来的划分线在图中也是一条直线，因此虽然我们对线性回归使用了非线性函数进行非线性化求得其输出的概率，但是本质上它还是线性的，因此也叫做“广义线性回归”。

## 1.4.4 交叉熵函数 (Cross Entropy Loss Function)

在分类问题中类别的概率$P\left(y={1} \mid x\right)$实际上是由权重参数$w,b$所影响为前提条件的，因此分类的概率还可以写为$P\left(y={1} \mid x ;w,b\right)$和$h_{w,b}(z)$。在线性回归中是要找出合适的参数$w,b$，在对数几率回归中同样需要找出合适的参数$w,b$来让模型对于新数据的分类判断更加准确。

最后，我们希望找到一组最优的权重参数$w$和$b$来让模型能够让每一个样本都正确分类，即模型正确分类的概率最大化。并且，我们希望能找到一个关于权重参数$w$和$b$的函数$l(w,b)$来解释成功分类的概率最大化，该函数称为“似然函数”。

我们希望利用似然函数解决概率最大化的参数估计问题，就要使用最大似然估计法(Maximum Likelihood Estimation)。假设训练数据集为$D=\{(x_1,y_{1}=1),(x_2,y_{2}=1),(x_3,y_{3}=0),\dots,(x_m,y_{m}=1)\}$，最大似然估计的步骤如下：

1. 基于“类别”的后验概率$P\left(y_{i} \mid x_i;w,b\right) =h_{w,b}(x_i)$，可以写出似然函数$l(w,b)$：

$$
\begin{aligned}
l(w, b)&=\prod_{i=1}^{m} P\left(y_{i} \mid x_i  ;w,b \right) \\
&=h_{w, b}\left(x_{1}\right) h_{w, b}\left(x_{2}\right)h_{w, b}\left(x_{3}\right) \cdots h_{w, b}\left(x_{m}\right)
\end{aligned}
\tag{1.4.5}
$$

其中，似然函数$l(w, b)$中为概率的乘积，意为每个样本正确分类的概率都是同时发生的。如同之前所说，我们希望的是这个同时发生的概率达到最大化，即$w^*,b^{*}=\mathrm{arg}\, \mathrm{max}\, l(w,b)$

2. 将式1.4.5中的“乘法”变成“加法”。具体做法是对式子中的每一项添加负号再取对数，那么最终求解方程最大值$w^*,b^{*}=\mathrm{arg}\, \mathrm{max}\, l(w,b)$就会变成求最小值$w^*,b^{*}=\mathrm{arg}\, \mathrm{max}\, -\mathrm{ln}(l(w,b))$。其中$ -\mathrm{ln}(\mathrm{lost}(w,b))$可以展开为：

$$
\begin{aligned}
\mathrm{ln}(l(w, b))&=-\sum_{i=1}^m \mathrm{ln}h_{w, b}\left(x_{i}\right)\\
&=-\left(\mathrm{ln}h_{w, b}\left(x_{1}\right)+\mathrm{ln}h_{w, b}\left(x_{2}\right)+\mathrm{ln}h_{w, b}\left(x_{3}\right) +,\cdots, +\mathrm{ln}h_{w, b}\left(x_{m}\right)\right)
\end{aligned}
\tag{1.4.6}
$$

其中等式取对数有两个原因，一是方便之后优化过程的求导操作（将乘法转变成加法），二是为了防止计算机计算连乘概率值的过程中出现浮点数下溢（即计算机无法显示太小的浮点数）。\\

事实上，最大似然估计是要估计出一个已知概率分布函数最有可能的参数，而在对数几率模型中，这个概率分布函数是关于权重参数$w$和$b$（为前提条件）的。由于二分类的结果是非“0”即“1”的，即符合伯努利分布（伯努利试验是只有两种可能结果的单次随机试验）：

$$
f(x)=p^x(1-p)^{1-x} \begin{cases}p & x=1 \\ 1-p & x=0\end{cases}
\tag{1.4.7}
$$

其中，$x$为随机变量，即对应分类任务的标签$y$；$p$对应的是模型成功分类的概率（模型输出值），即$h_{w,b}(x)$。因此，我们构建的似然函数只需优化出这一个符合伯努利分布的概率分布函数的参数值，即为我们要求解的权重参数$w$和$b$。\\

为了将似然函数中的项写成符合伯努利分布概率函数的形式，基于“类别1”的后验概率$h_{w,b}(x)=P\left(y={1} \mid x ;w,b\right)$，那么“类别2”的后验概率就可以表示为$P\left(y={0} \mid x; w,b\right)=1-h_{w,b}(x)$。最后，根据伯努利分布概率函数的定义代入标签$y$与模型输出$h_{w,b}(x)$，则似然函数（式1.4.5）可以改写为：

$$
\begin{aligned}
l(w, b)&=\prod_{i=1}^{m} P\left(y_{i} \mid x_i  ;w,b \right) \\
&=\prod_{i=1}^{m} h_{w, b}\left(x_{i}\right) \\
&=\prod_{i=1}^{m} h_{w,b}(x_i)^y_{i}(1-h_{w,b}(x_i))^{1-y_i}
\end{aligned}
\tag{1.4.8}
$$

同样地，对式1.4.8中最后一步的每一项取对数、添加负号，则式1.4.6中每一项概率可以改写为：

$$
-\mathrm{ln}h_{w, b}\left(x_{i}\right)=-[y_{i}\mathrm{ln}h(x_{i})+(1-y_{i})\mathrm{ln}(1-h(x_{i}))]
\tag{1.4.9}
$$

<center>
    <img  src="ML\ML_figure\transformation.png" width="60%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.4.5 Transformation of Terms</font>
    </div>
</center>

可以发现，式1.4.9很巧妙地选择了不同类别的概率，如图1.4.5所示。基于改写后的概率项，式1.4.8可以写为：

$$
\begin{aligned}
\mathrm{lost}(h_{w, b}(x),y)&=-\frac{1}{m}\sum_{i=1}^{m}l(w, b)\\
&=-\frac{1}{m}\sum_{i=1}^{m}[y_{i}\mathrm{ln}(h_{w, b}(x_{i}))+(1-y_{i})\mathrm{ln}(1-h_{w, b}(x_{i}))]
\end{aligned}
\tag{1.4.10}
$$

式1.4.10即为对数几率回归的损失函数，也称为交叉熵函数(Cross Entropy Function)。交叉熵函数的图像如图1.4.6所示，上式也可以写作：

$$
\mathrm{lost}(h_{w, b}(x),y)=\left\{
\begin{aligned}
-\mathrm{ln}(h_{w, b}(x)) & , & if \quad  y=1, \\
-\mathrm{ln}(1-h_{w, b}(x)) & , & if \quad  y=0.
\end{aligned}
\right.
\tag{1.4.11}
$$

<center>
    <img  src="ML\ML_figure\cross_entropy.png" width="70%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.4.6 Cross Entropy Function</font>
    </div>
</center>

然而，我们在介绍线性回归的时候提到，损失函数是一个计算模型输出$f(x)$与样本标签$y$误差的函数，而交叉熵函数似乎并不符合这个规则。这里就需要引出一个概念：相对熵 (Relative Entropy)，又称KL散度 (Kullback-Leibler divergence )，KL散度可以用来衡量两个概率分布之间的差异。给定概率分布$P$与概率分布$Q$，它们的信息量分别表示为与$f_P(p_i)$与$f_Q(q_i)$（定义参考决策树信息熵部分），那么它们的KL散度可以定义为：

$$
\begin{aligned}
D_{\text{KL}}(P \| Q)&=\sum_{c=1}^C p_c \cdot\left(f_Q\left(q_c\right)-f_P\left(p_c\right)\right) \\
&=\sum_{c=1}^C p_c \cdot\left(\left(-\log _2 q_c\right)-\left(-\log _2 p_c\right)\right) \\
&=\sum_{c=1}^C p_c \cdot\left(-\log _2 q_c\right)-\sum_{c=1}^C p_c \cdot\left(-\log _2 p_c\right)
\end{aligned}
\tag{1.4.12}
$$

其中C为随机变量的个数。当KL散度越接近0，两个概率分布之间的差异越小。我们要做的就是希望预测输出值的分布与样本标签的分布尽可能的接近，即$\left(f_Q\left(q_c\right)-f_P\left(p_c\right)\right)$趋于0。这种计算“误差”的形式就符合我们对损失函数定义的要求了，因此最小化KL散度等价于最大化似然函数（最大似然估计）。

可以看到，式1.4.12中的最后一步同时出现了$p$的信息熵与$p$的交叉熵两项（定义参见交叉熵信息熵），我们希望的是$p$的交叉熵尽可能与它信息熵的差越小越好。根据吉布斯不等式(Gibbs-Ungleichung)，若$\sum_{c=1}^C p_c=\sum_{c=1}^C q_c=1$，且$p_c, q_c \in(0,1]$，则有$-\sum_{c=1}^n p_c \log p_c \leq-\sum_{c=1}^n p_c \log q_c$。因此，$p$的交叉熵始终大于其信息熵，我们只需要让交叉熵作为损失函数让它最小化就可以了。到这里我们就介绍了交叉熵函数的推导过程和定义，以及解释了为什么交叉熵函数可以作为损失函数。

3. 对交叉熵损失函数进行求导，让关于参数$w,b$的损失函数$\mathrm{lost}(w,b)$的偏导数为0。类似地，对交叉熵函数的求导优化过程使用梯度下降法，其迭代过程为：

$$
w:= w-\eta\frac{\partial \mathrm{lost}(w,b)}{\partial w}
\tag{1.4.13}
$$

$$
\frac{\partial \mathrm{lost}(w,b)}{\partial w} = \frac{1}{m}\sum_{i=1}^m({f(x_{i})}-y_{i})x_{i}
\tag{1.4.14}
$$

公式1.4.14表明，$\mathrm{lost}(w,b)$的求导结果和MSE的求导结果完全一致。在Logistic回归中不使用MSE函数而选择交叉熵h函数是因为使用MSE函数会出现梯度消失现象，以及MSE求导后的结果为非凸函数，不易求解并且会得到局部最优。