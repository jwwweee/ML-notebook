# 1.8 支持向量机 (Support Vector Machine)
## 1.8.1 间隔与支持向量 (Margin and Support Vector)

给定一个数据集$D=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots,\left(x_{m}, y_{m}\right)\right\}$，其中“类别1”与“类别2”分别为$y=1$和$y=-1$。如果要对这个数据集进行分类任务，最直观、最理想的方式就是数据集$D$在样本空间内找到一个可以完美划分不同类别的超平面（假设数据集是严格线性可分的，即存在一个超平面能完全将两类数据分开）。然而，样本空间中能够将类别划分开的超平面有很多（如图1.8.1所示）。凭借人为直观感受，毫无疑问介于类别中间的黑色超平面为最理想的划分超平面。这个理想的超平面以最大间隔将类别划分开，这样就能容忍训练数据的一些局部扰动和噪声而正确分类，即这个超平面对类别划分的鲁棒性最强，对新数据的泛化能力也最强。

<center>
    <img  src="ML\ML_figure\classify_margin.png" width="35%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.8.1 Margins for Classifying Different Classes</font>
    </div>
</center>

在样本空间里，划分超平面为一个由参数法向量$\mathbf{w}=(w_1, w_2,\dots,w_n)$与截距$b$线性方程共同决定的线性方程：

$$
{\mathbf{w}}^{\mathsf{T}}X+b = 0
\tag{1.8.1}
$$
将超平面记为$(\mathbf{w}, b)$，样本空间里的任意点$X$到超平面$(\mathbf{w}, b)$的距离可以写作：
$$
r = \frac{\mathbf{w}^{\mathsf{T}}X+b}{\Vert \mathbf{w} \Vert}
\tag{1.8.2}
$$
假如超平面$(\mathbf{w}, b)$能将数据类别正确划分，即对与数据样本$(x_i,y_i) \in D$，若$y_i=+1$，则${\mathbf{w}}^{\mathsf{T}}x_i+b >0$；若$y_i=-1$，则${\mathbf{w}}^{\mathsf{T}}x_i+b <0$。那么就有：

$$
\begin{cases}\mathbf{w}^{\mathrm{T}} x_i+b \geqslant+1, & y_i=+1 \\ \mathbf{w}^{\mathrm{T}} x_i+b \leqslant-1, & y_i=-1\end{cases}
\tag{1.8.3}
$$



如图1.8.2所示，为了找到最大间隔的划分超平面，我们要找到离划分超平面最近的两个数据样本（图中黄色虚线标记样本），使式1.8.3中的等号成立，这两个数据样本称为支持向量(Support Vector)。

<center>
    <img  src="ML\ML_figure\margin_support_vector.png" width="40%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.8.2 Margin and Support Vectors</font>
    </div>
</center>

然后，过这两个支持向量找到与划分超平面平行的超平面，即$\mathbf{w}^{\mathrm{T}} x_i+b=+1$与$\mathbf{w}^{\mathrm{T}} x_i+b=-1$。这两个超平面之间的距离称之为间隔(Margin)。那么两个支持向量到划分超平面距离之和就为这两个超平面之间的距离（间隔的大小），即：

$$
\rho = \frac{2}{\Vert \mathbf{w} \Vert}
\tag{1.8.4}
$$

我们的目标就是找到最大间隔，即找到满足式1.8.3中约束参数$\mathbf{w}$和$b$，使间隔$\rho$最大化：

$$
\begin{aligned}
\max _{\mathbf{w}, b} & \frac{2}{\|\mathbf{w}\|} \\
\text { s.t. } & y_i\left(\mathbf{w}^{\mathrm{T}} {x}_i+b\right) \geqslant 1, \quad i=1,2, \dots, m 
\end{aligned}
\tag{1.8.5}
$$

要使间隔$\rho$最大化，只需让${\Vert \mathbf{w} \Vert}^{-1}$最大化，等价于${\Vert \mathbf{w} \Vert}^2$，那么式1.8.6可以改写为：

$$
\begin{aligned}
\min _{\mathbf{w},b}{\frac{1}{2}{\Vert \mathbf{w}\Vert}^2} \\
\text { s.t. } & y_i\left(\mathbf{w}^{\mathrm{T}} {x}_i+b\right) \geqslant 1, \quad i=1,2, \dots, m 
\end{aligned}
\tag{1.8.6}
$$
其中，$\frac{1}{2}$项的作用与线性回归中损失函数的$\frac{1}{2}$项相同，作用皆为方便求导。上式即为支持向量机(Support Vector Machine)的基本型，通过求解上式即可得到最优超平面$\mathbf{w}^*$和$b^*$。

## 1.8.2 对偶问题 (Dual Problem)

我们的目的是通过求解式1.8.6来让间隔最大化，得到最优超平面。式1.8.6所述问题为原问题(Primal Problem)，原问题又称原线性规划问题，是指每一个线性规划的原始问题，每个原问题均可以转化为与其对称的对偶问题。原问题与对偶问题是相对的，二者为同类型的规划。 若原问题是极小化问题，那么对偶问题就是极大化问题。我们可以将原问题转化为对偶问题进行求解，然后得到原问题的最优解。将原问题转换为其对偶问题来求解的原因如下：1. 对偶问题更易求解，对偶问题只需优化一个变量$\mathbf{\alpha}$且约束条件更简单；2. 能更加自然地引入核函数，进而推广到非线性问题。

优化原问题1.8.6可以利用拉格朗日乘子法构造拉格朗日函数(Lagrange Function)得到其对偶问题(Dual Problem)。具体来说，对式1.8.6中每条约束添加拉格朗日乘子$\alpha_i \geqslant 0 (i=1,2,\dots,n)$，那么该问题的拉格朗日函数可以写作：

$$
L(\mathbf{w}, b, \mathbf{\alpha})=\frac{1}{2}\|\mathbf{w}\|^2+\sum_{i=1}^m \alpha_i\left(1-y_i\left(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i+b\right)\right)
\tag{1.8.7}
$$

令$L(\mathbf{w}, b, \mathbf{\alpha})$对$\mathbf{w}$和$b$的偏导为$0$，再带入求偏导后得到的$\mathbf{w}$，和考虑约束问题，式1.8.6的对偶问题可以写作（推导过程略）：

$$
\begin{aligned}
&\max _\alpha \sum_{i=1}^m \alpha_i-\frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \mathbf{x}_i^{\mathrm{T}} \mathbf{x}_j \\
&\text { s.t. } \quad \sum_{i=1}^m \alpha_i y_i=0 \\
&\qquad \alpha_i \geqslant 0, \quad i=1,2, \ldots, m .
\end{aligned}
\tag{1.8.8}
$$

最后，当解出$\mathbf{\alpha}$后，求出$\mathbf{w}$和$b$即可得到模型：

$$
\begin{aligned}
f(\mathbf{x}) &=\mathbf{w}^{\top} \mathbf{x}+b \\
&=\sum_{i=1}^m \alpha_i y_i \mathbf{x}_i^{\top} \mathbf{x}+b
\end{aligned}
\tag{1.8.9}
$$

由于式1.8.6中有不等式约束，因此上述过程需满足Karush-Kuhn-Tucker(KKT)条件，具体推导过程不在本笔记讨论范围内。

## 1.8.3 软间隔 (Soft Margin)
之前我们一直假设训练数据是严格线性可分的，即存在一个超平面能完全将两类数据分开。前面提到的支持向量机形式是要求所有样本都满足1.8.6中的条件约束，即所有数据样本都能被划分正确，该分划间隔称为硬间隔(Hard Margin)。但是在实际任务中这个假设往往不成立，例如图1.8.3中所示的数据，不是线性可分的。这样一旦出现一些偏差样本，就会导致分类结果准确率下降。

<center>
    <img  src="ML\ML_figure\soft_margin_data.png" width="40%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.8.3 Linearly Inseparable Dataset</font>
    </div>
</center>

解决该问题的一个办法是允许支持向量机在少量样本上出错，即将之前的硬间隔最大化条件放宽一点，为此引入软间隔(Soft Margin)的概念。即允许少量样本不满足约束$y_i\left(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i+b\right) \geqslant 1$，但我们又同时希望这些不满足约束条件的样本尽可能少。为了使不满足约束条件的数据样本尽可能少，我们需要在优化目标函数1.8.6中加入惩罚项，最常用的是铰链损失函数(Hinge Lost Function)：

$$
\mathrm{lost}_{\text {hinge}}(z)=\max (0,1-z)
\tag{1.8.10}
$$
若样本点满足约束条件，则损失值为$0$，否则为$1-z$，则优化目标函数1.8.6可以改写为：

$$
\min _{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2+C \sum_{i=1}^m \max \left(0,1-y_i\left(\mathbf{w}^{\mathrm{T}} \mathbf{x}_i+b\right)\right)
\tag{1.8.11}
$$
其中$C$为惩罚参数。接着，引入“松弛变量”(Slack Variable) $\xi_i \geqslant 0$，可以将式1.8.11改写为：

$$
\begin{aligned}
\min _{\mathbf{w}, b, \xi_i} \frac{1}{2}\|\mathbf{w}\|^2+C \sum_{i=1}^m \xi_i \\
\text { s.t. } y_i\left(x_i^T \mathbf{w}+b\right) \geq 1-\xi_i \\
\quad \xi_i \geqslant 0, i=1,2, \ldots n
\end{aligned}
\tag{1.8.12}
$$

上式所述即为软间隔支持向量机。与硬间隔支持向量机类似，需要使用拉格朗日函数将软间隔支持向量机的原问题转化为对偶问题，然后求偏导解出$\mathbf{w}$与$b$得到最后的模型（同样需要考虑满足KKT条件）。

## 1.8.4 非线性支持向量机：核技巧 (Non-linear Support Vector Machine: Kernel Trick)

在之前讨论的支持向量机训练过程中，我们默认数据样本$\boldsymbol{x}$在特征平面或空间中都是线性可分的，但是实际情况中遇到的数据样本往往都是线性不可分（非线性可分）的。对于在一个原始空间中线性不可分的数据样本$\boldsymbol{x}$，我们可以把它们映射到更高维的特征空间。如图1.8.4所示，我们把在原始二维空间线性不可分的数据映射到三维空间中，这样以来就一定存在一个超平面能让数据样本线性可分。

<center>
    <img  src="ML\ML_figure\kernel_trick.png" width="55%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.8.4 Non-linear Transformation with data samples</font>
    </div>
</center>

令$\phi(\boldsymbol{x})$表示为数据样本$\boldsymbol{x}$映射到更高维特征空间的特征向量，例如数据集$\boldsymbol{x}$在原始特征空间为$\boldsymbol{x}=[x_1,x_2]$，在映射到更高维的特征空间中可以表示为$\phi(\boldsymbol{x})=[1,x_1,x_2,x_1x_2]$。根据映射后的特征向量$\phi(\boldsymbol{x})$，那么在新的特征空间中划分超平面对于的模型可以表示为：

$$
f(\boldsymbol{x})=\boldsymbol{w}^{\mathrm{T}} \phi(\boldsymbol{x})+b
\tag{1.8.13}
$$
其中$\boldsymbol{w}$与$b$是模型的参数，那么非线性映射支持向量机的优化问题可以写作：


$$
\begin{aligned}
\min _{\boldsymbol{w}, b} & \frac{1}{2}\|\boldsymbol{w}\|^2 \\
\text { s.t. } & y_i\left(\boldsymbol{w}^{\mathrm{T}} \phi\left(\boldsymbol{x}_i\right)+b\right) \geqslant 1, \quad i=1,2, \ldots, m
\end{aligned}
\tag{1.8.14}
$$
其对偶问题为：
$$
\begin{aligned}
\max _\alpha & \sum_{i=1}^m \alpha_i-\frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j \phi\left(\boldsymbol{x}_i\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_j\right) \\
& \\
\text { s.t. } & \sum_{i=1}^m \alpha_i y_i=0 \\
& \alpha_i \geqslant 0, \quad i=1,2, \ldots, m .
\end{aligned}
\tag{1.8.15}
$$
其中，在求解其对偶问题的过程中设计到$\phi\left(\boldsymbol{x}_i\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_j\right)$的计算，这是样本$x_i$与$x_j$映射到新特征空间后的内积。若特征空间中的维度很高，甚至是无限高，那么计算$\phi\left(\boldsymbol{x}_i\right)^{\mathrm{T}} \phi\left(\boldsymbol{x}_j\right)$的过程将会非常困难。为了解决这一问题，我们可以设想一个函数$K(x_i,x_j)$来代替内积计算，从而达到将原始特征空间中的数据样本映射到更高维特征空间的目的，函数$K(\cdot,\cdot)$就称之为核函数(Kernel Function)。因此，我们利用核函数（和技巧）同样可以把线性支持向量机推广为非线性支持向量机，只需将线性支持向量机中的内积$\Vert \mathbf{w} \Vert$换成核函数$K(x_i,x_j)$即可。那么，使用和技巧的支持向量机的优化问题可以改写为：

$$
\begin{aligned}
\min _\alpha & \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K\left(x_i, x_j\right)-\sum_{i=1}^n \alpha_i \\
\text { s.t. } & \sum_{i=1}^n \alpha_i y_i=0 \\
& 0 \leq \alpha_i \leq C, i=1,2, \ldots, n
\end{aligned}
\tag{1.8.16}
$$

上式为使用核技巧的支持向量机优化目标函数。然后，再使用与之前类似的二次规划问题（原问题转化为对偶问题）求解算法求得最优解。常见的核函数有：高斯核函数(Guassian Kernel Function)（式1.8.17）、多项式核(Polynomial Kernel Function)（式1.8.18）:

$$
K(x_i, x_j)=\exp \left(-\frac{\|x_i-x_j\|^2}{2 \sigma^2}\right)
\tag{1.8.17}
$$

$$
K(x_i, x_j)=\exp \left(-\frac{\|x_i-x_j\|^2}{2 \sigma^2}\right)
\tag{1.8.18}
$$
