# 1.7 贝叶斯分类器 (Bayesian Classifier)
## 1.7.1 频率学派与贝叶斯学派 (Frequentist Vs. Bayesian)
机器学习的算法有两大学派，它们分别基于两种不同的思维方式来看待事物。第一种是频率学派(Frequentist)，频率学派经典的模型有线性回归、对数几率回归、决策树、神经网络等，这些模型都被称为判别模型(Discriminative Models)。第二种学派是贝叶斯学派(Bayesian)，朴素贝叶斯分类器(Naive Bayes Classifier)就是其中一种经典模型，这种模型也被称为概率生成模型，或称生成模型(Generative Model)。

频率学派认为，当出现一组数据时，这些数据一定是符合一个规律的，只是这个规律是未知的。所谓的规律可以理解为构造这些数据所需要的权重参数，且这些参数是唯一确定的。判别模型的目标是从已知的数据$x$中去估计出这些权重参数$w$，而估计的参数可以使得这些标签数据$y$出现的概率是最大的。其中对数几率回归中所使用到的最大似然估计(Maximum Likelihood Estimation, MLE)就是频率学派中根据数据来估计概率分布参数的简单方法。

贝叶斯学派认为所有的权重参数$w$都是都服从一个概率分布的随机变量。那么只要先对这些权重参数设定一个假设的概率分布（先验概率），通过实验结果（数据集）来调整这个概率分布。最终我们得到一个正确的分布（后验概率），使得数据集都符合这个分布。贝叶斯学派一个经典估计参数的方法为最大后验估计(Maximum a Posteriori Estimation, MAP)。

## 1.7.2 朴素贝叶斯分类器 (Naive Bayes Classifier)

朴素贝叶斯分类器(Naive Bayes Classifier)与对数几率回归中提到的分类任务概率模型非常相似，都是基于贝叶斯公式来进行建模。不同的地方在于，朴素贝叶斯分类器是一种生成模型，也就是直接找出输出$y$与特征$x$的联合分布$P(x,y)$，然后基于贝叶斯定理估计出后验概率：

$$
P(y\mid x)=\frac{P(x,y)}{P(x)}
\tag{1.7.1}
$$

与对数几率回归中提到的分类任务概率模型另一个不同点是，朴素贝叶斯分类器是以数据集中的特征属性相互独立为前提的。那么，假设有特征相互独立的一个数据样本$D_i=\{(x_1,x_2,\dots,x_n),y\}$:其中，该样本有$n$个特征，对于类别标签$y$有$K$个类别，每种类别写作$y=c_{k}(c=1,2,\dots,K)$。当一个数据样本中的特征集为$\boldsymbol{x}=\{x_1,x_2,\dots,x_n\}$时，对于样本特征集为$\boldsymbol{x}$，该样本成为类别$c_k$的后验概率$P(y=c_k \mid \boldsymbol{x})$为：

$$
P(y=c_k \mid \boldsymbol{x})=\frac{P(y=c_k) P(\boldsymbol{x} \mid y=c_k)}{P(\boldsymbol{x})}=\frac{P(y=c_k)}{P(\boldsymbol{x})} \prod_{j=1}^{n} P\left(x_{j} \mid y_c \right)
\tag{1.7.2}
$$
其中$P(\boldsymbol{x})$为一个样本的特征集$\boldsymbol{x}$的全概率，即$P(\boldsymbol{x})=\sum_{k}^{K}P(y=c_k)P(\boldsymbol{x} \mid y=c_k)=\sum_{k}^{K}P(y=c_k)\prod_{j=1}^{n} P(\boldsymbol{x_j} \mid y=c_k)$。$P(x_j \mid y=c_k)$是特征$x_j$相对于标签类别$y=c_k$的类条件概率，即类别$y=c_k$里的特征值是$x_j$的概率。

假定有一个新的数据样本$D_o$只有特征集$\boldsymbol{x}'$，并需要对这个新样本进行分类。那么我们要求的就是新特征集$\boldsymbol{x}'$分别可能为样本类别$c_k$的后验概率$P(y=c_k \mid \boldsymbol{x}')$最大化的类别：

$$
\begin{aligned}
h(\boldsymbol{x}')&=\operatorname{arg}\, \max P(y=c_k \mid \boldsymbol{x}')\\
&=\operatorname{arg}\, \max \frac{P(y=c_k P(\boldsymbol{x}' \mid y=c_k)}{P(\boldsymbol{x}')}
\end{aligned}
\tag{1.7.3}
$$
根据式1.7.3得知，求$h(\boldsymbol{x}')$需要估计出特征$\boldsymbol{x}'$全概率$P(\boldsymbol{x}')$、类别条件概率$P(\boldsymbol{x}' \mid y=c_k)$和类先验概率$P(y=c_k)$。由于对于所有类别$c_1,c_2,\dots,c_K$计算后验概率$P(y=c_k \mid \boldsymbol{x}')$时分母相同，皆为$P(\boldsymbol{x}')$。因此式1.7.3可以简化为：

$$
h(\boldsymbol{x}')=\operatorname{arg}\, \max {P(y=c_k) P(\boldsymbol{x}' \mid y=c_k)}
\tag{1.7.4}
$$
最终，我们只需要估计类别条件概率$P(\boldsymbol{x}' \mid y=c_k)$和类先验概率$P(y=c_k)$。假设$D_c$为训练数据集$D$中第$k$类别的样本集合。若独立同分布样本足够多，那么就可以估计出类先验概率$P(y=c_k)$：
$$
P(y=c_k)=\frac{\left|D_{c}\right|}{|D|}
\tag{1.7.5}
$$

对于类别条件概率$P(\boldsymbol{x}' \mid y=c_k)$的估计有两种情况，离散值特征属性估计与连续值特征属性估计。首先是对离散值特征属性的估计，令$D_{c,x_{j}}$为$D_c$中在第$j$个特征上取值为$x_{j}'$的新样本所组成的集合，则$P(x_{j}' \mid y=c_k)$为：

$$
P(x_{j}' \mid y_c)=\frac{\left|D_{c, x_{j}'}\right|}{\left|D_{c}\right|}
\tag{1.7.6}
$$

其次，若是对连续值特征属性估计，可以使用概率密度函数，假设$p(x_{j}' \mid y=c_k) \sim \mathcal{N}\left(\mu_{c, j}, \sigma_{c, j}^{2}\right)$，其中$\mu_{c, j}$和$\sigma_{c, j}^{2}$分别为第$c$类别样本在第$j$个特征属性上取值的均值和方差，那么$P(x_{j}' \mid y=c_k)$为：

$$
P(x_{j}' \mid y=c_k)=\frac{1}{\sqrt{2 \pi} \sigma_{c, j}} \exp \left(-\frac{\left(x_{j}'-\mu_{c, j}\right)^{2}}{2 \sigma_{c, j}^{2}}\right)
\tag{1.7.7}
$$

最终，利用估计出的类别条件概率$P(\boldsymbol{x}' \mid y=c_k)$和类先验概率$P(y=c_k)$，就可以分类出新数据$D_{o}$的类别$y=c_o$。

然而，在实际的分类任务中，使用离散特征属性往往会出现零概率问题：在计算事件的概率时，如果某个事件在观察样本库（训练集）中没有出现过，会导致该事件的概率结果是0。这是不合理的，不能因为一个事件没有观察到，就被认为该事件一定不可能发生（即该事件的概率为0）。为了解决零概率问题，我们需要使用拉普拉斯校正(Laplacian Correction)进行平滑(Smoothing)。具体做法是，对每个分量$\left|D_{c, x_{j}'}\right|$的计数加1，那么对于式1.7.6的拉普拉斯校正可以写为：

$$
P(x_{j}' \mid y_c)=\frac{\left|D_{c, x_{j}'}\right|+1}{\left|D_{c}\right|+N_{x_j}}
\tag{1.7.8}
$$
其中$N_{x_j}$为特征$x_j$可能取值的个数。当训练样本数量很多时，平滑处理后造成的估计概率变化可以忽略不计，但可以方便有效的避免零概率问题。
