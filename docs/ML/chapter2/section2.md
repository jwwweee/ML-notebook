# 2.2 聚类 (Clustering)
## 2.2.1 K-Means

聚类(Clustering)是无监督学习里其中一类经典方法，其中K-Means算法又是聚类算法里的一种经典算法。聚类算法，顾名思义就是把数据样本根据它们的特点将他们聚集成不同的类别。K-Means的核心思想就是通过不断迭代找出最合适中心点，数据样本与哪一个中心点的欧氏距离最近，就被归为最近的那一类。对于使用K-Means算法的数据集有一个重要的前提：数据之间的相似度可以使用欧氏距离度量，如果不能使用欧氏距离度量，要先把数据转换到能用欧氏距离度量。假设数据集为$D=\{x_1,x_2,\dots,x_m\}$，K-Means算法的过程步骤如下（如图2.2.1所示）：
1. 设定簇（聚类类别）的个数为$K$。
2. 随机从数据集$D$中选择$K$个样本作为初始中心点$\{\mu_1,\mu_2,\dots,\mu_k\}$，每个中心点对应的类别簇为$\{C_1,C_2,\dots,C_K\}$。
3. 计算数据集中其他样本点与各中心点$\{\mu_1,\mu_2,\dots,\mu_K\}$的欧氏距离$d_{\operatorname{e}}(x_i, \mu_k)$。找到每个样本点距离最近的中心点，将它们一一归类为对应的簇。例如，假如与数据点$x_i$最近的中心点是$\mu_k$，那么就将数据点$x_i$划分为簇$C_k$。
4. 计算每个簇$C_k$中的数据点$x_p \in C_k$的均值，把均值作为新的中心点$\mu_{k}'$：

$$
\mu_{k}'=\frac{1}{\left|C_{k}\right|} \sum_{x_p \in C_{k}} x_p
\tag{2.2.1}
$$

5. 迭代重复步骤3、步骤4直到均值不再改变，则迭代完成，输出最终的簇划分$\{C_1,C_2,\dots,C_K\}$。

<center>
    <img  src="ML\ML_figure\k_means.png" width="55%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图2.2.1 Procedure of K-means</font>
    </div>
</center>

## 2.2.2 层次聚类 Hierarchical Clustering)
层次聚类(Hierarchical Clustering)是采用另一种思路对数据进行聚类的聚类算法。层次聚类通过计算属于不同类别的数据样本之间的相似度（或距离）来构建一棵有层次级别的嵌套聚类树。构建聚类树的方法有两种，一种是从下到上的的聚合(Agglomerative)策略，另一种是从上至下的分裂(Divisive)策略。一般情况下，构建聚类树通常使用从下到上的的聚合策略。

对于从下到上的聚类树策略（如图2.2.2所示），算法在开始时会把每一个原始数据样本分别看作一个单一的初始聚类簇，然后不断根据簇与簇之间距离$\operatorname{dist}(C_i, C_j)$最近的原则聚合小的聚类簇成为大的聚类簇，直到达到预设的聚类簇个数$H$为止。其中，由于簇与簇之间的最小距离$\operatorname{dist_{\operatorname{min}}}(C_i, C_j)$实际上是每个簇里数据样本点的距离。主要计算方式有三种：


1. 簇与簇之间样本点的最短距离，也称单链接(Single Linkage)：

$$
\operatorname{dist}(C_i, C_j)=\min _{x \in C_{i}, z \in C_{j}} \operatorname{dist}(x, z)
\tag{2.2.2}
$$

2. 簇与簇之间样本点的最长距离，也称全链接(Complete Linkage)：

$$
\operatorname{dist}(C_i, C_j)=\max _{x \in C_{i}, z \in C_{j}} \operatorname{dist}(x, z)
\tag{2.2.3}
$$

3. 簇与簇之间样本点的平均距离，也称均链接(Average Linkage)：

$$
\operatorname{dist}(C_i, C_j)=\frac{1}{\left|C_{i}\right|\left|C_{j}\right|} \sum_{{x} \in C_{i}} \sum_{{z} \in C_{j}} \operatorname{dist}({x}, {z})
\tag{2.2.4}
$$

<center>
    <img  src="ML\ML_figure\hierarchical_clustering.png" width="45%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图2.2.2 Hierarchical Clustering</font>
    </div>
</center>

层次聚类过程中要找的簇与簇之间的最短距离与不同簇里样本点的距离是两个不同的概念。我们要做的是找出簇与簇之间的最短距离，但是这个距离是以哪一种方式计算出来的，就需要我们自己决定。从图2.2.3中可以看出三种距离不同计算方法的区别。

<center>
    <img  src="ML\ML_figure\distance.png" width="50%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图2.2.3 Three Types of Distance Calculations</font>
    </div>
</center>

簇与簇之间的最短距离中的三种不同计算方式，其实可以理解为社区与社区之间的最短距离到底要怎么设置的问题。例如，社区之间到底是以距离最近的两个房屋作为它们的距离？还是以距离最大的两个房屋作为距离？又或者是以平均作为距离？三种计算距离方法在不同的情况下各有优劣。