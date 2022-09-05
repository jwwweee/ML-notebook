# K-邻近 (K-Nearest Neighbours)
## K-邻近算法流程 (Procedure of K-Nearest Neighbours)

K-邻近(K-Nearest Neighbours, KNN)算法是机器学习中最简单也是最基础的一种监督学习算法。K-邻近算法这个名字非常直观，很好的描述了它的原理过程：取$K$个邻居作为新数据样本的榜样参考。KNN有以下几个特点：KNN虽然是监督学习，但它不用像其他机器学习模型一样对模型进行训练，这一类算法也成为惰性学习(Lazy Learning)。KNN不仅可以处理分类任务，还可以解决回归任务。给定训练集$D=\left\{\left(x_{1}, y_{1}\right),\left(x_{2}, y_{2}\right), \ldots,\left(x_{m}, y_{m}\right)\right\}$，KNN具体的算法过程如下：

1. 设定$K$值（进行邻近比较数据点的个数）。
2. 假设有一个新数据点$x_o$，计算新数据点$x_o$与训练集中$\{x_1, x_2, \dots, x_m\}$各点之间的距离$\{d_i\}$。
3. 根据计算出的距离集合${d_i}$找到训练集$D$与$x_o$最近的$K$个点。记为邻域$N_k(x_o)$。
4. 在邻域$N_k(x_o)$中选择出现类别最多的点$(x_i, y_i)$中$y_i$的类别作为新数据点$x_o$的类别（投票法）。
- 若KNN进行的是回归任务，则要计算新数据点$x_o$与训练集数据在邻域$N_k(x_o)$中的平均值作为新数据点的预测值$y_o$（平均法）。


KNN的算法实现如图1.6.1所示，图中的新数据点根据其距离最近的$K=5$个数据点中，出现次数最多的类别作为新数据的类别。因此，图中的新数据点按照投票法规则应该归类为“红色圆圈”类。

<center>
    <img  src="ML\ML_figure\knn.png" width="40%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.6.1 K-Nearest Neighbours</font>
    </div>
</center>

新数据点$x_o$与训练集中$\{x_1, x_2, \dots, x_m\}$各点之间的距离计算方法有：欧氏距离、曼哈顿距离等。假设数据集有$m$个特征，每个特征点$x_{i,l}$与$x_{j,l}$点欧式距离与曼哈顿距离的计算公式分别为式1.6.1与式1.6.2。

$$
d_{\operatorname{eu}} \left(x_{i}, x_{j}\right)=\left(\sum_{l=1}^{n}\left|x_{i,l}-x_{j,l}\right|^{2}\right)^{\frac{1}{2}}
\tag{1.6.1}
$$

$$
d_{\operatorname{man}}\left(x_{i}, x_{j}\right)=\sum_{l=1}^{n}\left|x_{i,l}-x_{j,l}\right|^{2}
\tag{1.6.2}
$$