# 1.3 线性回归 (Linear Regression)

## 1.3.1 一维线性回归 (Simple Liner Regression)

线性回归(Linear Regression)是监督学习中最基础的模型，用于解决回归任务。该模型通过找出数据集特征$x$与标签$y$之间的线性关系，拟合出一个方程来对新数据进行预测。假设一个单变量数据集的特征为$x$，标签为$y$，那么一个关于单变量$x$的一维线性回归(Simple Linear Regression)的方程$f(x)$可以定义为：

\begin{equation}
f(x) = wx+b
\tag{1.3.1}
\end{equation}

其中$w$为线性回归方程的权重参数(Paramater)，$b$为偏置(Bias)。偏置的作用是为了更好的拟合数据以及调整模型的位置。

## 1.3.2 多元线性回归 (Multivariate Linear Regression)

若一组数据集为多特征（多变量）数据集，那么就可以将一维线性回归推广成多元线性回归(Multivariate Linear Regression)。假设一个多维特征数据集为$D=\{(x_1, x_2, x_3,\dots,x_m), y\}$，$y$为这些特征的标签，那么多元线性回归定义为：

$$
\begin{align*}
f(x_i) &= {w_1}{x_1}+{w_2}{x_2}+{w_3}{x_3}+\dots+{w_n}{x_n}+b \\ 
 &= \sum_{j=1}^n{{x_jw_j}+b}
\end{align*}
\tag{1.3.2}
$$




其中一组特征$(x_1,x_2,\dots,x_n)$与之对应的标签$y$的组合称为一个样本。然而在实际的任务中，一个完整的数据集会有若干个样本。例如，房价预测中会有很多不同的房子样本与其对应的价格，在时间序列任务中也会有多个不同时间点记录的数据。为了更好的表示这些数据，这里可以用矩阵$X$来表示每个特征$x_{i,j}$，其中$i=1,2,\dots,m$为样本个数，$j=1,2,\dots,n$为特征个数（数据维度）。对于每个样本的特征集$(x_{i,1},x_{i,2},\dots,x_{i,n})$有着对于的标签$y_i$，标签向量可以写作$\mathbf{y}=(y_1, y_2,\dots,y_m)$。最终一个完整的数据集$D$可以表示为$D=\{X,\mathbf{y}\}$。其中$X$的展开为：

\begin{equation}
X={
\begin{pmatrix}
 x_{1,1} &  x_{1,2} &  x_{1,3} & \dots &  x_{1,n} \\ 
 x_{2,1} &  x_{2,2} &  x_{2,3} & \dots &  x_{2,n} \\ 
 \vdots & \vdots & \vdots & \ddots & \vdots \\
 x_{m,1} &  x_{m,2} &  x_{m,3} & \dots &  x_{m,n} \\ 
 \end{pmatrix}}
\label{eq:x_mat}
\end{equation}

与之对应地，权重$w_j$也可以写成向量形式$\mathbf{w}$（式\ref{eq:w}），权重矩阵的最后一项是偏置向量$b$。为了让输入$X$与包含偏置的权重参数集$\mathbf{w}$相乘，因此需要在$X$的每一行的最后加多一项“$1$”，如式\ref{eq:x_mat_with1}。

\begin{equation}
\mathbf{w}={
\begin{pmatrix}
 w_1  \\ 
 w_2  \\ 
 \vdots  \\
 w_n  \\ 
 b    \\
 \end{pmatrix}}
\label{eq:w}
\end{equation}

\begin{equation}
X={
\begin{pmatrix}
 x_{1,1} &  x_{1,2} &  x_{1,3} & \dots &  x_{1,n} & 1 \\ 
 x_{2,1} &  x_{2,2} &  x_{2,3} & \dots &  x_{2,n} & 1 \\ 
 \vdots & \vdots & \vdots & \ddots & \vdots &  \vdots  \\
 x_{m,1} &  x_{m,2} &  x_{m,3} & \dots &  x_{m,n} & 1 \\ 
 \end{pmatrix}}
\label{eq:x_mat_with1}
\end{equation}

最后线性模型的矩阵形式的方程为：

\begin{equation}
f(X)=X\mathbf{w}
\end{equation}