# 1.2 模型选择与评估 (Model Selection and Evaluation)
## 1.2.1 过拟合与欠拟合(Overfitting and Underfitting)
在监督学习的[简介](ML\chapter1\section1)中提到了机器学习在训练的过程中会出现过拟合(Overfitting)和欠拟合(Underfitting)的现象。过拟合的定义是由于模型对训练数据集的过度拟合并且对训练集表现良好，但是对于新的数据集模型性能却表现很差。换句话说就是模型的泛化能力(Generalization)很差。过拟合的现象如图1.2.1所示。与之相对的另一种现象就是欠拟合，欠拟合的定义是一个模型无法对训练集进行较好的拟合。模型复杂度低是出现欠拟合现象的其中一个原因。欠拟合的现象如图1.2.2所示。

<center>
<div >
    <div style="float:left; width:50%; padding-left:100px">
        <img src="ML\ML_figure\underfitting.png" width="100%">
        <div style="margin-bottom: 50px; margin-top: 20px>
        <font size="3">图1.2.1 Underfitting</font>
        </div>
    </div>
    <div style="float:right; width:50%; padding-right:100px">
        <img src="ML\ML_figure\overfitting.png" width="100%">
        <div style="margin-bottom: 50px; margin-top: 20px>
        <font size="3">图1.2.2 Overfitting</font>
        </div>
    </div> 
</div>
</center>

## 1.2.2 交叉验证法 (Cross Validation)
交叉验证法(Cross Validation)是构建机器学习模型过程中，用于模型参数验证的常用方法。交叉验证法从字面上理解，就是重复地使用数据集，将数据样本切分成多份并排列组合，组成多组不同的训练集和测试集用于模型训练和测试。因此，在数据样本排列组合的过程中，一次训练集中的某个样本在下次可能成为测试集中的样本，即所谓“交叉”。通常，在数据集的规模很小或不充足的时候就会使用交叉验证法来重复使用数据集中的样本，达到验证模型参数的目的。

交叉验证法首先将数据集$D$划分为$k$个大小相似且互斥的子数据集$D=D_1 \cup D_2 \cup \ldots \cup D_k, D_i \cap D_j=\varnothing(i \neq j)$。其中，每个子数据集$D_i$都尽可能保持数据样本分布的一致性。在进行交叉验证的过程中，每次只使用$k-1$个子数据集用作训练集，剩余的$1$个子数据集作为测试集。经过数据集的排列组合后我们就能得到$k$组训练-测试集，这样一来模型就可以进行$k$次训练和测试（每次训练时需重新初始化参数）。在$k$次训练-测试的过程之后，最终将这$k$次测试结果的均值作为交叉验证最后的结果。由于交叉验证法是经过$k$次训练-测试的交叉验证过程，因此交叉验证法又被称为$k$折交叉验证($k$-fold Cross Validation)。其中，交叉验证的$k$值一般取值为$10$。10折交叉验证的过程如图1.2.3所示。

<center>
    <img  src="ML\ML_figure\cross_validation.png" width="60%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.2.3 10-fold Cross Validation</font>
    </div>
</center>

## 1.2.3 评估指标：回归 (Performance Measure: Regression)

评价机器学习模型的泛化能力需要有衡量模型性能的评价标准(Evaluation Criteria)，称之为性能度量(Performance Measure)。在对比不同模型的能力表现时，采用不同的性能度量会导致不同的评价结果，这意味着模型的性能好坏是相对的，一个机器学习任务中什么样的模型表现效果是好的不仅取决于算法和数据，还取决于该任务的需求。

在回归任务中，常用的性能度量指标有均方误差(Mean Squared Error, MSE)、平均绝对误差(Mean Absolute Error, MAE)、均方根误差(Root Mean Squared Error, RMSE)，$R^2$(R Squared)等。回归任务模型的输出值和数据集中的标签值都是连续值，因此回归任务性能度量的核心思想都是计算模型输出值与标签值的误差(Error)，判断它们是否足够地“靠近”以判断模型拟合效果的好坏。给定数据集$D=\{(x_{1}, y_{1}),(x_{2}, y_{2}), \ldots,(x_{m}, y_{m})\}$，其中$x_i$是数据集中的特征，$y_i$是数据集中的标签值。若将机器学习模型表示为$f(\cdot)$，则模型的输出结果表示为$f(x_i)$。基于以上表达，常见的4种回归任务性能度量可以表示为：

- 均方误差(Mean Square Error , MSE)：

\begin{equation}
E_\text{MSE}(f;D)=\frac{1}{m}\sum_{i=1}^m{(f(x_i)-y_{i})^2}
\tag{1.2.1}
\end{equation}


- 平均绝对误差(Mean Absolute Error, MAE)：

\begin{equation}
E_{\text{MAE}}(f;D)=\frac{1}{m}\sum_{i=1}^m|f(x_i)-y_i|
\tag{1.2.2}
\end{equation}

- 均方根误差(Root Mean Squared Error, RMSE)：

\begin{equation}
E_{\text{RMSE}}(f;D)=\sqrt{E_{\text{MSE}}(f;D)}=\sqrt{\frac{1}{m}\sum_{i=1}^m{(f(x_i)-y_i)^2}}
\tag{1.2.3}
\end{equation}

- R Squared ($R^2$):

\begin{equation}
E_{R^2}(f;D)=1-\frac{\sum_{i=1}^m{(f(x_i)-y_i)^2}}{\sum_{i=1}^m{(y_i-\bar{y})^2}}
\tag{1.2.4}
\end{equation}

其中，MSE、MAE、RMSE的结果越小，代表模型的性能越好。$R^2$的取值范围为$(0,1)$，$R^2$越接近$1$，模型的性能越好。

## 1.2.4 评估指标：分类 (Performance Measure: Classification)
### 1.2.4.1 二分类评估指标 (Criteria for Binary Classification)
在分类任务中，模型的输出和数据集的标签都是离散值，即数据样本的类别，因此分类任务模型的评估方法与回归任务也会有所不同。对于分类任务，可以根据计算模型分类的准确度来对分类模型的性能进行评价。在二分类任务中，对于一个分类模型$f(\cdot)$输出值$f(x_i)$和样本标签$y_i$的类别正类“$1$”和负类“$0$”的组合有以下四种情况：

- $TP$：真正例(True Positve)，模型将标签正类预测为正类的数量；
- $FP$：假正例(False Positive)，模型将标签负类预测为正类的数量，也称为误报率；
- $TN$：真负例(True Negative) ，模型将标签负类预测为负类的数量。
- $FN$：假负例(False Negative) ，模型将标签正类预测为负类的数量，也称为漏报率。

显然，样例总数$=TP+FP+TN+FN$。根据这四种结果组合，“混淆矩阵(Confusion Matrix)”可以表示为图1.2.4。

<center>
    <img  src="ML\ML_figure\binary_confusion_matrix.png" width="25%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.2.4 Confusion Matrix for Binary Classification</font>
    </div>
</center>

根据图1.2.4中A、B、C、D之间的加法组合，表格中的每一项俩俩组合相加后的项还有以下意义：
- A+B: Actual Negative，实际上负类的数量；
- C+D: Actual Positive，预测的负类数量；
- A+C: Predicted Negative，预测的负类数量；
- B+D: Predicted Positive，预测的正类数量。

根据混淆矩阵中的不同组合，分类任务中常用的四种评估指标，准确率(Accuracy)，精确率(Precision)，查全率(召回率，Recall)以及F-Measure：

- 准确率(Accuracy)：评价分类问题的性能指标一般是分类准确率，即对于给定的数据，分类正确的样本数占总样本数的比例。但是准确率这一指标在一些正负分布不均匀（数据不平衡）的数据集上会有偏差。例如，一个数据集中的样本有100个正样本和有9900个负样本，如果有一个模型直接把所有的样本都预测为负， 模型的准确率也会高达99\%，但是由于它把所有的正样本都错误分类，因此这个模型实际性能是非常差的。
    \begin{equation}
    \begin{split}
    \text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}
    \end{split}
    \tag{1.2.5}
    \end{equation}

- 精确率(Precision)：精确率是指在预测为正类的样本中标签正类所占的比例。即模型分类出是正类的所有样本中，有多少是真的正类。

    \begin{equation}
    \begin{split}
    \text{Precision}=\frac{TP}{TP+FP}
    \end{split}
    \tag{1.2.6}
    \end{equation}
    
    
- 召回率(Recall)：召回率是指在真实为正类的样本中被预测为正类的比例。即样本标签本来是正类的样本中，模型分类成功找回了多少真的正类。

    \begin{equation}
    \begin{split}
    \text{Recall}=\frac{TP}{TP+FN}
    \end{split}
    \tag{1.2.7}
    \end{equation}

- F-Measure：因为Precision和Recall是一对相互矛盾的量，当P高时，R往往相对较低，当R高时， P往往相对较低。所以为了更好的评价分类器的性能，一般使用F-Measure作为评价标准来衡量分类器的综合性能，可以理解为是将精确率和召回率进行一个“平均”结合。

    \begin{equation}
    \begin{split}
    \text{F}=\frac{(\alpha^2+1) \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
    \end{split}
    \tag{1.2.8}
    \end{equation}
    其中，一般情况下我们会把参数$\alpha$设为$1$，因此也称为F1-score。

### 1.2.4.2 二分类评估指标 (Criteria for Multiclass Classification)

对于评估多分类模型的混淆矩阵的思路是将多分类看作是多个不同的二分类。例如当我们在对“类别1”的结果进行评估的时候，那么我就把“类别1”看作是“正类”，而其他分类则统一看作为“负类”。因此，多分类下的混淆矩阵可以用图表示。

<center>
    <img  src="ML\ML_figure\multiclass_confusion_matrix.png" width="35%">
    <br>
    <div style="margin-bottom: 50px; margin-top: 20px">
        <font size="3">图1.2.5 Confusion Matrix for Multiclass Classification</font>
    </div>
</center>

基于多分类的混淆矩阵，我们可以把之前的四种评估指标进行推广，将它们变成宏观(marco)的评估指标。主要的做法是分别计算出数据样本中每一个单独类别的指标，再进行求和然后取平均：

- 宏观准确率(marco-Accuracy)：
    
    \begin{equation}
    \text{Accuracy} = \frac{T}{T+F} 
    \label{eq:marco_accuracy}
    \end{equation}

- 宏观精确率(marco-Precision)：
    
    \begin{equation}
    \text{macro-Precision} =\frac{1}{n} \sum^{n}_{k=0}\frac{T^k_k}{\sum_i F^{k}_i + T^k_k} 
    \label{eq:marco_precision}
    \end{equation}
    
- 宏观召回率(marco-Recall)：		
    \begin{equation}
    \text{macro-Recall} = \frac{1}{n} \sum^{n}_{k=0}\frac{T^{k}_k}{\sum_j F^{j}_k + T^k_k}
    \label{eq:macro_recall}
    \end{equation}

- 宏观F Measure (macro-F1)：		
    \begin{equation}
    \text{macro-F1} = \frac{1}{n} \sum^{n}_
    {k=0}\frac{(\alpha^2+1)T^{k}_{k}}{\sum_i F^{k}_i+\sum_j F^{j}_k + 2T^k_k}, (\alpha=1)
    \label{eq:macro_f}
    \end{equation}

其中，准确率是一种针对的是全部类别的评估指标，因此宏观准确率就是准确率。与准确率不同的是，召回率、精确率和F1指标是针对数据样本中单独一种类别的评估指标，而准确率是针对样本中所有类别的指标，因此在计算宏观指标时，召回率、精确率和F1指标都需要计算出每一种类别的性能度量值最后再进行求和平均最终得到它们的宏观指标。多分类任务中，除了宏观指标外还有微观(micro)指标，其思路是将多分类结果中的$TP$、$FP$、$TN$、$FN$分别进行平均，再计算召回率、精确率和F1指标得到它们的微观指标。

### 1.2.4.3 ROC曲线与AUC
未完待续