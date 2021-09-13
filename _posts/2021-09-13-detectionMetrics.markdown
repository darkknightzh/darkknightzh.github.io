---
layout: post
title:  "目标检测中的一些评价参数"
date:   2021-09-13 16:00:00 +0800
tags: [detection]
pin: true
math: true
---

<style> h1 { border-bottom: none } </style>

转载请注明出处：

<https://darkknightzh.github.io/posts/detectionMetrics>


参考网址：

（基本上都是从下面网址中复制，因而可以直接去相应网址查看）

<https://blog.csdn.net/CherDW/article/details/55813071>

<https://www.jianshu.com/p/5df19746daf9>

<https://blog.csdn.net/sinat_42239797/article/details/93651594>

<https://blog.csdn.net/u013063099/article/details/80964865>

<https://blog.paperspace.com/mean-average-precision/>

<https://pro.arcgis.com/es/pro-app/latest/tool-reference/image-analyst/how-compute-accuracy-for-object-detection-works.htm>

<https://blog.csdn.net/xskxushaokai/article/details/89419025>

<br>


## **分类准确率accuracy_score**

分类准确率分数是指所有的预测正确（正类负类）的占总的比重。它不能告诉你响应值的潜在分布，并且它也不能告诉你分类器犯错的类型。

$$Accuracy=\frac{TP+TN}{TP+TN+FP+FN}$$

```python
sklearn.metrics.accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
```

normalize：默认值为True，返回正确分类的比例；如果为False，返回正确分类的样本数

```terminal
>>>import numpy as np
>>>from sklearn.metrics import accuracy_score
>>>y_pred = [0, 2, 1, 3]
>>>y_true = [0, 1, 2, 3]
>>>accuracy_score(y_true, y_pred)
0.5
>>>accuracy_score(y_true, y_pred, normalize=False)
2 
```

## **精确率Precision**

查准率。即正确预测为正的占全部预测为正的比例。

$$Precison=\frac{TP}{TP+FP}$$

```python
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_predict)
```


## **召回率recall_score**

查全率。即正确预测为正的占全部实际为正的比例。

$$Precison=\frac{TP}{TP+FN}$$

```python
klearn.metrics.recall_score(y_true, y_pred, labels=None, pos_label=1,average='binary', sample_weight=None)
```

返回值是一个list，是每一类的召回率

参数:

average : string, [None, ‘micro’, ‘macro’(default), ‘samples’, ‘weighted’]

将一个二分类matrics拓展到多分类或多标签问题时，我们可以将数据看成多个二分类问题的集合，每个类都是一个二分类。接着，我们可以通过跨多个分类计算每个二分类metrics得分的均值，这在一些情况下很有用。你可以使用average参数来指定。

- macro：计算二分类metrics的均值，为每个类给出相同权重的分值。当小类很重要时会出问题，因为该macro-averging方法是对性能的平均。另一方面，该方法假设所有分类都是一样重要的，因此macro-averaging方法会对小类的性能影响很大。

- weighted:对于不均衡数量的类来说，计算二分类metrics的平均，通过在每个类的score上进行加权实现。

- micro：给出了每个样本类以及它对整个metrics的贡献的pair（sample-weight），而非对整个类的metrics求和，它会每个类的metrics上的权重及因子进行求和，来计算整个份额。Micro-averaging方法在多标签（multilabel）问题中设置，包含多分类，此时，大类将被忽略。

- samples：应用在multilabel问题上。它不会计算每个类，相反，它会在评估数据中，通过计算真实类和预测类的差异的metrics，来求平均（sample_weight-weighted）

- average：average=None将返回一个数组，它包含了每个类的得分.

```terminal
>>>from sklearn.metrics import recall_score
>>>y_true = [0, 1, 2, 0, 1, 2]
>>>y_pred = [0, 2, 1, 0, 0, 1]
>>>recall_score(y_true, y_pred, average='macro') 
0.33...
>>>recall_score(y_true, y_pred, average='micro') 
0.33...
>>>recall_score(y_true, y_pred, average='weighted') 
0.33...
>>>recall_score(y_true, y_pred, average=None)
array([1.,  0., 0.])
```


## **roc_curve**

ROC曲线指受试者工作特征曲线/接收器操作特性(receiver operating characteristic，ROC)曲线,是反映灵敏性和特效性连续变量的综合指标,是用构图法揭示敏感性和特异性的相互关系，它通过将连续变量设定出多个不同的临界值，从而计算出一系列敏感性和特异性,roc曲线上每个点反映着对同一信号刺激的感受性。ROC曲线是根据一系列不同的二分类方式（分界值或决定阈），以真正例率（也就是灵敏度）（True Positive Rate,TPR）为纵坐标，假正例率（1-特效性）（False Positive Rate,FPR）为横坐标绘制的曲线。

![1](/assets/post/2021-09-13-detectionMetrics/1roc.png)
_图1_

ROC观察模型正确地识别正例的比例与模型错误地把负例数据识别成正例的比例之间的权衡。TPR的增加以FPR的增加为代价。ROC曲线下的面积是模型准确率的度量，AUC（Area under roccurve）。

- TP : True Positive 预测为1，实际也为1

- TN：True Nagetive 预测为0，实际也为0

- FP：False Positive 预测为1，实际却为0

- FN：False Nagetive 预测为0，实际却为1

$$TPR=\frac{TP}{TP+FN}=Precison$$

$$FPR=\frac{FP}{FP+TN}$$

具体到识别任务中就是，ROC曲线上的每一点反映的是不同的阈值对应的FP（false positive）和TP（true positive）之间的关系。通常情况下，ROC曲线越靠近（0，1）坐标（左上角）表示性能越好。

**纵坐标**：真正类率（True Positive Rate , TPR）或灵敏度（sensitivity）。预测为正且实际为正的样本占所有正例样本的比例。

TPR = TP /（TP + FN）  （正样本预测结果数 / 正样本实际数）

**横坐标**：假正率（False Positive Rate , FPR），预测为正但实际为负的样本占所有负例样本的比例；(1-Specificity)

FPR = FP /（FP + TN） （被预测为正的负样本结果数 /负样本实际数）

```python
sklearn.metrics.roc_curve(y_true,y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
```

该函数返回这三个变量：fpr，tpr，和阈值thresholds；

这里理解thresholds:

分类器的一个重要功能“概率输出”，即表示分类器认为某个样本具有多大的概率属于正样本（或负样本）。

 “Score”表示每个测试样本属于正样本的概率。

接下来，我们从高到低，依次将“Score”值作为阈值threshold，当测试样本属于正样本的概率大于或等于这个threshold时，我们认为它为正样本，否则为负样本。每次选取一个不同的threshold，我们就可以得到一组FPR和TPR，即ROC曲线上的一点。当我们将threshold设置为1和0时，分别可以得到ROC曲线上的(0,0)和(1,1)两个点。将这些(FPR,TPR)对连接起来，就得到了ROC曲线。当threshold取值越多，ROC曲线越平滑。其实，我们并不一定要得到每个测试样本是正样本的概率值，只要得到这个分类器对该测试样本的“评分值”即可（评分值并不一定在(0,1)区间）。评分越高，表示分类器越肯定地认为这个测试样本是正样本，而且同时使用各个评分值作为threshold。我认为将评分值转化为概率更易于理解一些。

```terminal
>>>import numpy as np
>>>from sklearn import metrics
>>>y = np.array([1, 1, 2, 2])
>>>scores = np.array([0.1, 0.4, 0.35, 0.8])
>>>fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
>>>fpr
array([0. ,  0.5,  0.5, 1. ])
>>>tpr
array([0.5,  0.5,  1. , 1. ])
>>>thresholds
array([0.8 ,  0.4 ,  0.35, 0.1 ])
>>>from sklearn.metrics import auc 
>>>metrics.auc(fpr, tpr) 
0.75 
```


## **Auc**

AUC (Area Under Curve) 被定义为ROC曲线下的面积，显然这个面积的数值不会大于1。又由于ROC曲线一般都处于y=x这条直线的上方，所以AUC的取值范围一般在0.5和1之间。使用AUC值作为评价标准是因为很多时候ROC曲线并不能清晰的说明哪个分类器的效果更好，而作为一个数值，对应AUC更大的分类器效果更好。

- AUC = 1，是完美分类器，采用这个预测模型时，存在至少一个阈值能得出完美预测。绝大多数预测的场合，不存在完美分类器。

- 0.5 < AUC < 1，优于随机猜测。这个分类器（模型）妥善设定阈值的话，能有预测价值。

- AUC = 0.5，跟随机猜测一样（例：丢铜板），模型没有预测价值。

- AUC < 0.5，比随机猜测还差；但只要总是反预测而行，就优于随机猜测。

计算AUC值，其中x,y分别为数组形式，根据(xi,yi)在坐标上的点，生成的曲线，然后计算AUC值；

```python
sklearn.metrics.auc(x, y, reorder=False)
```


## **roc_auc_score**

直接根据真实值（必须是二值）、预测值（可以是0/1,也可以是proba值）计算出auc值，中间过程的roc计算省略。

```python
sklearn.metrics.roc_auc_score(y_true, y_score, average='macro', sample_weight=None)
```

average : string, [None, ‘micro’, ‘macro’(default), ‘samples’, ‘weighted’]

```terminal
>>>import numpy as np
>>>from sklearn.metrics import roc_auc_score
>>>y_true = np.array([0, 0, 1, 1])
>>>y_scores = np.array([0.1, 0.4, 0.35, 0.8])
>>>roc_auc_score(y_true, y_scores)
0.75
```


## **混淆矩阵confusion_matrix**

假设有一个用来对猫（cats）、狗（dogs）、兔子（rabbits）进行分类的系统，混淆矩阵就是为了进一步分析性能而对该算法测试结果做出的总结。假设总共有 27 只动物：8只猫， 6条狗， 13只兔子。结果的混淆矩阵如下图：

![1](/assets/post/2021-09-13-detectionMetrics/2confusionMat.png)
_图2_
 
在这个混淆矩阵中，实际有 8只猫，但是系统将其中3只预测成了狗；对于 6条狗，其中有 1条被预测成了兔子，2条被预测成了猫。从混淆矩阵中我们可以看出系统对于区分猫和狗存在一些问题，但是区分兔子和其他动物的效果还是不错的。所有正确的预测结果都在对角线上，所以从混淆矩阵中可以很方便直观的看出哪里有错误，因为他们呈现在对角线外面。

```python
sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
```

返回一个混淆矩阵；

labels：混淆矩阵的索引（如上面猫狗兔的示例），如果没有赋值，则按照y_true, y_pred中出现过的值排序。

```terminal
>>>from sklearn.metrics import confusion_matrix
>>>y_true = [2, 0, 2, 2, 0, 1]
>>>y_pred = [0, 0, 2, 2, 0, 2]
>>>confusion_matrix(y_true, y_pred)
array([[2,0, 0],
     [0, 0, 1],
     [1, 0, 2]])

>>>y_true = ["cat", "ant", "cat", "cat","ant", "bird"]
>>>y_pred = ["ant", "ant", "cat", "cat","ant", "cat"]
>>>confusion_matrix(y_true, y_pred, labels=["ant", "bird","cat"])
array([[2,0, 0],
      [0, 0, 1],
      [1, 0, 2]])
```


## **F-score**
用来衡量二分类模型精确度的一种指标。它同时兼顾了分类模型的准确率和召回率，综合权衡Precision和Recall。因为Precision和Recall是对立关系的，一般情况下，Precision高，Recall就低；Recall高，Precision就低。在实际中常常需要根据具体情况做出取舍，例如一般的搜索情况，在保证召回率的条件下，尽量提升精确率。

$$F-score=\left( 1+{ {\beta }^{2}} \right)\centerdot \frac{Precison\centerdot Recall}{ { {\beta }^{2}}\centerdot Precison+Recall}$$

当β=1时，称为F1-score，这时，精确率和召回率都很重要，权重相同。当有些情况下，我们认为精确率更重要些，那就调整β的值小于1，如果我们认为召回率更重要些，那就调整β的值大于1。

$${ {F}_{1}}=2\centerdot \frac{Precison\centerdot Recall}{Precison+Recall}$$

F1值为算数平均数除以几何平均数，且越大越好，将Precision和Recall的上述公式带入会发现，当F1值小时，True Positive相对增加，而false相对减少，即Precision和Recall都相对增加，即F1对Precision和Recall都进行了加权。

$$\frac{1}{ { {F}_{1}}}=\frac{1}{Precison}\text{+}\frac{1}{Recall}$$

公式转化之后为：

$${ {F}_{1}}=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}$$

```python
from sklearn.metrics import f1_score
f1_score(y_test, y_predict)
```

实际计算时，根据不同阈值，得到不同的Precision和Recall，而后通过上式计算F1 score，从而确定具体阈值，兼顾Precision和Recall。


## **Rank-n**

Rank-n和mAP均是衡量算法搜索能力的指标。

搜索结果中最靠前（置信度最高）的n张图有正确结果的概率

rank 1， 就是第一次命中。

rank k，就是在第k次以内命中。

rank-5准确度是rank-1准确度的扩展，我们对一张图片的预测结果是来自模型输出结果中top5对应的5个预测，而不是top1的1个预测

当处理的大型数据集各个类别之间存在许多具有相似特征时，我们往往会增加一个rank-5准确度，也就是说我们不止关心rank-1准确度，也关心rank-5准确度。结合两个准确度来以衡量神经网络的性能。理想情况下，随着预测数据增加，希望rank-1准确度和rank-5准确度同比例增加。但是，在某些数据集上，情况往往并非总是如此。

因此，我们也根据rank-5准确度检验模型，以确保我们的网络在后面的迭代中仍然是“学习”的。在训练快结束时，rank-1准确度可能会停滞不前，但是当我们的网络学习到更多的识别特征(虽然没有足够的识别能力超过top1的预测)时，rank-5准确度会继续提高。


## **平均精确度mAP**

mAP（mean Average Precision）即平均精确度的缩写。

先来看一下AP：所谓的AP，其实就是平均准确率。这里的平均是相对于不同验证集而言的。

假设我们想要训练一个猫分类器，如果是基于像素点级别的，那么就可以将它转换为一个二分类任务，那么它的mAP=正样本的准确率与负样本的准确率的平均值，多分类任务也可以如法炮制。

mAP这个性能指标对于数据不平衡问题就靠谱很多。假设我们有90%的正样本，10%的负样本，如果采用全部输出为正的分类器，那么P正=0.9,P负=0, 而mAP=0.45.

下面是几个在比较mAP值时需要记住的几个重点：

- 1、 mAP需要计算完整个数据集；

- 2、 虽然模型输出的绝对量化是难以解释的，但mAP可以通过一个很好的相关性标准来帮助我们。当我们在流行的公开数据集上计算这个标准时，它可以很容易地用来比较目标检测的新旧方法。

- 3、 根据类别在训练集上的分布方式，AP值可能在某些类别上从很高（这有很好的训练数据）变化到很低（对数据很少或不好的类别）。所以你的mAP可能是合适的，但你的模型可能对某些类别非常好而对某些类别非常差。因此当分析你的模型结果时，观察独立类别的AP是明智的。这些值可能作为添加更多训练样本的指示器。

一般目标检测任务中，使用IoU判断是否检测正确。IoU>0.5，则认为检测的正确；否则认为检测错误。得到每个目标实际的IoU后，根据检测到的目标的置信度confidence，计算相应的Precision（纵坐标）和Recall（横坐标），得到xy坐标系上的曲线。并通过下式计算AP（当前点的Recall和下一个点的Recall之差，乘以当前点的Precision，而后求和。要给Recall和Precison分别添加0和1到最后）

$$\begin{align}
  & AP=\sum\limits_{k=0}^{k=n-1}{\left[ Recalls\left( k \right)-Recalls\left( k+1 \right) \right]*Precisions\left( k \right)} \\ 
 & \quad \quad \quad \quad \quad Recalls\left( n \right)=0, \quad Precisions\left( n \right)=1 \\ 
 & \quad \quad \quad \quad \quad n=Number\text{ }of\text{ }thresholds \\ 
\end{align}$$

即AP为Precision-Recall曲线下方的面积：

![1](/assets/post/2021-09-13-detectionMetrics/3PRcurve.png)
_图3_

非VOC2007的算法：对于某类下全部的真实目标，将IOU>=0.5 的作为检测出来的目标，取不同的confidence 阈值计算对应的precision 和recall，对于每个recall，取其对应的最大precision，对这些precision 求平均即为该类的AP 值。所有类的AP 值求平均即为mAP。

VOC2007的算法：取[0, 0.1, 0.2, 0.3, ......, 1]作为confidence 阈值计算precision 和recall，其余同上。

在评测时，COCO评估了在不同的交并比(IoU)[0.5:0.05:0.95]共10个IoU下的AP，并且在最后以这些阈值下的AP平均作为结果，记为mAP@[.5, .95]。

而在Pascal VOC中，检测结果只评测了IOU在0.5这个阈值下的AP值。

因此相比VOC而言，COCO数据集的评测会更加全面：不仅评估到物体检测模型的分类能力，同时也能体现出检测模型的定位能力。因此在IoU较大如0.8时，预测框必须和真实的框具有很大的重叠比才能被视为正确。


## **累积匹配曲线CMC**

CMC曲线全称是Cumulative Match Characteristic (CMC) curve，也就是累积匹配曲线。

CMC曲线综合反映了分类器的性能，它的评价指标与现在deep learning中常用的top1 err或top5 err评价指标一样的意思，不同的是这里Rank1 recognition rate表示的是正确率而不是错误率，两者的关系是

Rank1识别率=1-top1 err

Rank5识别率=1-top5 ert

CMC将再识别问题看成是一种排序问题，CMC 曲线的具体含义是指：在候选行人库(gallery)中检索待查询(probe)的行人，前r个检索结果中包含正确匹配结果的比率。其中，第 1 匹配率 r=1 指标rank-1反映了最匹配候选目标刚好为待查询图片目标的概率，即该指标为真正的识别能力，因此很重要。但是当 r 值很小但大于1时，由于可以通过人眼进行辅助识别查找目标，因此也很有现实意义，如第 5 匹配率 r=5 指标反映前5个匹配候选目标中存在待查询图片目标的概率，如下图：横坐标表示排名等级r，纵坐标表示识别的准确率。

![1](/assets/post/2021-09-13-detectionMetrics/4CMC.png)
_图4_


## **Single shot和Muti shot**

前者是指gallery中每个人的图像为一张（N=1），而后者是指gallery中每个人的图像为N>1张图像，同样的Rank-1下，一般N越大，得到的识别率越高。

