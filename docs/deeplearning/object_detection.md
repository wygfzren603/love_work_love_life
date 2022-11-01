# Object Detection
目标检测算法是一个比较大的CV任务，目的在于在图像中检测物体，给出物体类型和位置。从算法的角度，可以理解为图像区域分类和位置回归。

```mermaid
graph LR;
输入图像 --> 区域选择 --> 特征提取
特征提取 --> 分类器
特征提取 --> 回归器
分类器 --> 类别+位置
回归器 --> 类别+位置
```

从目标检测算法的分类来看，可以分为：

* two-stage
* one-stage
或者

* anchor-based
* anchor-free

其中two-stage都是anchor-based的方法，而one-stage包括anchor-based和anchor-free的方法。
为此，将首先简要介绍一下two-stage的方法，目前并不是太常用；重点总结一下one-stage的方法，将主要从label assignment的角度来总结。

## two-stage
### RCNN
RCNN是首次将深度学习应用于目标检测的算法。ROI的提取采用传统的selective search, NN网络主要用于提取ROI特征，最后通过SVM来分类。其中NN网络的训练是通过fine-tuning已有的分类网络来实现的。  
存在的问题是，在inference的时候需要将每个ROI送入网络提取特征，这样导致特征提取的时间复杂度非常高。  
#### Bounding Box Regression(边界框回归算法)
BBR的目的是微调建议框，使其与真实框的位置和大小相似。 
[](https://zhuanlan.zhihu.com/p/404035883)

* 已知建议框的中心坐标($P_x, P_y$)和宽高($P_w, P_h$)以及该建议框区域的特征向量$P$，从本质上希望得到一个回归函数$(G_{x}^{'},G_{y}^{'},G_{w}^{'},G_{h}^{'})=f(P)$
* 而为了避免不同的图像尺寸对回归结果产生影响，提高预测精度，采用回归建议框和真实框之间的坐标偏移量
* 学习的目标
    * 真实框和建议框的中心点横坐标之差（除以建议框宽度）$t_x=(G_x-P_x)/P_w$
    * 真实框和建议框的中心点纵坐标之差（除以建议框高度）$t_y=(G_y-P_y)/P_h$
    * 真实框和建议框的宽度之比（取log）$t_w=log(G_w/P_w)$
    * 真实框和建议框的高度之比（取log）$t_h=log(G_h/P_h)$
    * smooth L1: 

        $$smooth_{L1}(x)=\begin{cases}
        0.5x^2 & \text{if }|x|<1 \\
        |x|-0.5 & \text{if }|x|\geq 1 \\
        \end{cases}$$

* inference
    * inference的位置结果为($t_{x}^{'}, t_{y}^{'}, t_{w}^{'}, t_{h}^{'}$)
    * 预测框的中心横坐标：$G_{x}^{'}=t_{x}^{'}*P_w+P_x$
    * 预测框的中心纵坐标：$G_{y}^{'}=t_{y}^{'}*P_h+P_y$
    * 预测框的宽度：$G_{w}^{'}=P_w*exp(t_{w}^{'})$
    * 预测框的高度：$G_{h}^{'}=P_h*exp(t_{h}^{'})$

#### NMS(非极大值抑制算法)
NMS是一种非极大值抑制算法，用于去除重复的预测框。假如一张图像预测了10个类别共100个预测框，则需要对每个类别的预测框分别进行NMS，算法步骤如下：

* 每个预测框的结果分别为$[x_1,y_1,x_2, y_2, score]$
* 对每个类别的预测框进行排序，按照预测框的得分从高到低排序
* 第一个得分最高的框自然保留
* 然后计算第一个框与其他框的IoU，如果IoU大于阈值，则该框被删除，否则保留

python代码如下[code](https://www.jb51.net/article/229498.htm)：
```
import numpy as np
def nms(dets, thresh):
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    areas = (y2-y1+1) * (x2-x1+1)
    scores = dets[:,4]
    keep = []
    index = scores.argsort()[::-1]
    while index.size >0:
        i = index[0]       # every time the first is the biggst, and add it directly
        keep.append(i)
 
        x11 = np.maximum(x1[i], x1[index[1:]])    # calculate the points of overlap 
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        
        w = np.maximum(0, x22-x11+1)    # the weights of overlap
        h = np.maximum(0, y22-y11+1)    # the height of overlap
       
        overlaps = w*h
        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
 
        idx = np.where(ious<=thresh)[0]
        index = index[idx+1]   # because index start from 1

```


### SPPNet
前面提到，RCNN在提取特征时，由于存在FC层，输入图像需要固定大小，为此需要将图像中的每个ROI处理到固定尺寸后送入网络提取特征。  
为了提高效率，提出了SPPNet，即将整张图像输入CNN网络，我们知道纯卷积网络是可以适应不同尺寸的输入图像的，然后对每个ROI对应的feature map区域通过金字塔池化处理成相同大小的特征向量，再将特征向量输入FC层进行分类，效率得到大大提高，其与RCNN的区别如下图所示：
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220614100330.png)

不足之处在于，和RCNN一样，SPP也需要训练CNN提取特征，然后训练SVM分类这些特征，这需要巨大的存储空间，并且多阶段训练的流程也很繁杂。除此之外，SPPNet只对全连接层进行微调，而忽略了网络其它层的参数。

### Fast-RCNN
Fast-RCNN在于进一步优化目标检测复杂度，其改进点如下：

* 去除了SVM分类器，采用FC+softmax进行ROI分类
* 采用ROI Pooling，相当于只有一层的空间金字塔池化SPP
* 多任务损失，分类和边框回归损失，其中边框回归损失采用L1损失，分类损失采用交叉熵损失

![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220624153738.png)

不足之处在于，Fast-RCNN仍然采用selective search算法来寻找感兴趣区域

### Faster-RCNN
[[paper](https://arxiv.org/pdf/1506.01497.pdf)][[code](https://github.com/Kyle1993/simplest_FasterRcnn_pytorch)]

Faster-RCNN是目标检测的里程碑之作，在检测速度和精度上都有了很大提升，其中主要改进点在于去除了通过selective search算法寻找感兴趣区域来获取ROI，而是采用RPN网络来获取ROI。。  
RPN网络的结构如下： 
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221008112724.png)

* 假设网络输入为(N, 3, 224, 224), 则backbone特征最后一层输出为(N, 1024, 14, 14)
* RPN网络的输入为(N, 1024, 14, 14)，输出分为两路，一路为分类输出(N, 2*num_anchors, 14, 14)，另一路为bbox回归输出(N, 4*num_anchors, 14, 14), 其中num_anchors为每个特征点的anchor数量  
* 产生anchors，ratios=[0.5, 1, 2], scales=[8, 16, 32]，以下为产生基本anchor的代码，然后将作用到feature map上得到所有的anchors：
```
def _make_anchors(w, h, x_ctr, y_ctr):
    anchors = np.array([x_ctr-(w-1)/2, y_ctr-(h-1)/2, x_ctr+(w-1)/2, y_ctr+(h-1)/2])
    return anchors.T

def _whctrs(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _ratio_enum(anchor, ratios):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    ws = np.round(np.sqrt(size / ratios))
    hs = np.round(ws * ratios)
    anchors = _make_anchors(ws, hs, x_ctr, y_ctr) 
    return anchors

def _scale_enum(anchor, scales):
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _make_anchors(ws, hs, x_ctr, y_ctr)
    return anchors

def get_anchor_np(base_size, anchor_scales, anchor_ratios):
    base_anchor = np.array([0, 0, base_size-1, base_size-1])
    ratio_anchors = _ratio_enum(base_anchor, anchor_ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], anchor_scales) for i in range(ratio_anchors.shape[0])])
    return anchors.astype(np.float32)

```
* proposal layer，将RPN网络的输出进行处理，得到最终的ROI，以下为proposal layer的流程：
    * 将RPN网络输出的loc作用于anchors，得到bbox
    * 对bbox的边界进行处理，使其不超过图片边界
    * 去除太小的bbox
    * 对bbox进行NMS
    * 保留前N(eg:300)个bbox，作为ROI
* anchors与gt进行配对：对于一张图来说，anchors为(N, 4)，gt为(K, 4)，然后计算iou(N, K), 最后需要通过匹配策略得到class label(N, C)和regression label(N,4)，其中C为分类数量，匹配策略为：

    * 从anchor的角度，如果一个anchor与所有GT的iou最大值大于一定的阈值（比如0.7）， 则这个anchor是正样本， 且其label为前景（RPN）或最大IOU对应的GT label（Fast rcnn）；
    * 从GT的角度，对于一个GT，所有anchors与该GT的iou最大值对应的anchor为正样本，这样做的目的是为了让GT尽量有anchor与之匹配；
    * 如果anchors与GT的IoU小于负样本阈值（比如0.3）， 则为负样本；
    * 如果anchors与GT的IoU介于2者之间， 则是忽略样本。
    * 每个anchor匹配好gt后，还需要通过BBR计算regression label(tx, ty, tw, th)

* 计算损失函数：利用RPN网络的输出以及配对得到的label计算分类和回归损失，其中分类损失函数为交叉熵，回归损失函数为smooth L1 loss

* 以上为RPN阶段（也就是第一阶段）的训练过程，接下来是第二阶段的训练过程，也就是Fast rcnn阶段，其训练过程如下：
    * 将proposal layer得到的ROI与gt进行配对，其中iou大于pos_iou_thresh的为正样本，处于[neg_iou_thresh_lo, neg_iou_thresh_hi)之间的为负样本，得到sample_roi, class label和regression label
    * 将sample_roi通过RoI pooling得到feature map，然后将feature map输入到Fast rcnn网络中，得到分类和回归的输出，注意这里的分类输出是n_class+1(包括背景类)，回归输出是n_class*4，比如n_class为21，则回归输出为84
    * 计算分类和回归损失，分类损失函数为交叉熵，回归损失函数为smooth L1 loss，其中计算回归损失的时候，需要将回归输出reshape为(n_class, n_class+1, 4)，然后根据class label选择对应的回归输出，然后计算smooth L1 loss

## One-Stage
以上RCNN系列为two-stage方法，下面主要介绍one-stage方法，one-stage方法的主要思想是将RPN和Fast rcnn合并为一个网络，这样就可以减少一次forward的计算，提高速度。

### SSD
[[paper](https://arxiv.org/pdf/1512.02325.pdf)][[code](https://github.com/amdegroot/ssd.pytorch)]

SSD是目标检测早期one-stage的代表算法之一，其主要思想是将输入图片分成多个feature map，然后在每个feature map上进行anchor的生成，然后将每个feature map的anchor通过卷积网络得到分类和回归的输出，最后将所有feature map的输出进行NMS，得到最终的检测结果。
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221009220219.png)

SSD的主要特点在于采用了多尺度的feature map，这样可以检测不同尺度的目标，比如小目标和大目标，同时也可以检测不同比例的目标，比如长方形和正方形。但与后来的retinaNet相比：

* SSD的多尺度是独立的，RetinaNet的FPN在不同尺度之间进行了层间融合
* SSD的采用softmax多分类，RetinaNet以及后续改进的FCOS都是采用K个二分类（Focal loss默认是二分类），RetinaNet的K个二分类可以看作是一个多分类，只不过每个类别的概率是通过sigmoid函数计算得到的，而不是softmax函数
* 为了与配合focal loss，RetinaNet的分类分支最后的卷积层的bias初始化为-log((1-0.01)/0.01)，可以加快训练和收敛，也就是为了避免初期分类损失过大的问题
* RetinaNet的不同层head部分共享参数，检测和分类分支之间参数不共享，各层分类/回归结果concat再求loss

SSD相关要点：

1. SSD采用6个不同尺度的feature map来做检测，以SSD300为例，6层feature map分别为(38,38),(19,19),(10,10),(5,5),(3,3),(1,1),其中大尺度的特征图用于检测小物体，小尺度的特征图用于检测大物体
2. SSD采用了多种尺度的anchor，比如(min_size,max_size)=(30,60),(60,111),(111,162),(162,213),(213,264),(264,315)，其中每个anchor的宽高比为1,2,3,1/2,1/3,1',比如第二个feature map,其有两种尺度的anchor，分别为sk=60,$\sqrt{60*111}$（也就是anchor的宽高为60和$\sqrt{60*111}$），对于尺度为60的anchor还需要变换宽高比（ar=2,3,1/2,1/3）来得到4个anchor，变换公式如下。另外对于sk=111的anchor则只有宽高比为1的anchor，因此总共有6个anchor。注意，对于第1，5，6个feature map，没有宽高比为3和1/3的anchor，因此只有4个anchor。
$$
w_{k}^{a}=s_{k} \sqrt{a_{r}}, h_{k}^{a}=s_{k} \sqrt{1 / a_{r}}
$$
3. anchor的匹配策略：
    - 对于图片中的每个ground truth，找到与其iou最大的anchor，将其标记为正样本，可以保证每个ground truth一定与某个anchor匹配
    - 对剩余的未匹配anchor，若与某个ground truth的IOU大于某个阈值，那么也该anchor也与这个ground truth匹配，一个ground truth可以匹配多个anchor，反之则不行；如果一个anchor与多个ground truth的IOU都大于某个阈值，那么只与iou最大的ground truth匹配
    - SSD采用hard negative mining，就是对负样本进行抽样，抽样时按照置信度误差进行降序排列，选取误差较大的top-k个负样本，以保证负样本比例接近1：3

### YOLO
[[paper](https://arxiv.org/pdf/1506.02640.pdf)][[code](https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py)]

YOLO是较早的one stage方法，本质上就是将检测问题统一成回归问题，将检测框的坐标回归到ground truth的坐标，同时将检测框的类别回归到ground truth的类别。YOLO的网络结构如下图所示。
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221024220442.png)

* YOLO的网络结构比较简单，只有一个卷积层和一个全连接层，卷积层的输出是一个7x7x30的tensor，其中7x7是feature map的尺寸，30是每个cell的输出
* 每个网格要预测B个bounding box，每个bounding box除了要回归自身的位置之外，还要附带预测一个confidence值。这个confidence代表了所预测的box中含有object的置信度和这个box预测的有多准两重信息
* 每个bounding box要预测(x, y, w, h)和confidence共5个值，每个网格还要预测一个类别信息，记为C类。则SxS个网格，每个网格要预测B个bounding box还要预测C个categories。输出就是S x S x (5*B+C)的一个tensor

YOLO的损失函数如下：
在数据处理时，会把target也处理成7x7x30的tensor，其中每个cell的输出是一个30维的向量，0-3，5-8个是bounding box的坐标，第4，9个是confidence，后20个是类别的one-hot编码。在计算损失时，会把每个cell的输出和对应的target进行比较，计算每个cell的损失，然后求和得到总的损失。损失函数的计算过程如下：
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221025093057.png)

### YOLOV2
[[paper](https://arxiv.org/pdf/1612.08242.pdf)][[code](https://github.com/longcw/yolo2-pytorch)]

YOLOV2相对于V1最主要的改进在于借鉴了region proposal的anchor机制，而其中比较有创新的点是anchor的尺度是通过k-means聚类来实现的，论文通过聚类得到5个尺度的width和height，聚类时的具体步骤：

* 首先给定k个聚类中心  ，这里的w和h是指anchor boxes的宽和高，anchor boxes的位置不确定，所以不需要x和y坐标；
* 计算每个gt box和每个聚类中心的距离d(按照上面那个公式计算)，计算时每个box的中心点都与聚类中心重合，这样才能计算IOU值，将box分配给距离最近的聚类中心；
* 所有box分配完毕以后，对每个聚类重新计算中心点，就是求该聚类中所有box的宽和高的平均值；
* 重复上面两步，直到聚类中心改变量很小。

YOLOV2对bbox的中心坐标回归采用V1的方式，即相对网格位置坐标的的偏移，这样可以将输出限制在0-1之间，示例如下图：
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221101102008.png)

另外，还涉及到较多trick，比如采用了Batch Normalization，预训练分类模型采用了更高分辨率的图片，Multi-Scale Training，Darknet-19

YOLOv2的训练主要包括三个阶段：

* 第一阶段就是先在ImageNet分类数据集上预训练Darknet-19，此时模型输入为224×224,共训练160个epochs。
* 第二阶段将网络的输入调整为448*448,继续在ImageNet数据集上finetune分类模型，训练10个epochs，此时分类模型的top-1准确度为76.5%，而top-5准确度为93.3%。
* 第三个阶段就是修改Darknet-19分类模型为检测模型，并在检测数据集上继续finetune网络。


### RetinaNet
[code](https://github.com/yhenon/pytorch-retinanet)

### FCOS

### CenterNet

### CornerNet

### ATSS

### FoveaBox

### NAS-FPN

### NAS-FCOS

