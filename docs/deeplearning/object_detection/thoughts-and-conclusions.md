# 思考与总结

## 结论

如开篇章节所述，本书将label assignment问题定义为确定正负样本及其权重，并确定对应位置的回归量的方法. 通过上文中对label assignment相关工作的讲述和相关实验，可以得出如下结论.

**1. Label Assignment 包括空间分配（Spatial Assignment）和尺度分配（Scale Assignment）两个核心要素**

空间分配表示在特征图上确定正负样本所在区域；尺度分配表示对于如FPN这样的特征金字塔结构，需要将不同scale的物体分配到不同的层进行检测.

**2. 中心先验（Center Prior）是简单有效的标签分配依据**

“中心”一词的含义是指包围框的中心，与之对应的，还有一个中心是物体的中心. 中心先验是指，物体在包围框内的分布，大致上是以框的中心为中心点分布的，即越靠近框中心的位置，越能够提取到该物体的有效信息，更加容易输出高质量的检测框，进而得到理想的检测效果. 因此，这一部分区域在网络的训练中应该分配较高的权重，或者在推理阶段应该增加输出检测框的分数；反之，远离中心越靠近物体边缘的特征点越容易丢失物体的有效信息，从而容易输出低质量的检测框，在训练过程中应该分配较低的权重，或者在推理阶段应该降低输出检测框的分数.

无论是anchor-based方法还是anchor-free方法，在label assignment过程中都利用了中心先验信息. 在anchor-based方法中，将与gt框的IoU值超过阈值的anchor作为正样本，实际也是在gt框中心点附近确定了一个正样本区域，不过没有区分不同位置正样本的权重，可以认为是一种hard的权重分配方式. 在anchor-free方法中，早期的方法将gt框内部的特征点作为正样本，没有区分权重; 在近年来的一些工作如FCOS、SAPD中，会考虑根据特征点的centerness分配不同的权重，取得了较好的效果.

除了充分利用中心先验，也有一些工作尝试改变这一prior：MetaAnchor 和 GuidedAnchoring 分别通过统计量和loss来动态生成anchor，达到了自适应的调整prior的目的. 这样做能work的一个直观insight是，通常很难保证所有物体在都分布在框的中心，因此设计动态调整center prior的方法是一个可行的思路.

**3. 空间分配（Spatial Assignment）和尺度分配（Scale Assignment）需要同时解决**

除了利用中心先验解决spatial assignment，scale assignment需要同时解决. 现有的目标检测模型基本都会采用FPN结构，将不同scale的物体分配到FPN的不同层进行检测，即使不使用FPN，像SSD或者YOLOv3同样使用了多尺度的预测，其同样包含将不同大小的物体映射到不同scale的特征图上的过程. Anchor-based方法通过在不同的scale上设置大小不一致的anchor完成尺度分配；anchor-free的方法则通过人工规则确定.

科学的尺度分配方法可以带来两大收益：

* 一方面在合适的FPN层可以提取到与待检测物体所匹配的特征，通常认为浅层特征对应着小物体，高层特征对应大物体；
* 另一方面将不同尺度的物体分配到不同的FPN层，可以缓解**样本模糊**问题. 

样本模糊问题是指一个特征点可能匹配给多个gt框，在anchor-free方法中非常常见. 在anchor-based方法中，基于IoU值分label assignment策略自然将不同大小的物体分配到FPN的不同层上; 对于anchor-free方法，通常需要额外的参数设置才能完成尺度分配问题，例如在FCOS中根据特征点距离gt框的距离，在FoveaBox中则根据gt框的大小.

除了基于人工规则进行尺度分配，也有一些方法使用模型选择的方法进行尺度分配. 例如，在FSAF或者SAPD根据样本的loss将物体分配到一个或多个FPN层中.

目前仍然存在的一个比较大的问题是，现有方法基本都是将spatial assignment和scale assignment割裂开来，分别采用不同的策略解决；我们近期的工作表明，这两个部分可以通过一致的思路同时解决.

**4. 正负样本定义是 Label Assignment 最根本的问题**

ATSS方法对anchor-based方法与anchor-free方法的本质区别进行了研究，以RetinaNet和FCOS进行了对比试验. 实验结果表明：当采用相同的确定正负样本的方法时，两种不同的包围框回归方式的效果相当；当采用相同的包围框回归方式时，FCOS所采用的确定正负样本的方法效果明显更好. 这说明：**anchor-based和anchor-free方法最本质的区别是如何定义正负样本，而与回归box还是回归points无关**，表明只要能选取类似的正样本区域，就能得到相近的检测效果.

**5. Appearance-aware Assignment 以及 Hyper-parameters**

现有的检测方法基本遵循 “Assign then Learn” 的模式：即先通过某种方式做label assignment，然后根据label assignmnet的结果指导feature map上的不同位置的学习目标.

这种模式的问题在于，Assign的时候，由于是发生在训练开始之前，现有的assign方法基本无法得知物体的appearance信息，也就不能进行自适应的assign；近期一些基于统计（ATSS）的方法，在我们看来，并没有很好的解决这个问题.

另一方面，早期检测方法中大多依赖于人工规则进行label assignment，人工规则固然简单，在以往的检测方法中也取得了较好的检测效果，然而人工先验难以考虑特征的有效性。手动设定阈值的方式还有一个很大的问题是，如果更换了数据，就需要重新调一波参数，这是十分耗时的；这也是为什么大家被叫做“调参”工程师的原因.

当我们统一看待这两个问题，就会发现，自适应的方法，既可以做到appearace-aware，又能够降低手动调参的成本，甚至做到完全无超参，这样可以极大的解放R的精力.

在FreeAnchor与MAL中，均首先依赖人工规则确定一些候选正样本，而后根据这些正样本在前向传播中输出的分数动态调整正样本的数量及相应的权重，取得了较好的检测效果；对于尺度分配问题，FSAF和SAPD方法也采用了模型进行选择. 基于模型选择的方法能够从特征角度出发，结合每个候选样本在前向传播中的表现进行调整，一定程度上弥补了人工规则的不足.

值得一提的是，目前基于模型选择的方法并没有完全脱离人工规则，尚需要与人工先验进行结合.

## 存在的问题

**1. 仍然存在不少超参数需要调节**

现有的label assignment方法，即使采用了模型从特征层面进行自动选择，也存在不少超参数需要调整. 这些超参数主要包括：**确定正负样本中的阈值**，**正样本的数量**，**尺度分配中的scale参数**等. 当前的方法大多对一部分超参数的设置采用了模型进行选择，尚没有方法能够完全规避label assignment问题中所遇到的超参数. 过多的超参数不利于业务落地的推广，往往一个涨点的经验切换到不同的业务模型之后仍然需要大量的调参工作.

**2. 依赖于人工先验的参与**

早期的检测模型依赖于人工规则从anchor与gt框的位置关系等确定正负样本，没能考虑特征的有效性. 近期有一些方法（如FreeAnchor、MAL）先基于人工先验确定候选正样本，并随着网络的训练动态调整正样本的label或者权重. 这些方法或多或少均依赖于人工规则的参与，未必能够取得最优效果.

**3. 只对中心先验、尺度分配、正负样本的定义等部分问题进行了针对性的解决，缺少一种整体性的思路来看待 label assignment**

无论是基于人工规则的assignment方法，还是部分基于模型选择的assign方法，都只对上述的中心先验、FPN分层或者spatial位置选取的一部分做了优化。 比如MetaAnchor和GuidedAnchoring只优化了anchor setting，并没有对FPN选层和spatial位置选取做进一步的优化. FSAF 只提出了一种基于loss动态选择 FPN层的方案，对其他方面并没有涉及. 而 FreeAnchor 和 ATSS 则只对如何动态选取 spatial 位置做了改进，对中心先验和 FPN 选层并没有优化. 这背后其实体现出目前缺少一种整体性的思路来看待 label assignment.

**4. 不能做到感知物体的 Appearance 信息**

现有的方法在做label assignment时，并没有利用到物体的特征或者表征. 现有的assign机制主要依赖于物体的包围框. 这也就意味着，如果一个框确定了，那么它的label assignment的结果也就确定了；不论这个框内是什么物体，label assignment 的结果不变. 这种不关心物体特征的label assignment机制，显然不是最优的.


## 改进方向

**1. 统一 Label Assignment 和 Loss Balance**

从FSAF我们可以得到的启发是，loss对label assignment 是有一定的指导作用的. 而且，FPN 分层本身也有在解决样本个数和平均正样本个数不平衡的问题. 将loss的balance问题（比如现有的 Focal Loss）和label assignment用同一套策略来解决，是一个值得研究的方向.

**2. 联合处理 Spatial Assignment 和 Scale Assignment**

设计一种将中心先验、FPN 分层和 spatial 位置选取同时兼顾的方法.

**3. Feature-aware Label Assignment**

让label assignment能够aware到物体的信息，而不是在计算loss前就完成划分.

**4. Hyperparameter-free Label Assignment**

能够尽量的减少 label assignment 过程中的超参数，降低人工调参的成本，同时方便实验经验在不同业务模型上进行传播.
