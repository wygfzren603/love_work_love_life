# 应用案例

<!----------------------- ATSS ----------------------->

## 说明

本章将介绍相关工作在业务模型上的落地效果，所采用的的网络结构均为业务上使用的网络结构，所测试的benchmark也为目前业务发版主要参考的部分benchmark，所采用的的指标也为目前业务发版主要看的Recall@FP指标.

注1：不同的业务模型的网络结构、FLOPS等均存在差异

注2：不同的业务模型所测试的benchmark存在差异

## ATSS在业务上的应用

### 1. ATSS在人脸检测与脸人绑定任务中的应用

本节将介绍ATSS方法在人脸检测与脸人绑定等任务中的实际应用效果.

注1：本节内容中的实验均来自服务器检测相关业务

#### 模型结构

| 模型 | backbone FLOPS | code | 备注 |
| --- | --- | --- | --- |
| RetinaNet | 203m | [示例code](https://git-core.megvii-inc.com/gd_products/face_retina_net/tree/jxw/models/wangjianxiong/config/retinanet-resnet203m-face2person-all-headface-data-finetune-191030-trt5-int8) | baseline |
| ATSS | 203m | [示例code](https://git-core.megvii-inc.com/gd_products/face_retina_net/tree/jxw/models/wangjianxiong/config/retinanet_res203m_fp_binding_using_0426_head2face_data_model_1030_atss_fp_binding_20200115) |  |

#### 人脸检测

**1. 参数设置**

- anchor setting
```python3
anchor_ratios_face = [1]
anchor_scales_face = [2 ** (-1 / 2), 1, 2 ** (1 / 2)]
```

- top k
```python3
top_k_face = 9
```

**2. 指标**

| 模型 | security_crowdv1 (46 FP) | inside (756 FP) | outside (784 FP) |
| --- | --- | --- | --- |
| RetinaNet | 0.7518 | 0.7998 | 0.7129 |
| ATSS | 0.7625（+0.0107）| 0.7941（-0.0057）| 0.7039（-0.0090）|

注：相应的FP数量为RetinaNet发版阈值下对应的FP数量

#### 人体检测

**1. 参数设置**

- anchor setting

```python3
anchor_ratios_person = [2.5]
anchor_scales_person = [1, 2 ** (1 / 2)]
```

- top k
```python3
top_k_person = 9
```

**2. 指标**

| 模型 | overall (7053 FP) | scale_too_large (153 FP) | scale_large (6467 FP) | scale_middle (1082 FP) | scale_small (408 FP) |
| --- | --- | --- | --- | --- | --- |
| RetinaNet | 0.6391 | 0.5155 | 0.7037 | 0.4484 | 0.1438 |
| ATSS | 0.6566 （+0.0175） | 0.7100 （+0.1945） | 0.7201 （+0.0164） | 0.4314 （-0.0170） | 0.1455 （+0.0017） |

注：相应的FP数量为RetinaNet发版阈值下对应的FP数量

ATSS在scale_too_large上涨点很多，这是由于ATSS相比RetinaNet可以未超大的person匹配更多的训练样本.

#### 脸人绑定

**1. 参数设置**

同上述人脸检测与人体检测任务.

**2. 指标**

- face指标

| 模型 | security_crowdv1 (55 FP) | inside (705 FP) | outside (779 FP) |
| --- | --- | --- | --- |
| RetinaNet | 0.7308 | 0.7792 | 0.6976 |
| ATSS | 0.7388 （+0.0080）| 0.7786 （-0.0006） | 0.6991 （+0.0015）|

- person指标

| 模型 | overall (6511 FP) | scale_too_large (131 FP) | scale_large (6021 FP) | scale_middle (882 FP) | scale_small (344 FP) |
| --- | --- | --- | --- | --- | --- |
| RetinaNet | 0.6101 | 0.5424 | 0.6786 | 0.3860 | 0.1067 |
| ATSS | 0.6333 （+0.0232）| 0.7161 （+0.1737）| 0.7021 （+0.0235）| 0.3550 （-0.0310）| 0.1002 （-0.0065）|

注：相应的FP数量为RetinaNet发版阈值下对应的FP数量

#### 结论

1. ATSS对于person有明显的涨点作用，且大person明显涨点

   对于大person涨点的原因在于：大person容易超出图片，ATSS相比RetinaNet基于IoU的匹配方式能够得到更多的正样本.

2. ATSS对face的影响与模型结构有关，对于脸人绑定模型可以认为几乎没有影响

3. ATSS用于服务器脸人绑定模型发版

更多详细的实验记录见：[ATSS实验记录](https://wiki.megvii-inc.com/pages/viewpage.action?pageId=138389304#id-03-ATSS%E6%95%88%E6%9E%9C%E9%AA%8C%E8%AF%81-1.1%E3%80%81face&personbinding%EF%BC%8Cseparate%E6%B5%8B%E8%AF%95)


### 2. ATSS在人脸检测与脸人绑定任务中的应用 - 对person进行缩框

#### Motivation

原始ATSS在计算候选正样本时，考虑了gt框内全部样本. 对于person gt框而言，由于人体存在一些动作或者形变，通常gt框的边缘部分会包含一些无用的背景信息，因此这部分样本质量较低，可以去掉. 为此，这里将person以中心点进行缩框，以去掉一些低质量的anchor.

#### 模型结构

| 模型 | backbone FLOPS | code | 备注 |
| --- | --- | --- | --- |
| ATSS | 500m | [示例code](https://git-core.megvii-inc.com/gd_products/face_retina_net/tree/jxw/models/wangjianxiong/config/retinanet_res500m_fp_binding_using_all_data_atss_finetune_shrink_person_0.3_20200316) | |

#### 实验

**1. 参数设置**

在脸人绑定模型中进行了实验，将person框以person中心进行缩框，去掉anchor中心不在缩框后的框内的anchor. 这里缩框系数为0.6. 相应的code如下：

```python3
def check_anchor_in_gt(gt_box, anchors, ratio_top=0.3, ratio_right=0.3, ratio_bottom=0.3, ratio_left=0.3):
    anchors_cx = 0.5 * (anchors[:, 2] + anchors[:, 0])
    anchors_cy = 0.5 * (anchors[:, 3] + anchors[:, 1])
    
    width = gt_box[2] - gt_box[0] + 1
    height = gt_box[3] - gt_box[1] + 1
    gt_cx = (gt_box[0] + gt_box[2]) / 2
    gt_cy = (gt_box[1] + gt_box[3]) / 2

    gt_l = gt_cx - width * ratio_left
    gt_r = gt_cx + width * ratio_right
    gt_t = gt_cy - height * ratio_top
    gt_b = gt_cy + height * ratio_bottom

    l = anchors_cx - gt_l
    t = anchors_cy - gt_t
    r = gt_r - anchors_cx
    b = gt_b - anchors_cy

    is_in_gt = np.stack([l, t, r, b], axis=1).min(axis=1) > 0.01
    return is_in_gt
```

**2. 指标**

- face指标

| 模型 | security_crowdv1 (59 FP) | overall (278 FP) | inside (224 FP) | outside (54 FP) |
| --- | --- | --- | --- | --- |
| baseline | 0.7715 | 0.8853 | 0.8763 | 0.9095 |
| baseline + center sampling | 0.7736 （+0.0021）| 0.8855 （+0.0002）| 0.8758 （-0.0005）| 0.9126 （+0.0031）|

- person指标

| 模型 | overall (3000 FP) | video_overall (9708 FP) |
| --- | --- | --- |
| baseline | 0.6591 | 0.6217 |
| baseline + center sampling | 0.6736 （+0.0145）| 0.6304 （+0.0087）|

注：相应的FP数量为RetinaNet发版阈值下对应的FP数量


#### 结论

1. face指标几乎不受影响

2. person涨点比较明显：人像卡口整体+1.4，但是该阈值下结构化场景FP增加较多；同FP时结构化场景也涨点0.8

更多详细的实验记录见：[ATSS缩框实验记录](https://wiki.megvii-inc.com/pages/viewpage.action?pageId=154600931#id-11%E3%80%81atss+shrinkgtboxes-2%E3%80%81personshrink0.6)


<!----------------------- FCOS ----------------------->

## FCOS在业务上的应用

### 1. FCOS在人脸检测与脸人绑定任务中的应用

本节将介绍FCOS方法在人脸检测与脸人绑定等任务中的实际应用效果.

注1：本节内容中的实验均来自服务器检测相关业务

#### 模型结构

| 模型 | backbone FLOPS | code | 备注 |
| --- | --- | --- | --- |
| RetinaNet | 500m | [示例code](https://git-core.megvii-inc.com/gd_products/face_retina_net/tree/jxw/models/wangjianxiong/config/retinanet_res500m_fp_binding_using_0426_head2face_data_model_20200229_atss_fp_binding_finetune_add_data_20200228) | baseline |
| FCOS | 500m | [示例code](https://git-core.megvii-inc.com/gd_products/face_retina_net/tree/jxw/models/wangjianxiong/config/retinanet_res500m_fp_binding_using_all_data_fcos_face_only_20200326) |  |

#### 人脸检测

**1. 参数设置**

- 各fpn层的scale区间
```python3
sep_win_face = np.asarray([0, 32, 64, 128, 256, 10000])
sep_win_person = np.asarray([0, 64, 128, 256, 512, 10000])
```

**2. 指标**

- face指标

| 模型 | box分数 | security_crowdv1 (59 FP) | overall (278 FP) | inside (224 FP) | outside (54 FP) |
| --- | --- | --- | --- | --- | --- |
| RetinaNet | cls | 0.7770 | 0.8850 | 0.8743 | 0.9127 |
| FCOS | cls | 0.7726 （-0.0044）| 0.8882 （+0.0027）| 0.8771 （+0.0028）| 0.9192 （+0.0033）|
| FCOS | cls * cts | 0.7726 （-0.0044）| 0.8863 （+0.0008）| 0.8767 （+0.0024）| 0.9107 （-0.0020）|

注：相应的FP数量为RetinaNet发版阈值下对应的FP数量

#### 人体检测

- person指标

| 模型 | box分数 | overall (3000 FP) | video_overall (9708 FP) |
| --- | --- | --- | --- |
| RetinaNet | cls | 0.6588 | 0.6651 |
| FCOS | cls | 0.3971 （-0.2617）| 0.5564 （-0.1087）|
| FCOS | cls * cts | 0.6418 （-0.0170）| 0.6778 （+0.0127）|

注：相应的FP数量为RetinaNet发版阈值下对应的FP数量

#### 脸人绑定

- face指标

| 模型 | box分数 | security_crowdv1 (59 FP) | overall (278 FP) | inside (224 FP) | outside (54 FP) |
| --- | --- | --- | --- | --- | --- |
| RetinaNet | cls | 0.7653 | 0.8808 | 0.8715 | 0.9044 |
| FCOS | cls | 0.7524 （-0.0074）| 0.8763 （-0.0045）| 0.8655 （-0.0060）| 0.9057 （-0.0010）|
| FCOS | cls * cts | 0.7478 （-0.0175）| 0.8810 （+0.0002）| 0.8705 （-0.0010）| 0.9078 （+0.0011）|

注：相应的FP数量为RetinaNet发版阈值下对应的FP数量

- person指标

| 模型 | box分数 | overall (3000 FP) | video_overall (6206 FP) |
| --- | --- | --- | --- |
| RetinaNet | cls | 0.6219 | 0.6179 |
| FCOS | cls | 0.3971 （-0.2248）| 0.5179 （-0.1000）|
| FCOS | cls * cts | 0.6178 （-0.0041）| 0.6404 （+0.0225）|

注：相应的FP数量为RetinaNet发版阈值下对应的FP数量

#### 结论

1. Face整体上轻微涨点，但是小脸掉点较多

2. Person整体上掉点(0.5左右)，但是小人涨点较多，有进一步优化的空间

3. Centerness对于person格外重要，对于face反而会导致非常轻微掉点

   究其原因，face的scale较小，没有复杂的形变，gt框所所包含的基本都是有效信息，因此根据centerness抑制远离物体中心的检测框这一出发点未必成立；对于person框，由于其有丰富的形变，导致gt框的边缘会包含很多无用背景信息，因此使用centerness进行抑制有很好地效果.

更多详细的实验记录见：[FCOS在人脸检测、人体检测以及脸人绑定任务的试验记录](https://wiki.megvii-inc.com/pages/viewpage.action?pageId=154599979)


### 2. FCOS在人脸检测与脸人绑定任务中的应用 - 根据centerness而非面积解决模糊样本

#### Motivation

在FCOS中，如果一个pixel同时位于多个物体内部，会选择面积最小的物体进行匹配，可能会使得一些位于大物体中心点而的特征点匹配到小物体，显然是不太合理的. 直观上讲，根据centerness进行匹配应该更加合理.

#### 模型结构

| 模型 | backbone FLOPS | code | 备注 |
| --- | --- | --- | --- |
| FCOS_Area | 500m | [示例code](https://git-core.megvii-inc.com/gd_products/face_retina_net/tree/jxw/models/wangjianxiong/config/retinanet_res500m_fp_binding_using_all_data_fcos_fp_binding_20200326) | 根据面积进行匹配 |
| FCOS_Centerness | 500m | [示例code](https://git-core.megvii-inc.com/gd_products/face_retina_net/tree/jxw/models/wangjianxiong/config/retinanet_res500m_fp_binding_using_all_data_fcos_overlap_cts_fp_binding_20200405) | 根据centerness进行匹配 |

#### 人脸检测

**1. 指标**

- face指标

| 模型 | security_crowdv1 (59 FP) | overall (278 FP) | inside (224 FP) | outside (54 FP) |
| --- | --- | --- | --- | --- |
| FCOS_Area | 0.7726 | 0.8863 | 0.8767 | 0.9136 |
| FCOS_Centerness | 0.7710 （-0.0016）| 0.8867 （+0.0004）| 0.8761 （-0.0006）| 0.9174 （+0.0038）|

注：相应的FP数量为RetinaNet发版阈值下对应的FP数量

#### 人体检测

- person指标

| 模型 | overall (3000 FP) | video_overall (9708 FP) |
| --- | --- | --- |
| FCOS_Area | 0.6418 | 0.7128 |
| FCOS_Centerness | 0.6439 （+0.0021）| 0.7144 （+0.0016）|

注：相应的FP数量为RetinaNet发版阈值下对应的FP数量

#### 脸人绑定

- face指标

| 模型 | security_crowdv1 (59 FP) | overall (278 FP) | inside (224 FP) | outside (54 FP) |
| --- | --- | --- | --- | --- |
| FCOS_Area | 0.7478 | 0.8810 | 0.8705 | 0.9050 |
| FCOS_Centerness | 0.7553 （+0.0075）| 0.8838 （+0.0028）| 0.8743 （+0.0038）| 0.9079 （+0.0029）|

注：相应的FP数量为RetinaNet发版阈值下对应的FP数量

- person指标

| 模型 | overall (3000 FP) | video_overall (6206 FP) |
| --- | --- | --- |
| FCOS_Area | 0.6178 | 0.6404 |
| FCOS_Centerness | 0.6187 （+0.0009）| 0.6395 （-0.0009）|

注：相应的FP数量为RetinaNet发版阈值下对应的FP数量

#### 结论

1. 使用centerness替代面积进行模糊样本的分配，效果会更好一些

更多详细的实验记录见：[FCOS在人脸检测、人体检测以及脸人绑定任务的试验记录](https://wiki.megvii-inc.com/pages/viewpage.action?pageId=154599979)
