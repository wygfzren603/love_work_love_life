# 相关工作汇总

为了方便用户查看相关工作，在本章汇总相关工作，点击表格右边的链接即可跳转至相应工作的讲述章节.

## 说明

本文目前只关注one-stage方法，重点对anchor-based和anchor-free方法进行讲述.

本文在讲述相关工作时，重点关注相关工作的基本原理、复现code、业务落地效果、思考与求证、方法总结这几方面的内容.

为了方便知识的积累与传播，所有工作均基于 [basedet](https://git-core.megvii-inc.com/lizeming/basedet) (支持[MedDL](http://master.br.megvii-inc.com/docs/)，强烈推荐) 或 [cvpack2](https://git-core.megvii-inc.com/zhubenjin/cvpack2) (基于 PyTorch)复现.

注1：为了方便落地应用，请优先基于basedet的codebase进行复现和实验.

注2：本文目前只关注one-stage检测方法，有关two-stage检测方法的相关工作暂不收录.

注3：一些keypoing-based的anchor-free方法如CornerNet、CenterNet等暂不收录，这些方法与上文中介绍的anchor-free方法有本质区别.

## Anchor-based方法

**基于人工定义**

| 序号 | 名称及论文链接    | 本文链接     |  code链接  |
|------|----------------|-------------|------------|
| 1    | [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf) | [点击跳转](https://luoshu.iap.wh-a.brainpp.cn/docs/label-assign-white-book/en/latest/chapter-1-anchor-based-methods.html#yolov3)     | [点击跳转](https://git-core.megvii-inc.com/zhubenjin/cvpack2_playground/tree/liusongtao/songtao/yolo3) |
| 2    | [SSD](https://arxiv.org/abs/1512.02325)      | 待补充     | [点击跳转](https://git-core.megvii-inc.com/zhubenjin/cvpack2_playground/tree/master/examples/ssd/ssd.vgg16.coco.300size) |
| 3    | [RetinaNet](https://arxiv.org/pdf/1708.02002.pdf)      | [点击跳转](https://luoshu.iap.wh-a.brainpp.cn/docs/label-assign-white-book/en/latest/chapter-1-anchor-based-methods.html#retinanet)     | [点击跳转](https://git-core.megvii-inc.com/lizeming/basedet/blob/master/basedet/model/det/retina_net.py) |
| 4    | [ATSS](https://arxiv.org/pdf/1912.02424v1.pdf)           | [点击跳转](https://luoshu.iap.wh-a.brainpp.cn/docs/label-assign-white-book/en/latest/chapter-1-anchor-based-methods.html#atss)     | [点击跳转](https://git-core.megvii-inc.com/lizeming/basedet/blob/master/basedet/model/det/atss.py) |

**基于模型选择**

| 序号 | 名称及论文链接    | 本文链接     |  code链接  |
|------|----------------|-------------|------------|
| 1    | [MetaAnchor](https://arxiv.org/abs/1807.00980) | 待补充    | 待补充 |
| 2    | [GuidedAnchoring](https://arxiv.org/abs/1901.03278) | [点击跳转](https://luoshu.iap.wh-a.brainpp.cn/docs/label-assign-white-book/en/latest/chapter-1-anchor-based-methods.html#guidedanchoring)   | 待补充 |
| 3    | [FreeAnchor](https://arxiv.org/pdf/1909.02466.pdf)     | [点击跳转](https://luoshu.iap.wh-a.brainpp.cn/docs/label-assign-white-book/en/latest/chapter-1-anchor-based-methods.html#freeanchor)     | [点击跳转](https://git-core.megvii-inc.com/lizeming/basedet/blob/master/basedet/model/det/retina_net_free_anc.py) |
| 4    | [MAL](https://arxiv.org/pdf/1912.02252.pdf)            | [点击跳转](https://luoshu.iap.wh-a.brainpp.cn/docs/label-assign-white-book/en/latest/chapter-1-anchor-based-methods.html#mal)     | 待补充 |
| 5    | [FSAF](https://arxiv.org/abs/1903.00621)           | 待补充     | 待补充     |
| 6    | [Noisy anchors](https://arxiv.org/pdf/1912.05086.pdf)  | 待补充     | 待补充     |

## Anchor-free方法

**基于人工定义**

| 序号 | 名称及论文链接    | 本文链接     |  code链接  |
|------|----------------|-------------|------------|
| 1    | [YOLO](https://arxiv.org/abs/1506.02640) | 待补充     | 待补充 |
| 2    | [DenseBox](https://arxiv.org/abs/1509.04874) | 待补充     | 待补充 |
| 3    | [FCOS](https://arxiv.org/pdf/1904.01355.pdf)           | [点击跳转](https://luoshu.iap.wh-a.brainpp.cn/docs/label-assign-white-book/en/latest/chapter-2-anchor-free-methods.html#fcos)     | [点击跳转](https://git-core.megvii-inc.com/lizeming/basedet/blob/master/basedet/model/det/fcos.py) |
| 4    | [FoveaBox](https://arxiv.org/pdf/1904.03797.pdf)       | [点击跳转](https://luoshu.iap.wh-a.brainpp.cn/docs/label-assign-white-book/en/latest/chapter-2-anchor-free-methods.html#foveabox)     | [点击跳转](https://git-core.megvii-inc.com/lizeming/basedet/blob/mxy/basedet/model/det/fovea.py) |

**基于模型选择**

| 序号 | 名称及论文链接    | 本文链接     |  code链接  |
|------|----------------|-------------|------------|
| 1    | [SAPD](https://arxiv.org/pdf/1911.12448.pdf)           | [点击跳转](https://luoshu.iap.wh-a.brainpp.cn/docs/label-assign-white-book/en/latest/chapter-2-anchor-free-methods.html#sapd)     | 待补充 |

