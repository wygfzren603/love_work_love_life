# CNN Attention
Attention的本质是从众多信息中选择出对当前任务目标更关键的信息，也可以理解为定位到感兴趣的信息，抑制无用信息。  

实现的核心方法在于获取attention map，然后将attention map作用于相应的维度（通道，空间，时间，分支等）  

Attention一般的表达形式为：
$$
        Attention=f(g(x), x)
$$
其中$g(x)$表示获取attention map的方法，$f(g(x), x)$则表示将attention map作用于$x$的方法

## Attention分类
* Channel Attention
* Spatial Attention
* Temporal Attention
* Channel&Spatial Attention
* Spatial&Temporal Attention
* Branch Attention

## Channel Attention
通道注意力机制简单理解就是对于每一个feature map有一个对应的权重，即得到一个权重向量，然后与feature map对应相乘。
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20211227121832.png)
### SENet
SENet是通道注意力的先驱，主要包括squeeze和excitation模块，用公式可以表示为：
$$
\begin{align}
g(x)&=\sigma(W_2\delta(W_1GAP(X)))\\\\
Y&=g(x)X
\end{align}
$$

### GSoP-Net
作者觉得SENet中squeeze模块太简单，限制了获取高维统计信息的能力，为此采用二阶全局池化模块；其实本质上就是在squeeze模块上做文章，把它搞的更复杂些。
用公式可以表示为：
$$
\begin{align}
g(x)&=\sigma(WRC(Cov(Conv(X))))\\\\
Y&=g(x)X
\end{align}
$$
其中$Conv(·)$表示通过1x1的卷积减少channels，$Cov(·)$表示计算channels之间的协方差矩阵，$RC(·)$表示基于行的卷积（可以理解为将一行看着一个feature map后做卷积

### SRM
受到风格转化的启发，将风格转化与注意力机制相结合，也就是利用特征的均值和方差来提高获取全局信息的能力。
用公式可以表示为：
$$
\begin{align}
g(x)&=\sigma(BN(CFC(SP(X))))\\\\
Y&=g(x)X
\end{align}
$$  
其中$SP(·)$其实就是计算每个feature map的均值和方差，$CFC(·)$表示channel-wise fully-connected，即每个通道单独做全连接，这样可以减少参数量

### GCT
对于SE block来说，参数量较大，不适合在每个卷积层都添加，同时利用全连接层对通道之间的关系建模是一种隐式的过程，为了解决这些问题，提出了gated channel transformation (GCT)。  
用公式表示为：
$$
\begin{align}
g(x)&=\tanh(\gamma CN(\alpha Norm(X))+\beta)\\\\
Y&=g(x)X+X
\end{align}
$$
其中$Norm(·)$表示对每个通道做L2-norm，$s_c=\alpha{\Vert x_c\Vert}_2=\alpha\{[\sum_{i=1}^{H}\sum_{j=1}^{H}(x_{c}^{i,j})^2]+\epsilon\}^{\frac{1}{2}}$，$CN$表示channel normalization，这里其实也是l2-norm  

### ECANet
将SENet中通过FC降维的部分改为利用1D卷积来建模通道之间的关系
用公式表示为:
$$
\begin{align}
g(x)&=\sigma(Conv1D(GAP(X)))\\\\
Y&=g(x)X
\end{align}
$$

### FcaNet
作者认为GAP限制了模型的表达能力，为此从压缩的视角和在频率域分析GAP，证明了GAP是离散余弦变换（DCT）的一个特例，因此提出了多谱通道注意力（multi-spectral channel attention）。  
用公式表示为：
$$
\begin{align}
g(x)&=\sigma(W_2\delta(W_1[(DCT(Group(X)))]))\\\\
Y&=g(x)X
\end{align}
$$
其中$Group(·)$表示对输入X进行分组，$DCT(·)$表示2D离散余弦变换，这里对DCT的计算有些技巧。

### EncNet
主要用在语义分割上,context encoding module首先在top layer引入encoding layer。这个是一作之前的一篇文章，记输入feature map为C×W×H，记N=W×H为输入向量x个数。现在有K个同样是C维的字典向量dd，将N个输入向量和这K个字典向量算残差rij，再根据残差算权重，把权重再乘在残差上进行求和，最终输出是一个C维向量eij，[参考](https://zhuanlan.zhihu.com/p/91007433)

## Spatial Attention

### RAM(recurrent attention model)
利用RNN和RL实现attention，比较复杂，其中重要的一个结构是glimpse sensor，如下图
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220310192552.png)

### STN(spatial transformer networks)
由于CNN只具有平移不变性，不具有旋转，缩放，放射变换不变性，STN通过一个显示的过程去学习这些不变性，使得模型更加关注最相关的区域，其主要包含三个部分：

* 本地网络（Localisation Network）,常规CNN，用于提取特征
* 网格生成器（Grid Genator）
* 采样器（Sampler）

```
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # 空间变换器定位 - 网络
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # 3 * 2 affine矩阵的回归量
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # 使用身份转换初始化权重/偏差
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # 空间变换器网络转发功能
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # 执行一般的前进传递
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net().to(device)
```

### DCN(Deformable Convolutional Networks)
与STN类似，都是为了使CNN具有几何不变性；但DCN不是显式地通过学习得到变化矩阵，而是采用了不一样的方式。
DCN将卷积分为两步，首先学习每个卷积核参数的位移offset，然后将offset作用在普通卷积之上，以此来获取更大感受野
可变形卷积一般放在网络的最后几层
```
import torch
import torchvision.ops
from torch import nn

class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x
```

### self-attention
对于一个feature map $F\in\Re^{C\times W\times H}$，首先通过线性映射和reshape得到queries,keys,values，$Q,K,V\in\Re^{C^{'}\times N},N=H\times W$，self-attention公式为：
$$
\begin{align}
A&=(a)_{i,j}=Softmax(Q^{T}K)\\\\
Y&=VA
\end{align}
$$
其中$A\in\Re^{N\times N}$，self-attention存在复杂度较高的问题，因此有许多针对此问题的优化算法。

### vision transformer
Transformer在自然语言处理中获得了巨大成功，google将transformer应用到视觉中，同样得到了sota结果 

ViT是算法流程为：

* 对于b张图像(b,256,256,3)的图像，将图像分割成(b,8x8,32,32,3)的64张小图；
* 然后对每张(32,32,3)做线性映射为1024维，即(b,64,1024)的embedding；
* cls tokens为(1,1,1024)向量，然后扩展成(b,1,1024)的向量，然后和embedding进行concat，得到(b,65,1024)；
* embedding(b,65,1024)和位置编码向量(64+1, 1024)，结果为(b,65,1024)；
* 然后进入transformer阶段，transformer中总共层数为depth，每层由attention和feedforward组成，且attention和feedforward的结果会经过layernorm。
* attention为多头注意力，heads为8，每个head输出64维的q/k/v，为此多head输出为(b,65,8x64x3),chunk为qkv(b,65,8x64)x3，然后分别将q、k、v变换为(b,8,65,64)，再将q(b,8,65,64)与k.T(b,8,64,65)进行矩阵乘法，得到(b,8,65,65),经过softmax和dropout后，与v(b,8,65,64)相乘，得到(b,8,65,64),再变换为(b,65,8x64)，最后通过线性映射变换为(b,65,1024)
* feedforward为两层线性映射，输入输出的维数相同，中间隐藏层为hidden_dim，用到GELU和dropout操作
* layernorm就是对1024维的向量进行norm
* 对transformer的结果以第1维进行平均，得到(b,1024)
* 最后对其进行layernorm和线性映射，得到(b,num_classes)

### GENet
受SENet启发，提出了GENet，通过在空间域获取大范围的上下文信息，通过聚合信息以及插值得到attention map，处理过程如下：
$$
\begin{align}
g&=f_{gather}(X)\\\\
s&=f_{excite}(g) = \sigma(Interp(g))
Y&=sX
\end{align}
$$

### PSANet
主要用于语义分割网络，可以细细品之。

## Temporal Attention
Temporal attention主要用于视频处理

|  Category  |  Method  |  Publication  |  Tasks  |  g(x)  |
| ---- | ---- | ---- | ---- | ---- |
| Self-attention based methods Combine | GLTR | ICCV2019 | ReID | dilated 1D Convs -> self- attention in temporal di- mension |
| Combine local attention and global attention Method | TAM | Arxiv2020 | Action | a)local: global spatial average pooling -> 1D Convs, b) global: global spatial average pooling -> MLP -> adaptive con- volution |

## Branch Attention
### Highway networks
神经网络的深度对模型效果有很大的作用，可是传统的神经网络随着深度的增加，训练越来越困难，这篇paper基于门机制提出了Highway Network，使用简单的SGD就可以训练很深的网络，而且optimization更简单，甚至收敛更快。
$$
\begin{align}
Y_{l}&=H_{l}(X_{l})T_{l}(X_{l}) + X_{l}(1 - T_{l}(X_{l}))\\\\
T_{l}(X)&=\sigma(W_{l}^{T}X + b_{t})
\end{align}
$$

### SKNet
“选择性内核”（SK）卷积使神经元能够自适应地调整其RF大小。具体来说，我们通过三个运算符-Split，Fuse和Select来实现SK卷积。下图表示两个分支的情况。 
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220418230058.png)

* Split：使用不同的卷积核对原图进行卷积。
* Fuse：组合并聚合来自多个路径的信息，以获得选择权重的全局和综合表示。
* Select：根据选择权重聚合不同大小的内核的特征图

三分支的情况如下：
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220419095219.png)

### CondConv
该文是谷歌大神Quov V.Le出品，一种条件卷积，或称之为动态卷积。 卷积是当前CNN网络的基本构成单元之一，它的一个基本假设是：卷积参数对所有样例共享。作者提出一种条件参数卷积，它可以为每个样例学习一个特定的卷积核参数，通过替换标准卷积，CondConv可以提升模型的尺寸与容量，同时保持高效推理。作者证实：相比已有标准卷积网络，基于CondConv的网络在精度提升与推理耗时方面取得了均衡(即精度提升，但速度持平)。在ImageNet分类问题中，基于CondConv的EfficientNet-B0取得了78.3%的精度且仅有413M计算量。CondConv可以替换网络的任何卷积层，一般替换网络的后面层效果较好。
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220419103108.png)

普通卷积可以表示为：$Y=W*X$  
而condconv可以表示为： [代码](https://github.com/nibuiro/CondConv-pytorch/blob/master/condconv/condconv.py) 
$$
\begin{align}
Y&=(\alpha_{1}W_{1}+\cdots+\alpha_{n}W_{n}) * X \\\\
\alpha&=\sigma(W_{r}(GAP(X)))
\end{align}
$$
 
### Dynamic Convolution
相比高性能深度网络，轻量型网络因其低计算负载约束(深度与通道方面的约束)导致其存在性能降低，即比较有效的特征表达能力。为解决该问题，作者提出动态卷积：它可以提升模型表达能力而无需提升网络深度与宽度。

​ 不同于常规卷积中的单一核，动态卷积根据输入动态的集成多个并行的卷积核为一个动态核，该动态核具有数据依赖性。多核集成不仅计算高效，而且具有更强的特征表达能力(因为这些核通过注意力机制以非线性形式进行融合)。[代码](https://github.com/kaijieshi7/Dynamic-convolution-Pytorch/blob/master/dynamic_conv.py)
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220419231638.png)

## Channel & Spatial Attention

### Residual Attention Network
感觉可以理解为将残差网络的shortcut分支改造成一个attention分支，本质上是希望得到一个(c,h,w)的attention map，通过bottom-up，top-down的结构来得到一个与输入大小相同的mask
$$
\begin{align}
s&=\sigma(Conv_{2}^{1\times 1}(Conv_{1}^{1\times 1}(h_{up}(h_{down}(X)))))
X_{out} = sf(X) + f(X)
\end{align}
$$
其中$h_{down}$是采用在残差网络单元后采用多次最大池化层来增大感受野，h_{up}则采用线性插值来实现最后的attention map与输入的feature map一致
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220420182233.png)

### CBAM(convolutional block attention module)
CBAM结合了channel attention和spatial attention，可以表示为：
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220503210943.png)
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220503211031.png)

用公式可以表示为：

$$
\begin{align}
channel\space attention(CAM):\\\\
F_{avg}^{c}&=GAP^{s}(X)\\\\
F_{max}^{c}&=GMP^{s}(X)\\\\
s_{c}&=\sigma(W_{2}\delta(W_{1}F_{avg}^{c})+W_{2}\delta(W_{1}F_{max}^{c}))\\\\
M_{c}(X)&=s_{c}X\\\\
spatial\space attention(SAM):\\\\
F_{avg}^{s}&=GAP^{c}(X)\\\\
F_{max}^{s}&=GMP^{c}(X)\\\\
s_{s}&=\sigma(Conv([F_{avg}^{s};F_{max}^{s}]))\\\\
M_{s}(X)&=s_{s}X\\\\
summarized\space as:\\\\
X^{'}&=M_{c}(X)\\\\
Y&=M_{s}(X^{'})
\end{align}
$$

### BAM(bottleneck attention module)
与CBAM属于姊妹篇，其采用两个分支分别获取通道注意力和空间注意力，其中通道注意力与SENet相同，空间注意力采用膨胀卷积增大感受野，结构如下：
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220504114529.png)

公式表示为：

$$
\begin{align}
s_{c}&=BN(W_{2}(W_{1}GAP(X)+b_{1})+b_{2})\\\\
s_{s}&=BN(Conv_{2}^{1\times 1}(DC_{2}^{3\times 3}(Conv_{1}^{1\times 1}(X))))\\\\
s&=\sigma(Expand(s_{s}) + Expande(s_{c}))\\\\
Y&=sX+X
\end{align}
$$
对于膨胀卷积，以pytorch中的nn.Conv2d来说，其中dilation参数就是用于控制是否为膨胀卷积，当dilation为1时，默认为普通卷积，经卷积操作后输出的宽高可以按照下面的公式计算：

$$
\begin{align}
H_{out}&=\lfloor {\frac{H_{in}+2\times padding[0]-dilation[0]\times (kernel\underline{~}size[0]-1)-1}{stride[0]} + 1} \rfloor \\\\
W_{out}&=\lfloor {\frac{W_{in}+2\times padding[1]-dilation[1]\times (kernel\underline{~}size[1]-1)-1}{stride[1]} + 1} \rfloor
\end{align}
$$
如果采用dilation，且步长为1时，padding大小为：$padding=\frac {dilation\times (kernel-1)}{2}$

### scSE
其实就是将通道注意力和空间注意力结合而已，通道注意力就是SENet，而空间注意力则是采用1x1的卷积得到spatial attention map，跟SENet是相似的，结构如下：
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220505095432.png)

公式表示为：

$$
\begin{align}
s_{c}&=\sigma(W_{2}\delta(W_{1}GAP(X)))\\\\
X_{chn}&=s_{c}X\\\\
s_{s}&=\sigma(Conv^{1\times 1}(X))\\\\
X_{spa}&=s_{s}X\\\\
Y&=f(X_{spa},X_{chn})
\end{align}
$$

其中$f$表示融合函数，可以为maximum, addition, multiplication or concatenation

### Triplet Attention
本文提出了可以有效解决跨维度交互的triplet attention。相较于以往的注意力方法，主要有两个优点：

* 可以忽略的计算开销
* 强调了多维交互而不降低维度的重要性，因此消除了通道和权重之间的间接对应
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220506111023.png)

公式表示为：

$$
\begin{align}
X_{1}&=Pm_{1}(X)\\\\
X_{2}&=Pm_{2}(X)\\\\
s_{0}&=\sigma(Conv_{0}(Z-Pool(X)))\\\\
s_{1}&=\sigma(Conv_{1}(Z-Pool(X_{1})))\\\\
s_{2}&=\sigma(Conv_{2}(Z-Pool(X_{2})))\\\\
Y&=\frac{s_{0}X+Pm_{1}^{-1}(s_{1}X_{1}) + Pm_{2}^{-1}(s_{2}X_{2})}{3}
\end{align}
$$

其中$Pm_{1},Pm_{2}$表示关于H和W分别逆时针旋转90°，$Z-Pool$表示concatenate max-pooling and average pooling

### simAM
提出了一种简单有效的3D注意力模块，基于著名的神经科学理论，提出了一种能量函数，并且推导出其快速解析解，能够为每一个神经元分配权重。主要贡献如下：

* 受人脑注意机制的启发，我们提出了一个具有3D权重的注意模块，并设计了一个能量函数来计算权重；
* 推导了能量函数的封闭形式的解，加速了权重计算，并保持整个模块的轻量；
* 将该模块嵌入到现有ConvNet中在不同任务上进行了灵活性与有效性的验证。

能量公式：

$$
\begin{align}
e_t^* &= \frac {4(\sigma^2+\lambda)} {(t-\mu)^2+2\sigma^2+2\lambda}\\\\
Y &= sigmoid(\frac{1}{E}X)
\end{align}
$$

pytorch代码：

```
def forward(X, lambda):
    n = X.shape[2] * X.shape[3] - 1
    d = (X - X.mean(dim=[2, 3])).pow(2)
    v = d.sum(dim=[2, 3]) / n
    E_inv = d / (4 * (v + lambda)) + 0.5
    return X * torch.sigmoid(E_inv)
```

### CA(Coordinate attention)
作者提出了一种新的高效注意力机制，通过将位置信息嵌入到通道注意力中，使得轻量级网络能够在更大的区域上进行注意力，同时避免了产生大量的计算开销。为了缓解2D全局池化造成的位置信息丢失，论文作者将通道注意力分解为两个并行的1D特征编码过程，有效地将空间坐标信息整合到生成的注意图中。更具体来说，作者利用两个一维全局池化操作分别将垂直和水平方向的输入特征聚合为两个独立的方向感知特征图。然后，这两个嵌入特定方向信息的特征图分别被编码为两个注意力图，每个注意力图都捕获了输入特征图沿着一个空间方向的长程依赖。因此，位置信息就被保存在生成的注意力图里了，两个注意力图接着被乘到输入特征图上来增强特征图的表示能力。由于这种注意力操作能够区分空间方向（即坐标）并且生成坐标感知的特征图，因此将提出的方法称为坐标注意力（coordinate attention）。
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220516194832.png)

公式表示为：

$$
\begin{align}
z^h &= GAP^h(X)\\\\
z^w &= GAP^w(X)\\\\
f &= \delta(BN(Conv_{1}^{1\times 1}([z^h;z^w])))\\\\
f^h, f^w &= Split(f)\\\\
s^h &= \sigma(Conv_{h}^{1\times 1}(f^h))\\\\
s^w &= \sigma(Conv_{w}^{1\times 1}(f^w))\\\\
Y &= Xs^hs^{w}
\end{align}
$$

```
import torch
import torch.nn as nn

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h
```

### DANet(dual attention network)
提出了双重注意网络（DANet）来自适应地集成局部特征和全局依赖。在传统的扩张FCN之上附加两种类型的注意力模块，分别模拟空间和通道维度中的语义相互依赖性。

* 位置注意力模块通过所有位置处的特征的加权和来选择性地聚合每个位置的特征。无论距离如何，类似的特征都将彼此相关。
* 同通道注意力模块通过整合所有通道映射之间的相关特征来选择性地强调存在相互依赖的通道映射。
* 将两个注意模块的输出相加以进一步改进特征表示，这有助于更精确的分割结果
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220517134134.png)

公式表示为：

$$
\begin{align}
Q,K,V&=W_qX,W_kX,W_vX\\\\
Y^{pos}&=X+VSoftmax(Q^TK)\\\\
Y^{chn}&=X+Softmax(XX^T)X\\\\
Y&=Y^{pos}+Y^{chn}
\end{align}
$$

```
import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable
torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
```
### Self-Calibrated Convolutions
自校正卷积(SC)可以作为一个即插即用的模块来替代传统的卷积层。主要有两个优点：

* 它使得每个空间位置能够自适应地编码来自远程区域的信息上下文，就是自适应增加了感受野
* SC模块通用性强，而且不会额外引入参数
![](https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20220601190010.png)
公式表示为：[code](https://github.com/MCG-NKU/SCNet/blob/master/scnet.py)

$$
\begin{align}
T_1&= AvgPool_r(X_1)\\\\
X_{1}^{'}&=Up(Conv_2(T_1))\\\\
Y_{1}^{'}&=Conv_3(X_1)\sigma(X_1+X_{1}^{'})\\\\
Y_1&=Conv_4(Y_{1}^{'})\\\\
Y_2&=Conv_1(X_2)\\\\
Y&=[Y_1;Y_2]
\end{align}
$$

其中，$AvgPool$和$Conv_2$起到了空间和通道注意力的作用。

### SPNet(strip pooling network)
由于spatial pooling只针对一个小的区域，限制了获取长程依赖和远距离区域注意力的能力。为此，作者提出了一种新的pooling方法，能在水平和垂直方向上编码长程上下文。简单理解就是，通过向x和y方向分别投影得到一维向量，然后扩展成二维向量，还是比较简单粗暴的。

公式表示为：

$$
\begin{align}
y^1&=GAP^w(X)\\\\
y^2&=GAP^h(X)\\\\
y_h&=Expand(Conv1D(y^1))\\\\
y_w&=Expand(Conv1D(y^2))\\\\
s&=\sigma(Conv^{1\times 1}(y_v+y_h))\\\\
Y&=sX
\end{align}
$$

### GALA(global and local attention)
GALA结合了global attention和local attention, 其中采用SE block实现global attention，采用1x1的卷积层实现local attention。创新点在于对global attention和local attention结果的融合，使得其可以获得更好的结果。

公式表示为：

$$
\begin{align}
s_g&=W_2\delta(W_1GAP(x))\\\\
s_l&=Conv_2^{1\times 1}(\delta(Conv_1^{1\times 1})(X))\\\\
s_g^*&=Expand(s_g)\\\\
s_l^*&=Expand(s_l)\\\\
s&=\tanh(a(s_g^*+s_l^*) + m(s_g^*s_l^*))\\\\
Y&=sX
\end{align}
$$

