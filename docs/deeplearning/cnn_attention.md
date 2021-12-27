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
通道注意力机制简单理解就是对于每一个feature map有一个对应的权重，即得到一个权重向量，然后与feature map对应相乘 
![](https://s2.loli.net/2021/12/27/GloNrpgQiXTaO26.png)
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