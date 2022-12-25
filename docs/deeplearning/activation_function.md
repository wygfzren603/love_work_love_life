# 总结nn中涉及到的各种激活函数

## ReLU
$$
y = \max(0, x)
$$
<center><img src="https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221117105722.png", height = "300", width = "300" /></center>

## Leaky ReLU
LeakyReLU的好处就是：在反向传播过程中，对于LeakyReLU激活函数输入小于零的部分，也可以计算得到梯度(而不是像ReLU一样值为0)，这样就避免了上述梯度方向锯齿问题。
$$
y = \max(\alpha x, x)
$$
如果将$\alpha$设置为可学习的参数，则为PRelu激活函数。
<center><img src="https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221117111107.png", height = "300", width = "300" /></center>

## ELU
理想的激活函数应该满足以下条件：

1. 输出的分布是零均值，可以加快训练速度
2. 激活函数是单侧饱和的，可以更快的收敛

ELU激活函数就是满足以上两个条件的激活函数。
$$
y = \begin{cases}
\alpha(e^x - 1), & x < 0 \\\\
x, & x \geq 0
\end{cases}
$$
<center><img src="https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221117111941.png", height="300", width="300"/></center>

## sigmoid
$$
y = \frac{1}{1 + e^{-x}}
$$
<center><img src="https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221117113728.png", height="300", width="300"/></center>

## tanh
$$
y = \frac{e^{x} - e^{-x}} {e^{x} + e^{-x}}
$$
<center><img src="https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221117142925.png", height="300", width="300"/></center>

## softmax
$$
y_i = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}}
$$

## swish
* 有助于防止慢速训练期间，梯度逐渐接近0并导致饱和
* 导数恒大于0
* 平滑度在优化和泛化中起了重要作用
$$
y = \frac{x}{1 + e^{-x}}
$$
<center><img src="https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221117144209.png", height="300", width="300"/></center>

## mish
$$
y = x \cdot tanh(\ln(1 + e^x))
$$
<center><img src="https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221117144817.png", height="300", width="300"/></center>

## gelu
高斯误差线性单元激活函数
$$
y = \frac{x}{2} \cdot (1 + tanh(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715x^3)))
$$
<center><img src="https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221117151658.png", height="300", width="300"/></center>

## hardswish
Swish激活函数已经被证明是一种比 ReLU 更佳的激活函数，但是相比 ReLU，它的计 算更复杂，因为有 sigmoid 函数。为了能够在移动设备上应用 swish 并降低它的计算开销， 提出了 h-swish。
$$
y = \begin{cases}
0, & x \leq -3 \\\\
x, & x \geq 3 \\\\
\frac{x}{6} \cdot (x + 3), & -3 < x < 3
\end{cases}
$$
<center><img src="https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221117151017.png", height="300", width="300"/></center>

## hardshrink
$$
y = \begin{cases}
x, & x \leq -\lambda \\\\
x, & x \geq \lambda \\\\
0, & -\lambda < x < \lambda
\end{cases}
$$

## softshrink
$$
y = \begin{cases}
x - \lambda, & x \leq -\lambda \\\\
x + \lambda, & x \geq \lambda \\\\
0, & -\lambda < x < \lambda
\end{cases}
$$

## softsign
Softsign函数是Tanh函数的另一个替代选择。就像Tanh函数一样，Softsign函数是反对称、去中心、可微分，并返回-1和1之间的值。其更平坦的曲线与更慢的下降导数表明它可以更高效地学习，比tTanh函数更好的解决梯度消失的问题。另一方面，Softsign函数的导数的计算比Tanh函数更麻烦。
$$
y = \frac{x}{1 + |x|}
$$
<center><img src="https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221117145207.png", height="300", width="300"/></center>

## softplus
$$
y = \ln(1 + e^x)
$$
<center><img src="https://raw.githubusercontent.com/wygfzren603/love_work_love_life/main/imgs/20221117145953.png", height="300", width="300"/></center>

## tanhshrink
$$
y = x - tanh(x)
$$

## threshold
$$
y = \begin{cases}
x, & x \geq \theta \\\\
0, & x < \theta
\end{cases}
$$