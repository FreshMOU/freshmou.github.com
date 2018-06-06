---
layout: post
title: Winograd for CNN（一）
date: 2018-05-31
tags: 深度学习 性能优化
---

### 前言

&#8195;&#8195;虽然现在深度卷积神经网络在计算机视觉领域表现的非常优秀，但它在大型数据集上训练时需要花费大量GPU计算时间，并且前向推理需要大量的计算力。我们希望深度卷积网络可以在嵌入式平台部署，并且希望在保证精度的情况下加快它的推理速度。常规的基于FFT的卷积对于大型滤波器是快速的，但是现有技术的卷积神经网络一般使用小的3×3滤波器。论文引入了基于Winograd的最小滤波算法，一种新的卷积神经网络快速算法。算法计算出小卷积上的最小复杂度，这使得它在过滤器和批量小的情况下更快。论文使用VGG网络对算法的GPU实现进行基准测试，并显示批处理大小从1到64的时时吞吐量。[1]

>Cong和Xiao使用Strassen算法进行快速矩阵乘法，以减少卷积网络层中的调度次数，从而降低其总算术复杂度。 作者还提出，来自算术复杂性理论的更多技术可能适用于衔接。[2]

&#8195;&#8195;原始的Winograd算法，前置了很多数论方面的知识，为了效率我就没有深入的去阅读了。本文主要针对阅读了`Fast Algorithms for Convolutional Neural Networks`。

### 卷积公式

&#8195;&#8195;假设卷积为G，图像为D，输入参数数量N，通道C，高H，宽W   
卷积核参数通道C，高R，宽S，则卷积公式如下

![](/images/posts/2018-05-31-Winograd/tex1.png){:height="40%" width="40%"}

&#8195;&#8195;我们可以将整个图像的输出写作（其中*指代2D相关性）

![](/images/posts/2018-05-31-Winograd/tex2.png){:height="20%" width="20%"}

### 算法

&#8195;&#8195;假设用长度为r的FIR滤波器来得到输出m的式子为$$F(m, r)$$，传统的winograd算法需要$$µ(F(m,r)) = m + r - 1$$次乘法。我们可以通过堆叠一维算法来得到二维的最小算法——假设$$F(m\times n, r\times s)$$指代用$$r\times s$$的滤波器来计算得到$$m\times n$$的输出，则它需要
![](/images/posts/2018-05-31-Winograd/tex3.png){:height="50%" width="50%"}
次乘法。以此为例，我们可以继续堆叠一维算法来得到多维的最小算法。  
&#8195;&#8195;**<font color="red">但是需要注意不管是一维、二维还是多维的最快计算法，它要求输入的数量与所需乘法数一样。</font>**

## $$F(2\times 2,3\times 3)$$

&#8195;&#8195;我们知道，乘法和加法在硬件实现上的时间复杂度一般是不一样的，乘法运算所需的时间通常远大于加法所需的时间。因此，用廉价运算代替昂贵运算也是加速运算的一种方法。原始的矩阵运算对于$$F(2,3)$$需要6次乘法，而Winograd提出了如下算法，

![](/images/posts/2018-05-31-Winograd/tex4.png){:height="50%" width="50%"}
&#8195;&#8195;其中，
![](/images/posts/2018-05-31-Winograd/tex5.png){:height="50%" width="50%"}

&#8195;&#8195;该算法只用了$$2+3-1=4$$个乘法就计算得到了$$F(2,3)$$，不过它涉及了4个与输入数据有关的加法，还有与常数滤波器有关的3个加法（$$g_0+g_2$$只要算一次就行了）和2个乘法（因为滤波器为常数，所以这3个加法和2个乘法可以认为不占用时间）。

&#8195;&#8195;我们可以将矩阵公式写成
![](/images/posts/2018-05-31-Winograd/tex6.png){:height="30%" width="30%"}
&#8195;&#8195;其中，$$\odot$$指逐元素的乘法（就是点乘，卷积用的）。对于$$F(2,3)$$而言，上述公式各元素表示的意义如下

![](/images/posts/2018-05-31-Winograd/tex7.png){:height="50%" width="50%"}

&#8195;&#8195;堆叠一维算法可以得到二维算法$$F(m\times m, r\times r)$$如下（这一步论文没有具体的分析，不是很懂为什么，估计是各种线性变换(￣▽￣) ）
![](/images/posts/2018-05-31-Winograd/tex8.png){:height="30%" width="30%"}
&#8195;&#8195;其中，$$g$$是一个$$r\times r$$的滤波器，$$d$$是一个$$(m+r-1)\times(m+r-1)$$的输入图像块。

&#8195;&#8195;$$F(2\times 2, 3\times 3)$$用winograd只需要$$4\times 4=16$$次乘法，而原始矩阵运算则需要$$2\times 2\times 3\times 3=36$$次乘法运算。尽管winograd法还需要用32次加法来进行数据转换，用28个浮点运算指令来进行滤波器转换，用24次加法来进行反转变换，但是相比原始矩阵运算法还是提升很大。

&#8195;&#8195;$$F(2\times 2,3\times 3)$$可以被用来计算卷积核为$$r\times r$$的卷积操作。其中，输入图像的每个通道需要被切割成$$(m+r-1)\times(m+r-1)$$大小的块（每个块与相邻块间有$$r-1$$的重叠区域），则每个通道可以有$$P=\left\lceil H/m\right\rceil\times\left\lceil W/m\right\rceil$$个块。然后$$F(2\times 2,3\times 3)$$可以分别计算所有块然后累加得到最终结果。

&#8195;&#8195;假设$$U=G_gG^T$$和$$V=B^{T}dB$$，则
$$
Y=A^{T}[U\odot V]A
$$

&#8195;&#8195;以$$(\widetilde {x},\widetilde {y})$$为各个块坐标，$$i$$指单张图片，$$k$$为滤波器，则可以将上述卷积公式改写成，
![](/images/posts/2018-05-31-Winograd/tex11.png){:height="40%" width="40%"}

&#8195;&#8195;以下是具体实现的伪代码

![](/images/posts/2018-05-31-Winograd/tex12.png){:height="70%" width="70%"}

&#8195;&#8195;论文接下来的算法介绍主要提供了$$F(3\times 3,2\times 2)$$和$$F(4\times 4,3\times 3)$$的$$A,G,B$$矩阵。

### 理解

&#8195;&#8195;下面这个公式是最重要的一块。

![](/images/posts/2018-05-31-Winograd/tex8.png){:height="30%" width="30%"}

$$A,G,B$$根据不同的卷积核有不同的值，而且是提前计算好的，可以用<https://github.com/andravin/wincnn>的脚本计算。不过，$$A,G,B$$也不是可以通过脚本直接得到的，还需要自己确定$$m+r-2$$个插值点，wincnn的作者推荐了<https://openreview.net/forum?id=H1ZaRZVKg&noteId=H1ZaRZVKg>可以帮助确定插值点。（感觉好难啊（−＿−；））

&#8195;&#8195;自己推了好久，发现根本就没办法算，之后只能找代码看（[winograd.py](https://github.com/NervanaSystems/neon/blob/master/neon/backends/winograd.py)），原来winograd是一定要加padding补全的，加多少padding视情况而定。所以其实这个算法就是针对卷积套公式，不过卷积计算选择不同算法，速度会不太一样。（现在有许多版本的winograd）

&#8195;&#8195;对于不同的输入用不同的padding补全，最后会选择性地舍弃部分数据。如$$7\times 7$$的输入，在用$$F(2\times 2, 3\times 3)$$计算时，会在左边补一个padding，在右边补两个padding，最后winograd卷积得到一个$$8\times 8$$的输出，这时就要舍弃最右边的一列。

&#8195;&#8195;我尝试写了一下自己版本的winograd算法，以便后面学习优化，具体可以看下一篇。

<br/>

参考文献

##### 1. Lavin A, Gray S. [Fast algorithms for convolutional neural networks[C]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 4013-4021.

##### 2. Cong J., Xiao B. (2014) [Minimizing Computation in Convolutional Neural Networks.](https://link.springer.com/chapter/10.1007/978-3-319-11179-7_36) In: Wermter S. et al. (eds) Artificial Neural Networks and Machine Learning – ICANN 2014. ICANN 2014. Lecture Notes in Computer Science, vol 8681. Springer, Cham

##### 3. [Winograd 方法快速计算卷积](http://shuokay.com/2018/02/21/winograd/)

##### 4. [知乎，如何通俗易懂地解释卷积？](https://www.zhihu.com/question/22298352)