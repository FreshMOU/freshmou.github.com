---
layout: post
title: Winograd
date: 2018-05-31
tags: 深度学习
---

### 前言

&#8195;&#8195;虽然现在深度卷积神经网络在计算机视觉领域表现的非常优秀，但它在大型数据集上训练时需要花费大量GPU计算时间，并且前向推理需要大量的计算力。我们希望深度卷积网络可以在嵌入式平台部署，并且希望在保证精度的情况下加快它的推理速度。常规的基于FFT的卷积对于大型滤波器是快速的，但是现有技术的卷积神经网络一般使用小的3×3滤波器。论文引入了Winograd的最小滤波算法，一种新的卷积神经网络快速算法。算法计算出小卷积上的最小复杂度，这使得它在过滤器和批量小的情况下更快。论文使用VGG网络对算法的GPU实现进行基准测试，并显示批处理大小从1到64的时时吞吐量。

>Cong和Xiao使用Strassen算法进行快速矩阵乘法，以减少卷积网络层中的调度次数，从而降低其总算术复杂度。 作者还提出，来自算术复杂性理论的更多技术可能适用于衔接。[1]

### 卷积公式

&#8195;&#8195;假设卷积公式如下，

<img src="http://www.forkosh.com/mathtex.cgi? Y_{ i }_{ , }_{ k }_{ , }_{ x }_{ , }_{ y }=\sum _{ c=1 }^{ C }{ \sum _{ v=1 }^{ R }{ \sum _{ u=1 }^{ S }{ D_{ i }_{ , }_{ c }_{ , }_{ x }_{ + }_{ u }_{ , }_{ y }_{ + }_{ v }G_{ k }_{ , }_{ c }_{ , }_{ x }_{ , }_{ y } }  }  } ">




参考文献

##### 1. Cong J., Xiao B. (2014) [Minimizing Computation in Convolutional Neural Networks.](https://link.springer.com/chapter/10.1007/978-3-319-11179-7_36) In: Wermter S. et al. (eds) Artificial Neural Networks and Machine Learning – ICANN 2014. ICANN 2014. Lecture Notes in Computer Science, vol 8681. Springer, Cham

##### 2. Lavin A, Gray S. [Fast algorithms for convolutional neural networks[C]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 4013-4021.