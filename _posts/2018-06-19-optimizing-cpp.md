---
layout: post
title: C++代码优化
date: 2018-06-19
tags: 性能优化
---

### 前言

&#8195;&#8195;相对于其他简单的高级语言来说，C++可以更好地发挥硬件性能，但就算是相同的算法，代码写的不好，C++也不一定比其他语言就快很多。一般来说，没有一种简单的方法可以完美优化所有情况，所以优化只是尽可能接近最完美的情况。

&#8195;&#8195;一般优化过程中有两个标准：  
-Principle of diminishing returns.  先从耗时耗力少且优化效果好的部分开始着手优化。  
-Principle of diminishing portability.  先从跨平台通用的代码开始优化。





<br/>

参考文献

##### [optimizing C++](https://en.wikibooks.org/wiki/Optimizing_C%2B%2B)