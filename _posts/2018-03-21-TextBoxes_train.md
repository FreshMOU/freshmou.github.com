---
layout: post
title: 用ICDAR2015数据训练TextBoxes
date: 2018-03-21
tags: 深度学习
---

### ICDAR2015

&#8195;&#8195;国际文档分析与识别大会（ICDAR）是全球文档分析以及模式识别领域最重要的国际学术会议之一，由国际模式识别协会(International Association of Pattern Recognition, IAPR)主办。该会议每两年举办1次，同时会举办ICDAR竞赛。ICDAR竞赛主要是考验模型对文字的定位和识别的准确度。

&#8195;&#8195;在ICDAR2015比赛中，官方提供标有转录文字及其位置的图像，这就是ICDAR2015数据集。

### TextBoxes训练

&#8195;&#8195;之前提到TextBoxes_plusplus是由TextBoes和crnn合并得到的，其是由两个模型来分别进行位置检测和文字识别的。今天先把TextBoxes的训练先搞定了。

#### 数据格式转化

&#8195;&#8195;ICDAR2015提供的数据是由txt格式([gt_img_1.txt](https://github.com/FreshMOU/scripts-for-myself/formatConversion/examples/gt_img_1.txt))保存的，其中的格式为[x1,y1,x2,y2,x3,y3,x4,y4,text]，而TextBoxes需要的是xml文件([example.xml](https://github.com/FreshMOU/scripts-for-myself/formatConversion/examples/example.xml))，所以我们先要把数据转过来。

&#8195;&#8195;TextBoxes是基于caffe实现的，caffe训练用的数据类型为lmdb格式的，要得到自己的lmdb文件需要有一个txt文件，其中的保存格式为：  
&#8195;&#8195;your_path_to_img.jpg your_path_to_xml.xml  
&#8195;&#8195;训练需要两个lmdb，一个train_lmdb，一个test_lmdb，所以需要两个txt文件。

&#8195;&#8195;为了快速得到最后的train.txt和test.txt，我用python写了一个[脚本](https://github.com/FreshMOU/scripts-for-myself/formatConversion/icdrtxt2xml.py)来对数据格式进行转换。