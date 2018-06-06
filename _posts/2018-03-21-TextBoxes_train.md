---
layout: post
title: 用ICDAR2015数据训练TextBoxes++
date: 2018-03-21
tags: 深度学习
---

### ICDAR2015

&#8195;&#8195;国际文档分析与识别大会（ICDAR）是全球文档分析以及模式识别领域最重要的国际学术会议之一，由国际模式识别协会(International Association of Pattern Recognition, IAPR)主办。该会议每两年举办1次，同时会举办ICDAR竞赛。ICDAR竞赛主要是考验模型对文字的定位和识别的准确度。

&#8195;&#8195;在ICDAR2015比赛中，官方提供标有转录文字及其位置的图像，这就是ICDAR2015数据集。

### TextBoxes++训练

&#8195;&#8195;之前提到TextBoxes_plusplus是由TextBoes和crnn合并得到的，其是由两个模型来分别进行位置检测和文字识别的。今天先把TextBoxes的训练先搞定了。

&#8195;&#8195;ICDAR2015提供的数据是由txt格式([gt_img_1.txt](https://github.com/FreshMOU/scripts-for-myself/blob/master/formatConversion/examples/gt_img_1.txt))保存的，其中的格式为`[x1,y1,x2,y2,x3,y3,x4,y4,text]`，而TextBoxes需要的是xml文件([example.xml](https://github.com/FreshMOU/scripts-for-myself/blob/master/formatConversion/examples/example.xml))，所以我们先要把数据转过来。

&#8195;&#8195;TextBoxes是基于caffe实现的，caffe训练用的数据类型为lmdb格式的，要得到自己的lmdb文件需要有一个txt文件，其中的保存格式为：  
```
    path_to_example1.jpg path_to_example1.xml
    path_to_example2.jpg path_to_example2.xml
```

&#8195;&#8195;训练需要两个lmdb，一个train_lmdb，一个test_lmdb，所以需要两个txt文件。

&#8195;&#8195;为了快速得到最后的train.txt和test.txt，我用python写了一个[脚本](https://github.com/FreshMOU/scripts-for-myself/blob/master/formatConversion/icdrtxt2xml.py)来对数据格式进行转换。将该脚本放在TextBoxes_plusplus主目录的`./data/icdar2015`目录下运行，运行结束后即可得到在`./data/text`目录下的train.txt和test.txt（注意icdar2015的图片数据解压在`./data/icdar2015/image`目录下，ground truth数据解压在`./data/icdar2015/local_gt`下）。

&#8195;&#8195;接下来直接执行`./data/text/creat_data.sh`即可生成lmdb文件（在生成lmdb文件的时候一路报问题no such node(annotation.size.height)和bounding box irregular，猜测应该是由于xml文件写的框为不规则四边形的缘故，最后还报了一个链接错误不过没有什么影响，可以忽略），注意要修改`./examples/modelConfig.py`文件中lmdb的路径。

&#8195;&#8195;下载预训练模型放到`./models`下就可以开始训练了。执行`python examples/text/train.py`。
