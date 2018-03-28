---
layout: post
title: TextBoxes_plusplus的Textboxes部分基于ncnn实现
date: 2018-03-25
tags: 深度学习
---

### TextBoxes_plusplus的实现

&#8195;&#8195;TextBoxes是基于caffe实现的。针对其网络，作者修改了caffe的源码，主要是prior_box层和detection_output层。为了区别普通caffe代码，作者为其添加了两个参数，分别是prior_box层的`denser_prior_boxes`和detection_output层的`use_polygon`，在这个两个参数下，作者添加了自己的算法代码，所以我们可以通过这部分算法代码来对TextBoxes进行移植。

### TextBoxes caffe模型转ncnn模型

&#8195;&#8195;Ncnn提供了caffe转ncnn的tools  `./tools/caffe/caffe2ncnn.cpp`，但是由于`denser_prior_boxes`和`use_polygon`是Textboxes_plusplus自己集成的参数，所以ncnn并不支持，这里就要自己去添加了。

```
找到 else if (layer.type() == "PriorBox")
在if的末尾添加  fprintf(pp, " 14=%d", prior_box_param.denser_prior_boxes());
找到 else if (layer.type() == "DetectionOutput")，在if末尾添加  
fprintf(pp, " 5=%d", detection_output_param.use_polygon());
```

&#8195;&#8195;然后编译ncnn  
```shell
mkdir build
cd build
cmake ..
make -j4
./tools/caffe/caffe2ncnn your_path_deploy.prototxt your_path_model.caffemodel xxx.param xxx.bin
```

&#8195;&#8195;这样caffe模型就顺利的转成ncnn模型(`xxx.param`和`xxx.bin`)了，接下来就是如何使用ncnn模型的问题了。

### 为ncnn添加TextBoxes算法实现部分

&#8195;&#8195;先找到priorbox层的`load_param`，添加`denser_prior_boxes = pd.get(14, 0);`（别忘了在的.h文件中初始化），再找到detectionoutput层的`load_param`，添加`use_polygon = pd.get(5, 0);` 接下来就可以开始将Textboxes_plusplus的算法移植到ncnn上了。Priorbox层比较好修改，就不写了。

&#8195;&#8195;Detectionoutput层需要修改的地方如下：  
1.	添加Polygon的decode。
2.	对bbox添加一个序号以便追踪，这样可以在最后输出时输出对应的polygon，因为polygon不进行nms和sort。

&#8195;&#8195;看着简单，实现起来确实也简单，polygon的decode是算法的核心，具体caffe的实现集成在TextBoxes_plusplus中的detection_output层里的`DecodeAllBoxes`函数中，一层层点进去就可以看到这部分的算法代码了，一部部将它从caffe翻译过来就可以了。

&#8195;&#8195;然后是第二步，因为caffe在`keep_nms_k`和`keep_top_k`中是提取每个bbox的index保存下来，然后最后根据index来输出bbox和polygon，而ncnn不是，ncnn是直接舍弃掉了不需要的bbox，没有使用index，所以就无法直接输出与要输出的bbox相匹配的polygon，所以需要对bbox添加一个序号，类似于caffe的index(indices)。

&#8195;&#8195;这样ncnn基本可以运行 Textboxes_plusplus 的TextBoxes部分了，但是可能还是会有一个问题，这个问题我也是调了好久才发现。本来以为是ncnn的长方形卷积有问题，一步步看下来发现原来是 caffe2ncnn.cpp 中没有把网络的 pad_h 和 pad_w 转进来，所以如果你也有这个问题，不妨去看看 caffe2ncnn.cpp 的 convolution 转参数部分。

### END

&#8195;&#8195;模型转完了，也可以使用了，似乎一切都结束了，但是其实下面还有一个难点，就是你要怎么给不规则四边形施加nms。已知2个四边形共8个点坐标，如何计算它俩的iou？具体可以看我[下一篇博客](./2018-3-31-compute_iou_of_quad.md)。

&#8195;&#8195;如果有需要，可以参考我的代码，[点击这里](https://github.com/FreshMOU/ncnn)
