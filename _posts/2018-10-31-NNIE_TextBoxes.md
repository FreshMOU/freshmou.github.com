---
layout: post
title: TextBoxes_plusplus的NNIE实现文档
date: 2018-10-31
tags: HiSi
---

### 模型转换

在执行代码之前，我们先要将caffe的模型转换为nnie的模型（目前nnie只支持caffe模型转换）。

假设我们手上现在有tbpp的模型，即一个caffemodel文件和一个depoly.prototxt文件，现在要将模型转换为nnie的模型wk文件。<font color='red'>下面是一些模型转换时需要注意的地方。

#### nnie_mapper配置

将二进制文件解压出来即可使用。

#### prototxt文件配置

按照HiSVP开发指南的说法，prototxt需要遵循一定的格式。在这里我们只需要将需要CPU实现的层删除即可，这些层分别为<font color='red'>PriorBox层、Softmax层和DetectionOutput层</font>（虽然文档说似乎是支持的<font color='red'>Flatten层</font>的，但sample里是在CPU中实现的，所以也要删除）。
网络从前往后只保留到各个Permute层，Permute后面的层要全部删除。

然后input层也需要遵循格式要求
##### deploy.prototxt 输入层格式
deploy.prototxt 输入层支持如下两种格式，n 维度的 dim 值建议写 1，mapper 会根据参考图片路径中的图片张数自动生成 n 值: 
格式一:
 
```
input: "data"
input_shape{
   dim:1
   dim:3
   dim:224
   dim:224
}
```

格式二:

```
layer {
name: "data"
type: "Input"
top: "data"
input_param {
   shape: {
      dim: 1
      dim: 3
	  dim: 227
	  dim: 227 
    }
  } 
}
```

##### 中间上报层

如果想要将某些中间层的结果抽取出来，可以使用`report`关键词

用户需要中间层结果输出时，需要对应层的`top`域中添加`_report`标识符进行标注。

-  top 后续无节点，自然上报，_report 不增加上报点;
- top 对应的后续节点有多个 bottom，且其中一个 bottom 是 cpu 层，则该 top 上报;
- top 对应的后续节点是 cpu 层(其中，cpu 层指 proposal、custom、_cpu 层);
- top 有后续节点，_report 增加上报点;
- custom 有 top 加_report，报错;
- proposal 有 top 加_report，不报错，也不增加上报点;
- _cpu 有 top 加_report，不报错，也不增加上报点;
- data 层加_report，不会报错，也不会增加上报点;
- inplace 激活，_report 应加在 conv 层上，原因是多个激活共享了 conv 的 blob，因此这些层只输出一个 blob，加在激活层上不会报错，也不会增加上报点;
- conv 加激活，如果用户想在 conv 层上报，必须把两个节点拆开(激活写成 non- inplace 方式，即激活的 top、bottom 不同名);

##### 指定任意层高精度

用户指定自定义计算精度(compile_mode = 2)时，在对应层的层名后加上高精度`_hp`(16 比特)标记，可实现指定任意层为高精度输入。格式如下所示

```
layer {
    name: "conv5_hp"
    type: "Convolution"
    bottom: "conv4"
    top: "conv5"
    convolution_param {
       num_output: 256
       kernel_size: 3
       pad: 1
       stride: 1
	} 
}
```

#### 使用nnie_mapper生成模型

nnie_mapper需要用户提供一个cfg文件，如下是常用cnn的基本配置（ssd也一样）。

```
[prototxt_file]  ./lenet.prototxt
[caffemodel_file] ./lenet_iter_10000.caffemodel
[batch_num] 0
[net_type] 0
[sparse_rate] 0
[compile_mode] 1
[is_simulation] 0
[log_level] 2
[instruction_name] ./lenet
[RGB_order] BGR
[data_scale] 0.0039062
[internal_stride] 16
[image_list] ./image_ref_list.txt
[image_type] 1
[mean_file] ./lenetmean.txt
[norm_type] 2
```

具体的参数说明可以去文档中查看，这里只稍微介绍一些常常要改的重要参数。
- `[compile_mode]`表示编译模式，默认为0，表示低精度高带宽。如果配置为1则是全网络高精度，这里的高精度其实也是有压缩的，是以16位int型计算的，如果配置为2的话则是部分层高精度，哪些层需要高精度需要用户自己配置，具体如何配置参照上面prototxt修改。
- `[image_list]`为NNIE mapper 用于数据量化的参考图像 list 文件或 feature map 文件。
NNIE mapper 量化时需要的图片是典型场景图片，建议从网络模型 的测试场景随机选择 20~50 张作为参考图片进行量化，选择的图像要尽量覆盖模型的各个场景(比如检测人、车的模型，参考图像中必须由人、车，不能仅使用人或者无人无车的图像进行量化)。网络中如果存在多个输入层，则需要配置多个 image_list 项，顺 序、个数与 prototxt 完全对应。
- `[mean_file]` 均值文件

配置好cfg文件后执行`nnie_mapper xxx.cfg`即可

模型转换基本的注意事项如上。

### sample代码修改

官方提供了ssd的sample代码，而TextBoxes_plusplus是基于ssd修改的，所以为了在NNIE上实现TextBoxes_plusplus，我基于官方的ssd代码进行了修改。

基本可以参照之前TextBoxes_plusplus基于ncnn实现的文档。<font color='red'>（注意内存的分配）</font>

不过还是有些不同的地方，ncnn处只需要修改PriorBoxes和DetectionOutput层，而nnie处还有其他地方代码需要修改，具体如下。

首先是`pstSoftWareParam`参数初始化。

NNIE输出的是一连串的数据，需要你自己来截断（换句话说，就是NNIE的输出是最原始的数据，除了数据，其他信息一点没有）
所以对于PriorBox层，需要提供各层的大小

```
pstSoftWareParam->au32PriorBoxWidth[0] = 48;
pstSoftWareParam->au32PriorBoxWidth[1] = 24;
pstSoftWareParam->au32PriorBoxWidth[2] = 12;
pstSoftWareParam->au32PriorBoxWidth[3] = 6;
pstSoftWareParam->au32PriorBoxWidth[4] = 4;
pstSoftWareParam->au32PriorBoxWidth[5] = 2;
```

对于softmax层和detecionout层需要提供输入的参数数量。

```
pstSoftWareParam->au32SoftMaxInChn[0] = 92160;
pstSoftWareParam->au32SoftMaxInChn[1] = 23040;
pstSoftWareParam->au32SoftMaxInChn[2] = 5760;
pstSoftWareParam->au32SoftMaxInChn[3] = 1440;
pstSoftWareParam->au32SoftMaxInChn[4] = 640;
pstSoftWareParam->au32SoftMaxInChn[5] = 160;

pstSoftWareParam->au32DetectInputChn[0] = 552960;
pstSoftWareParam->au32DetectInputChn[1] = 138240;
pstSoftWareParam->au32DetectInputChn[2] = 34560;
pstSoftWareParam->au32DetectInputChn[3] = 8640;
pstSoftWareParam->au32DetectInputChn[4] = 3840;
pstSoftWareParam->au32DetectInputChn[5] = 960;
```

由于TextBoxes_plusplus的ratio数量同ssd不同，它有5个ratio，所以对应也需要全部修改。反正同priorbox层相关的数据都是需要修改的。

```
pstSoftWareParam->au32InputAspectRatioNum[0] = 4;
pstSoftWareParam->af32PriorBoxAspectRatio[0][0] = 2;
```

以上都需要自己手动计算。
之后是根据上述参数计算所需内存并进行分配。


### 输入的格式

输入的格式文档中并没有说，我是根据后缀这些猜出来的，确实也是如此。直接依次排序保存像素值，图片大小需要自己记录下来。