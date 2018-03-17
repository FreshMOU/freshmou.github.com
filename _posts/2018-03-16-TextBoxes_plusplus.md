---
layout: post
title: TextBoxes_plusplus在ubuntu14.04上的编译
date: 2018-03-16
tags: 深度学习
---

### TextBoxes_plusplus

&#8195;&#8195;TextBoxes_plusplus是基于TextBoxes改进的用于场景文字识别的项目，它用SSD来检测文字，然后对框出来的文字用CRNN进行识别。因为最近要用TextBoxes_plusplus，然后在编译它的过程中也遇到了一些问题，所以用这篇博客记录一下。

&#8195;&#8195;首先还是要先执行`git clone https://github.com/MhLiao/TextBoxes_plusplus.git`

#### 编译TextBoxes

&#8195;&#8195;TextBoxes是在caffe上实现的，直接按照普通caffe编译即可。  
&#8195;&#8195;在编译前需要安装caffe的各种依赖，依赖安装可以网上查询。  
&#8195;&#8195;编译时，根据不同的系统环境，需要修改主目录下的Makefile.config

```Makefile
USE_CUDNN := 1
CUDA_DIR := /usr/local/cuda-8.0
WITH_PYTHON_LAYER := 1
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/
```

&#8195;&#8195;接下来执行

```
make -j4
make py
```

&#8195;&#8195;在make 的过程中我有遇到error: ‘xxx’ does not name a type，经过查询了解到是由于Makefile中的一句话  

&#8195;&#8195;`COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-isystem $(includedir))`

&#8195;&#8195;其中 -isystem 是 gcc 参数，表示引用路径，但是当 -isystem 里面如果与 -I 里面的头文件有冲突会忽略 -I，所以如果系统中其他地方有同名文件，此时就不会进行本地的头文件搜索。

&#8195;&#8195;由于我的服务器中有安装另一个ssd的caffe环境，所以发生了错误。只要修改`-system`为`-I`即可。

&#8195;&#8195;这样TextBoxes就编译完了

#### 编译crnn

&#8195;&#8195;接下来就是crnn的配置，crnn编译前也需要安装多个依赖。

&#8195;&#8195;首先是torch7的安装，torch7安装比较简单，可以直接网上找教程。

&#8195;&#8195;然后是LMDB，如果系统中没有LMDB，可以直接`apt-get install liblmdb-dev`

&#8195;&#8195;再是fblualib的编译，fblualib确实有点难编，因为对于 ubuntu14.04，fblualib都是编译的老版本的库。

```shell
git clone https://github.com/facebookarchive/fblualib.git
cd fblualib
./install_all.sh
```

&#8195;&#8195;此时，系统就开始自动下载那些需要的库并自动编译，在编译fbthrift时会报`autoconf`的错误，我查询半天无果，所以只能自己手动编译。找到install_all.sh脚本下载的 fbthrift 位置，将其删除，然后执行:

```shell
git clone https://github.com/facebook/fbthrift
cd build
cmake ..
make
```

&#8195;&#8195;此时又会有很多的错误，这是因为fbthrift这个库也有很多的依赖，所以要自己手动再将依赖装好，具体缺什么依赖可以看报的错误。然后依赖的地址可以去[fbthrift](https://github.com/facebook/fbthrift)看。

&#8195;&#8195;接下来就可以继续编译thpp了，将install_all.sh中编译thpp的代码手动输入，但还是发生了问题，gtest-1.7.0文件不存在，这是由于google将gtest-1.7.0换了个位置存。此时找到thpp的目录，手动编译

```shell
cd $dir/thpp/thpp				（$dir是thpp在当前系统中的位置，需要自己去找）
curl -JLOk https://github.com/google/googletest/archive/release-1.7.0.zip
unzip googletest-release-1.7.0.zip
mv googletest-release-1.7.0 gtest-1.7.0
mkdir –p build
cd build
cmake ..
make
ctest
sudo make install
```

&#8195;&#8195;这样thpp就编译好了，接下来再编译fblualib

```shell
cd ../../fblualib/fblualib
./build.sh
```

&#8195;&#8195;这样整个 fblualib 就编译完了。  
&#8195;&#8195;此时就可以返回 TextBoxes_plusplus 目录去编译 crnn 了

```shell
cd ~/TextBoxes_plusplus/crnn/src
sh build_cpp.sh
```

&#8195;&#8195;最后要测试一下，官方提供了两个模型。

&#8195;&#8195;一个是TextBoxes的caffe模型 [BaiduYun](https://pan.baidu.com/s/1bqekTun)，下载来之后放到`./models`

&#8195;&#8195;一个是crnn的torch模型 [BaiduYun](https://pan.baidu.com/s/1jJwmneI)，下载来之后放到`./crnn/model/`

&#8195;&#8195;现在可以在主目录执行 `python examples/text/demo.py`

&#8195;&#8195;最终结果将被保存在`./demo_images`

