---
layout: post
title: DockerFile
date: 2018-09-22
tags: 工具
---

### 前言

前面已经介绍了DockerFile的几个优点了，其实主要的优点就是它有一个系统的步骤，可以让别人很容易理解，并且很容易知道在基本映像中改变了什么确切的配置。

### DockerFile编写

DockerFile其实就是一种被docker解释的脚本，由多条指令组成，每条指令对应linux下的指令。

Dockerfile的指令是忽略大小写的，建议使用大写，使用#作为注释，每一行只支持一条指令，每条指令可以携带多个参数。

常用命令如下。

#### FROM（指定初始镜像）

用法：`FROM <image>:<tag>`

一般从官方仓库直接拉取。

#### MAINTAINER（制定镜像制作者信息）

将镜像的制作者相关的信息写入到镜像中。

用法：`MAINTAINER <name>`

#### RUN（调用命令）

RUN可以运行任何被基础镜像支持的命令。如基础镜像选择了ubuntu，那么软件管理能使用ubuntu的命令。

用法：`RUN <shell command>`

#### ENV（设置环境变量）

用法：`ENV <key> <value>`

#### ADD（添加文件）

将主机的文件添加至镜像中，一般用来替换镜像源。

用法：`ADD <src> <dst>`

#### WORKDIR（切换目录）

用法：`WORKDIR <dir>`

### DockerFile用docker build命令来构建镜像

docker build 命令用于使用 Dockerfile 创建镜像。

一般常用：
`docker build -t <name> DockerFilePath`

语法
`docker build [OPTIONS] PATH | URL | -`

OPTIONS说明：
```
--build-arg=[] :设置镜像创建时的变量；

--cpu-shares :设置 cpu 使用权重；

--cpu-period :限制 CPU CFS周期；

--cpu-quota :限制 CPU CFS配额；

--cpuset-cpus :指定使用的CPU id；

--cpuset-mems :指定使用的内存 id；

--disable-content-trust :忽略校验，默认开启；

-f :指定要使用的Dockerfile路径；

--force-rm :设置镜像过程中删除中间容器；

--isolation :使用容器隔离技术；

--label=[] :设置镜像使用的元数据；

-m :设置内存最大值；

--memory-swap :设置Swap的最大值为内存+swap，"-1"表示不限swap；

--no-cache :创建镜像的过程不使用缓存；

--pull :尝试去更新镜像的新版本；

--quiet, -q :安静模式，成功后只输出镜像 ID；

--rm :设置镜像成功后删除中间容器；

--shm-size :设置/dev/shm的大小，默认值是64M；

--ulimit :Ulimit配置。

--tag, -t: 镜像的名字及标签，通常 name:tag 或者 name 格式；可以在一次构建中为一个镜像设置多个标签。

--network: 默认 default。在构建期间设置RUN指令的网络模式
```

### TextBoxes_plusplus的CPU版DockerFile

因为TextBoxes_plusplus没有提供CPU版本的DockerFile，而我的mac又不支持GPU，所以根据其提供的GPU版本略微修改了DockerFile。（修改了镜像源以加速配置）

```shell
FROM ubuntu:14.04
MAINTAINER caffe-maint@googlegroups.com

RUN mv /etc/apt/sources.list /etc/apt/sources-bak.list
# sources.list在当前目录下，里面是修改的阿里云镜像源
ADD sources.list /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
	    libgeos-dev \
        protobuf-compiler \
        python-dev \
        python-numpy \
        python-pip \
        python-scipy \
	    python-opencv && \
    rm -rf /var/lib/apt/lists/*

# 同上配置阿里云源
ADD pip.conf /root/.pip/pip.conf

ENV CAFFE_ROOT=/opt/caffe
WORKDIR $CAFFE_ROOT

# FIXME: clone a specific git tag and use ARG instead of ENV once DockerHub supports this.
ENV CLONE_TAG=master

ARG CLONE_REPO

RUN git clone -b ${CLONE_TAG} --depth 1 $CLONE_REPO .

RUN for req in $(cat python/requirements.txt) pydot; do pip install $req; done && \
    mkdir build && cd build && \
    cmake -DCPU_ONLY=1 .. && \
    make -j"$(nproc)"

RUN ln -s /dev/null /dev/raw1394

ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace
```

使用方法：
`docker build -t tbpp:cpu --build-arg CLONE_REPO=$(git remote get-url --all origin) DockerFilePATH`