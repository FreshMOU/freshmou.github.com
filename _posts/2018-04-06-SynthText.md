---
layout: post
title: TextBoxes++使用SynthText数据集
date: 2018-04-06
tags: 深度学习
---

### SynthText数据集

&#8195;&#8195;SynthText(synthetic text)其实是指用代码合成的文本图像数据，它的源码在https://github.com/ankush-me/SynthText，如果有需要你可以用它的源码来合成自己的文本训练集。我们这里使用的是用这种方法得到的官方提供[数据集SynthText](http://www.robots.ox.ac.uk/~vgg/data/scenetext/)，这个数据集包含了80万张图像，其中融入了800万个文本。

### 数据格式转成适用于TextBoxes++的xml格式

&#8195;&#8195;有许多语言可以读取并处理mat格式文件，我在这里选用python来处理。

```python
# 读取gt.mat数据
import scipy.io as sio
data = sio.loadmat('gt.mat')
```

&#8195;&#8195;之前我直接用string来规范文本数据，但是并不如xml包来的好管理。

```python
import scipy.io as sio
import numpy as np
import xml.dom.minidom
import sys
import random
import os

def MatRead(matfile):
    data = sio.loadmat(matfile)

    train_file = open('train.txt', 'w')
    test_file = open('test.txt', 'w')
    
    for i in range(len(data['txt'][0])):
        contents = []
        for val in data['txt'][0][i]:
            v = [x.split("\n") for x in val.strip().split(" ")]
            contents.extend(sum(v, []))
        print >> sys.stderr, "No.{} data".format(i)
        rec = np.array(data['wordBB'][0][i], dtype=np.int32)
        if len(rec.shape) == 3:
            rec = rec.transpose(2,1,0)
        else:
            rec = rec.transpose(1,0)[np.newaxis, :]

        doc = xml.dom.minidom.Document() 
        root = doc.createElement('annotation') 
        doc.appendChild(root) 
        print("start to process {} object".format(len(rec)))
        
        for j in range(len(rec)):
            nodeobject = doc.createElement('object')
            nodecontent = doc.createElement('content')
            nodecontent.appendChild(doc.createTextNode(str(contents[j])))

            nodename = doc.createElement('name')
            nodename.appendChild(doc.createTextNode('text'))

            bndbox = {}
            bndbox['x1'] = rec[j][0][0]
            bndbox['y1'] = rec[j][0][1]
            bndbox['x2'] = rec[j][1][0]
            bndbox['y2'] = rec[j][1][1]
            bndbox['x3'] = rec[j][2][0]
            bndbox['y3'] = rec[j][2][1]
            bndbox['x4'] = rec[j][3][0]
            bndbox['y4'] = rec[j][3][1]
            bndbox['xmin'] = min(bndbox['x1'], bndbox['x2'], bndbox['x3'], bndbox['x4'])
            bndbox['xmax'] = max(bndbox['x1'], bndbox['x2'], bndbox['x3'], bndbox['x4'])
            bndbox['ymin'] = min(bndbox['y1'], bndbox['y2'], bndbox['y3'], bndbox['y4'])
            bndbox['ymax'] = max(bndbox['y1'], bndbox['y2'], bndbox['y3'], bndbox['y4'])

            nodebndbox = doc.createElement('bndbox')
            for k in bndbox.keys():
                nodecoord =  doc.createElement(k)
                nodecoord.appendChild(doc.createTextNode(str(bndbox[k])))
                nodebndbox.appendChild(nodecoord)

            nodeobject.appendChild(nodecontent)
            nodeobject.appendChild(nodename)
            nodeobject.appendChild(nodebndbox)
            root.appendChild(nodeobject)

        filename = data['imnames'][0][i][0].replace('.jpg', '.xml')
        fp = open(filename, 'w')
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")
        fp.close()
        rad = random.uniform(10,20)
        pwd = os.getcwd()
        img_path = os.path.join(pwd, data['imnames'][0][i][0])
        xml_path = os.path.join(pwd, filename)
        file_line = img_path + " " + xml_path + '\n'
        if rad > 18:
            train_file.write(file_line)
        else:
            test_file.write(file_line)    

    train_file.close()
    test_file.close()
```

### 生成lmdb数据

&#8195;&#8195;将`train.txt`和`test.txt`移到`TextBoxes_plusplus/data/text`目录下，再执行`./create_data.sh`即可得到lmdb文件。