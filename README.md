## YOLOv4 Inference
### 简要介绍

本项目使用 PyTorch 构建 YOLOv4 模型, 没有在代码的层面上去解析 yolov4.cfg 配置文件, 而是根据 [Netron](https://github.com/lutzroeder/netron) 可视化的结果, 直接搭建模型

<br>

### 目录结构

```
.
├── image
│   └── demo.png      默认使用该图像推理
├── main.py           解析命令行和推理
├── model.py          YOLOv4 的网络模型
├── README.md
├── requirements.txt
└── utils.py          过滤和绘制目标框

1 directory, 6 files
```

<br>

### 网络模型

- shortcut: 将两个相同维度的特征图相加, 相加后维度不变, 其后可接激活函数, <font color="brown">如果使用 "linear" 激活函数, 则表示直接输出</font>
- route: 将相同尺寸的特征图叠加

#### 向上采样

使用 2*2 数组示例

```python
import torch
# [[0, 1]
#  [2, 3]]
img = torch.arange(4).view(2,2)
# [[0, 0, 1, 1]
#  [0, 0, 1, 1]
#  [2, 2, 3, 3]
#  [2, 2, 3, 3]]
ex = img.view(2,1,2,1).expand(-1,2,-1,2).contiguous().view(4,4)
```

<br>

### 加载权重

使用 PyTorch 定义网络模型时, 在 `__init__` 函数中所有卷积层出现的顺序与 `yolov4.cfg` 中的 "convolutional" 的顺序一致, 所以仅需依次加载 `yolov4.weights` 



YOLOv4 中的卷积层有两种, 一种是后接 BatchNorm 但没有 Bias 的卷积层; 另一种是在末端 YOLO Layer 的卷积层, 这一种带有偏置 Bias 但后面没有 BatchNorm

<br>

### 运行推理

```shell
$ python3 main.py  --weight ~/Datasets/YOLOv4.weights
```

默认使用 image 下的 demo.png

![demo](image/demo.png)

推理图片默认命名为 detect.png, 且与原始图片位于同一目录

![detect.png](https://i.loli.net/2020/09/11/uzIJ7iso3GaY8UX.png)

<br>

### 进度

- [x] Infer
- [ ] Train
- [ ] TensorRT