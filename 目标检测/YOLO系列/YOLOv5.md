## YOLOV5 目标检测原理

源码仓库地址：https://github.com/ultralytics/yolov5
源码目录结构：
- data
  - images:包含测试的图片
  - scripts:用于下载数据集的脚本
  - yaml文件：主要包括具体数据集的 yaml 文件以及训练超参数的 yaml 文件
- models: 主要用于构建模型的目录
- utils: 用于存放小工具代码目录
- train.py：训练程序
- test.py：测试程序
- detect.py：检测程序

#### 模型构建

`models` 目录下包含许多和模型相关的 yaml 文件，例如 `yolov5l.yaml` 等。这些 yaml 文件主要包含以下几个内容：
- nc:表示训练数据集的类别数目
- depth_multiple：深度因子，yolo 中主要使用该参数控制某些模块的重复次数
- width_multiple：宽度因子，主要用于控制特征图通道数目
- anchors: 锚框
- backbone: 骨架网络的参数
- head: 包括多尺度融合和预测头部分的网络参数

yolov5 初始化了 9 中不同尺度的锚框类型，分别在不同层次的特征图上使用，浅层的特征图对应的感受野较小，主要用于检测小目标，所以对应的锚框尺度也相对小。深层特征图的下采样倍率较大，感受野也相对较大，适合检测大目标，所以锚框相对大。由于 yolov5 主要使用了三层的特征图大小用于目标检测，故每个层级的特征图对应于3个锚框尺度。

backbone 参数解析：
- from 表示从那一层获得输入数据， -1 表示上一层
- number 表示该模块重复的次数
- module 表示网络层的名字
- args 表示模块初始化的参数，例如卷积层的卷积核大小，步长等信息
```yaml
# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, C3, [1024, False]],  # 9
  ]
```

模块重复的次数乘以深度因子，即得到实际的模块重复次数

```python
anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
n = max(round(n * gd), 1) if n > 1 else n
```

`common.py` 包含了网络中使用的各个子模块部件。例如， `Conv` 主要是将普通的卷积+BN+激活函数打包成一个模块。

```python
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
```

构建网络模型的源代码：

```python
def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    
    # layers 按照顺序存放各个模块实例化对象
    # save 用于保存输入不是来自前一层的索引，方便后续不同层特征图融合时读取数据
    # c2 ch[-1] 表示其实的输入通道数目，默认是 3 , c2 表示每一层的输出通道数目
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain 计算该模块重复的次数
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:
            c1, c2 = ch[f], args[0] # c1 表示输入通道数目， c2 表示输出通道数目
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8) # 乘以宽度因子计算实际的输出通道数目

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2) # ch 用于记录每个模块输出的通道数目
    return nn.Sequential(*layers), sorted(save)
```

`class Model` 实现包裹了实际的 yolov5 模型，同时包含了前向传播的计算，NMS 计算，初始化参数等方法。
- `__init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None)`
```python
def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
    super(Model, self).__init__()
    if isinstance(cfg, dict):
        self.yaml = cfg  # model dict
    else:  # is *.yaml
        import yaml  # for torch hub
        self.yaml_file = Path(cfg).name
        with open(cfg) as f:
            self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # model dict

    # Define model
    ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
    if nc and nc != self.yaml['nc']:
        logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
        self.yaml['nc'] = nc  # override yaml value
    if anchors:
        logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
        self.yaml['anchors'] = round(anchors)  # override yaml value
    self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
    self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
    # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

    # Build strides, anchors
    m = self.model[-1]  # Detect()
    if isinstance(m, Detect): # 判断最后一层是 Detect 层
        s = 256  # 2x min stride
        # 获取下采样倍率，默认是 [8, 16, 32]
        m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        m.anchors /= m.stride.view(-1, 1, 1)
        check_anchor_order(m)
        self.stride = m.stride
        self._initialize_biases()  # only run once
        # print('Strides: %s' % m.stride.tolist())

    # Init weights, biases
    initialize_weights(self)
    self.info()
    logger.info('')
```
- `forward`
```python
def forward(self, x, augment=False, profile=False):
    if augment: # 数据增强的前向传播
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi[..., :4] /= si  # de-scale
            if fi == 2:
                yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
            elif fi == 3:
                yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
            y.append(yi)
        return torch.cat(y, 1), None  # augmented inference, train
    else: # 普通的前向传播
        return self.forward_once(x, profile)  # single-scale inference, train
```
- `forward_once`
```python
def forward_once(self, x, profile=False):
    y, dt = [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer 需要融合多层特征图的模块
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

        if profile:
            o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPS
            t = time_synchronized()
            for _ in range(10):
                _ = m(x)
            dt.append((time_synchronized() - t) * 100)
            print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output 保存后续需要融合的特征图

    if profile:
        print('%.1fms total' % sum(dt))
    return x
```
- `fuse` 融合卷积和BN层的运算，加速推理速度
```python
def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
    print('Fusing layers... ')
    for m in self.model.modules():
        if type(m) is Conv and hasattr(m, 'bn'):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
            delattr(m, 'bn')  # remove batchnorm
            m.forward = m.fuseforward  # update forward
    self.info()
    return self
```
![image](https://user-images.githubusercontent.com/62278179/222716572-963eb35e-78fb-43fd-a73e-333eb7b4841b.png)

- `nms` 用于在模型尾部添加 NMS 模块
```python
def nms(self, mode=True):  # add or remove NMS module
    present = type(self.model[-1]) is NMS  # last layer is NMS
    if mode and not present:
        print('Adding NMS... ')
        m = NMS()  # module
        m.f = -1  # from
        m.i = self.model[-1].i + 1  # index
        self.model.add_module(name='%s' % m.i, module=m)  # add
        self.eval()
    elif not mode and present:
        print('Removing NMS... ')
        self.model = self.model[:-1]  # remove
    return self
```

`Detect` 检测头部分源码：
```python
class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor 5 表示目标框的坐标参数以及置信度
        self.nl = len(anchors)  # number of detection layers 表示有几层输出，yolov5中一共有三层输出预测，分别对应不同的尺度大小
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)]) # 生成网格化的坐标
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
```

整体的网络示意图：

![image](https://user-images.githubusercontent.com/62278179/222720439-8ca58107-4099-444f-a2e7-aa69d47ff6fd.png)

损失函数的相关介绍：https://zhuanlan.zhihu.com/p/582265998
评价指标参数：https://zhuanlan.zhihu.com/p/584772395
TP：预测的框与真实框的 IOU 高于设置的阈值
FP：错误的正样本，也即预测的框与真实框的 IOU 低于设置的阈值
FN：错误的负样本，也就是存在目标的区域没有预测出目标框
TN：一般目标检测中不指明该参数，也就是背景没有被预测成目标，该参数没有实际意义
mAP: 表示类别 AP 平均精度
PR曲线：由于设置阈值的不同会导致精度(P)和召回率(R)不同，从而可以构建两者的 PR 曲线
AP: 表示平均精度，是针对某一类别来计算的，通常是 PR 曲线下方的面积
