# 卷积模块文档

## 概述

`causal_bald.library.modules.convolution` 模块提供了用于构建卷积神经网络的各种组件，包括激活函数、预激活卷积层、残差卷积层和 ResNet 架构。

## 主要类

### `Activation`

2D 激活函数模块，包含批归一化、激活函数和 Dropout。

**初始化参数：**
- `num_features` (int): 特征图数量（通道数）
- `negative_slope` (float): LeakyReLU 的负斜率（< 0 时使用 ELU）
- `dropout_rate` (float): Dropout 率
- `batch_norm` (bool): 是否使用批归一化

**结构：**
```
BatchNorm2d (可选) → LeakyReLU/ELU → Dropout2d
```

**使用示例：**
```python
from causal_bald.library.modules import convolution

activation = convolution.Activation(
    num_features=64,
    negative_slope=-1.0,  # 使用 ELU
    dropout_rate=0.1,
    batch_norm=True,
)
```

### `PreactivationConv`

预激活卷积层，采用 Preactivation 设计。

**初始化参数：**
- `dim_input` (list): 输入维度 `[C, H, W]`
- `out_channels` (int): 输出通道数
- `kernel_size` (int, 默认=3): 卷积核大小
- `stride` (int, 默认=1): 步长
- `padding` (int, 默认=1): 填充
- `dilation` (int, 默认=1): 膨胀率
- `groups` (int, 默认=1): 分组数
- `bias` (bool, 默认=True): 是否使用偏置
- `negative_slope` (float, 默认=1.0): LeakyReLU 负斜率
- `dropout_rate` (float, 默认=0.0): Dropout 率
- `batch_norm` (bool, 默认=False): 是否使用批归一化
- `spectral_norm` (float, 默认=0.0): 谱归一化系数

**结构：**
```
Activation → Conv2d (可选谱归一化)
```

**使用示例：**
```python
layer = convolution.PreactivationConv(
    dim_input=[3, 32, 32],
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    negative_slope=-1.0,
    dropout_rate=0.1,
    batch_norm=True,
    spectral_norm=0.95,
)
```

### `ResidualConv`

残差卷积层，支持残差连接。

**初始化参数：**
- `dim_input` (list): 输入维度 `[C, H, W]`
- `out_channels` (int): 输出通道数
- `bias` (bool): 是否使用偏置
- `negative_slope` (float): LeakyReLU 负斜率
- `dropout_rate` (float): Dropout 率
- `batch_norm` (bool): 是否使用批归一化
- `spectral_norm` (float): 谱归一化系数
- `stride` (int): 步长

**结构：**
```
output = PreactivationConv(x) + shortcut(x)
```

**Shortcut 连接：**
- 如果 `dim_input[0] == out_channels` 且 `stride == 1`：使用恒等映射
- 否则：使用 `Dropout2d → Conv2d(1x1)` 投影

**使用示例：**
```python
layer = convolution.ResidualConv(
    dim_input=[64, 32, 32],
    out_channels=64,  # 相同通道数
    bias=True,
    negative_slope=-1.0,
    dropout_rate=0.1,
    batch_norm=True,
    spectral_norm=0.95,
    stride=1,
)
```

### `ResNet`

完整的 ResNet 架构，用于图像数据。

**初始化参数：**
- `dim_input` (list): 输入维度 `[C, H, W]`
- `layers` (List[int]): 每层的块数，如 `[2, 2, 2]` 表示 3 层，每层 2 个块
- `base_width` (int, 默认=64): 基础宽度（通道数）
- `negative_slope` (float, 默认=0.0): LeakyReLU 负斜率
- `dropout_rate` (float, 默认=0.0): Dropout 率
- `batch_norm` (bool, 默认=False): 是否使用批归一化
- `spectral_norm` (float, 默认=0.0): 谱归一化系数
- `stem_kernel_size` (int, 默认=7): Stem 卷积核大小
- `stem_kernel_stride` (int, 默认=2): Stem 步长
- `stem_kernel_padding` (int, 默认=3): Stem 填充
- `stem_pool` (bool, 默认=True): 是否使用 Stem 池化
- `activate_output` (bool, 默认=False): 是否激活输出

**结构：**
```
Stem Conv → MaxPool (可选)
    ↓
Layer 1 (base_width channels)
    ↓
Layer 2 (base_width * 2 channels)
    ↓
Layer 3 (base_width * 4 channels)
    ↓
...
    ↓
AdaptiveAvgPool2d(1, 1)
    ↓
Output Activation (可选)
```

**属性：**
- `dim_output`: 输出维度（等于最后一层的通道数）

**使用示例：**
```python
resnet = convolution.ResNet(
    dim_input=[3, 28, 28],  # CIFAR-10 / MNIST
    layers=[2, 2, 2],       # 3 层，每层 2 个块
    base_width=64,
    negative_slope=-1.0,
    dropout_rate=0.1,
    batch_norm=True,
    spectral_norm=0.95,
    stem_kernel_size=5,
    stem_kernel_stride=1,
    stem_kernel_padding=2,
    stem_pool=False,        # 小图像不使用池化
    activate_output=True,
)

# 前向传播
output = resnet(inputs)  # shape: (batch_size, dim_output)
```

## 架构细节

### Stem 层

ResNet 的初始层，用于快速下采样：

```python
stem_conv = Conv2d(
    in_channels=dim_input[0],
    out_channels=base_width,
    kernel_size=stem_kernel_size,
    stride=stem_kernel_stride,
    padding=stem_kernel_padding,
)
```

**配置建议：**
- **大图像**（224x224）：`kernel_size=7, stride=2, padding=3, pool=True`
- **小图像**（28x28, 32x32）：`kernel_size=5, stride=1, padding=2, pool=False`

### 残差块结构

每个残差块包含两个卷积层：

```python
ResidualConv(
    # 第一个卷积：保持维度
    PreactivationConv(dim_input, dim_input, stride=1),
    # 第二个卷积：可能改变维度
    PreactivationConv(dim_input, out_channels, stride=stride),
    # Shortcut
    shortcut,
)
```

### 通道扩展

每层通道数翻倍：
- Layer 1: `base_width` channels
- Layer 2: `base_width * 2` channels
- Layer 3: `base_width * 4` channels
- ...

### 空间下采样

通过 stride=2 的卷积进行下采样：
- 每层第一个块的最后一个卷积使用 `stride=2`
- 最后一层不使用下采样

## 使用场景

### 1. 图像分类
```python
resnet = convolution.ResNet(
    dim_input=[3, 224, 224],
    layers=[2, 2, 2, 2],
    base_width=64,
    activate_output=True,
)

# 添加分类头
classifier = nn.Sequential(
    resnet,
    nn.Linear(resnet.dim_output, num_classes),
)
```

### 2. 特征提取
```python
resnet = convolution.ResNet(
    dim_input=[3, 28, 28],
    layers=[2, 2, 2],
    base_width=64,
    activate_output=False,  # 不激活输出
)

# 提取特征
features = resnet(images)  # shape: (batch_size, 512)
```

### 3. 在 TARNet 中使用
```python
from causal_bald.library.modules import tarnet

model = tarnet.TARNet(
    dim_input=[3, 28, 28],  # 图像输入
    dim_hidden=200,
    dim_output=1,
    depth=3,
    # ...
)
```

## 完整示例

### 构建 ResNet-18
```python
resnet = convolution.ResNet(
    dim_input=[3, 224, 224],
    layers=[2, 2, 2, 2],  # ResNet-18
    base_width=64,
    negative_slope=-1.0,
    batch_norm=True,
    dropout_rate=0.1,
    spectral_norm=0.95,
    stem_kernel_size=7,
    stem_kernel_stride=2,
    stem_kernel_padding=3,
    stem_pool=True,
    activate_output=True,
)
```

### 构建轻量级 ResNet
```python
resnet = convolution.ResNet(
    dim_input=[3, 32, 32],
    layers=[2, 2, 2],
    base_width=32,  # 较小的基础宽度
    negative_slope=-1.0,
    batch_norm=True,
    dropout_rate=0.1,
    spectral_norm=0.95,
    stem_kernel_size=3,
    stem_kernel_stride=1,
    stem_kernel_padding=1,
    stem_pool=False,  # 小图像不使用池化
    activate_output=True,
)
```

## 注意事项

1. **输入格式**：输入必须是 4D 张量 `(batch, C, H, W)`
2. **维度匹配**：确保输入维度与 `dim_input` 匹配
3. **内存使用**：深层网络和大量通道会占用大量内存
4. **谱归一化**：会增加计算开销，但能提高稳定性
5. **批归一化**：训练和评估模式的行为不同

## 性能优化

1. **使用较小的 base_width**：减少内存和计算
2. **减少层数**：对于小数据集，较少的层可能足够
3. **禁用不必要的组件**：如不需要 Dropout 可以设置为 0
4. **使用混合精度**：可以加速训练

## 相关文档

- [模块概览](overview.md)
- [谱归一化模块](spectral_norm.md)
- [TARNet 模块](tarnet.md)
- [ResNet 论文](https://arxiv.org/abs/1512.03385)

