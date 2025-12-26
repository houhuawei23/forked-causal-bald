# 全连接层模块文档

## 概述

`causal_bald.library.modules.dense` 模块提供了用于构建全连接神经网络的各种组件，包括激活函数、预激活层、残差层和完整的神经网络模块。

## 主要类

### `Activation`

激活函数模块，包含批归一化、激活函数和 Dropout。

**初始化参数：**
- `dim_input` (int): 输入维度
- `negative_slope` (float): LeakyReLU 的负斜率（< 0 时使用 ELU）
- `dropout_rate` (float): Dropout 率
- `batch_norm` (bool): 是否使用批归一化

**结构：**
```
BatchNorm1d (可选) → LeakyReLU/ELU → Dropout
```

**使用示例：**
```python
from causal_bald.library.modules import dense

activation = dense.Activation(
    dim_input=100,
    negative_slope=-1.0,  # 使用 ELU
    dropout_rate=0.1,
    batch_norm=True,
)
```

### `PreactivationDense`

预激活全连接层，采用 Preactivation 设计。

**初始化参数：**
- `dim_input` (int): 输入维度
- `dim_output` (int): 输出维度
- `bias` (bool): 是否使用偏置
- `negative_slope` (float): LeakyReLU 负斜率
- `dropout_rate` (float): Dropout 率
- `batch_norm` (bool): 是否使用批归一化
- `spectral_norm` (float): 谱归一化系数（> 0 时应用）

**结构：**
```
Activation → Linear (可选谱归一化)
```

**使用示例：**
```python
layer = dense.PreactivationDense(
    dim_input=100,
    dim_output=200,
    bias=True,
    negative_slope=-1.0,
    dropout_rate=0.1,
    batch_norm=True,
    spectral_norm=0.95,
)
```

### `ResidualDense`

残差全连接层，支持残差连接。

**初始化参数：** 同 `PreactivationDense`

**结构：**
```
output = PreactivationDense(x) + shortcut(x)
```

**Shortcut 连接：**
- 如果 `dim_input == dim_output`：使用恒等映射
- 否则：使用 `Dropout → Linear` 投影

**使用示例：**
```python
layer = dense.ResidualDense(
    dim_input=100,
    dim_output=100,  # 相同维度使用恒等映射
    bias=True,
    negative_slope=-1.0,
    dropout_rate=0.1,
    batch_norm=True,
    spectral_norm=0.95,
)
```

### `NeuralNetwork`

完整的全连接神经网络模块。

**初始化参数：**
- `architecture` (str): 架构类型（`"basic"` 或 `"resnet"`）
- `dim_input` (int): 输入维度
- `dim_hidden` (int): 隐藏层维度
- `depth` (int): 网络深度
- `negative_slope` (float): LeakyReLU 负斜率
- `batch_norm` (bool): 是否使用批归一化
- `dropout_rate` (float): Dropout 率
- `spectral_norm` (float): 谱归一化系数
- `activate_output` (bool): 是否激活输出层

**结构：**
```
Input Layer (Linear + 谱归一化)
    ↓
Hidden Layers (depth - 1 层)
    ├── basic: PreactivationDense
    └── resnet: ResidualDense
    ↓
Output Activation (可选)
```

**属性：**
- `dim_output`: 输出维度（等于 `dim_hidden`）

**使用示例：**
```python
network = dense.NeuralNetwork(
    architecture="resnet",
    dim_input=25,
    dim_hidden=200,
    depth=3,
    negative_slope=-1.0,
    batch_norm=True,
    dropout_rate=0.1,
    spectral_norm=0.95,
    activate_output=True,
)

# 前向传播
output = network(inputs)  # shape: (batch_size, dim_hidden)
```

## 模块字典

```python
MODULES = {
    "basic": PreactivationDense,
    "resnet": ResidualDense,
}
```

## 架构对比

### Basic 架构
- 简单的预激活全连接层
- 无残差连接
- 计算效率高
- 适用于浅层网络

### ResNet 架构
- 带残差连接的全连接层
- 更好的梯度流动
- 适用于深层网络
- 更稳定的训练

## 激活函数选择

### LeakyReLU
当 `negative_slope >= 0.0` 时使用：
```python
LeakyReLU(negative_slope=negative_slope)
```

### ELU
当 `negative_slope < 0.0` 时使用：
```python
ELU()
```

**推荐值：** `negative_slope=-1.0`（使用 ELU）

## 谱归一化

当 `spectral_norm > 0.0` 时，对线性层应用谱归一化：

```python
if spectral_norm > 0.0:
    linear = spectral_norm_fc(linear, spectral_norm)
```

**作用：**
- 控制权重矩阵的谱范数
- 稳定训练过程
- 提高泛化能力

**推荐值：** `spectral_norm=0.95`

## 完整示例

### 构建简单网络
```python
from causal_bald.library.modules import dense

# Basic 架构
network = dense.NeuralNetwork(
    architecture="basic",
    dim_input=25,
    dim_hidden=100,
    depth=2,
    negative_slope=-1.0,
    batch_norm=False,
    dropout_rate=0.0,
    spectral_norm=0.0,
    activate_output=True,
)
```

### 构建深层 ResNet
```python
# ResNet 架构
network = dense.NeuralNetwork(
    architecture="resnet",
    dim_input=25,
    dim_hidden=200,
    depth=5,
    negative_slope=-1.0,
    batch_norm=True,
    dropout_rate=0.1,
    spectral_norm=0.95,
    activate_output=True,
)
```

### 自定义层
```python
# 构建自定义层序列
layers = []
layers.append(dense.PreactivationDense(25, 100, ...))
layers.append(dense.ResidualDense(100, 100, ...))
layers.append(dense.ResidualDense(100, 200, ...))
layers.append(dense.Activation(200, ...))

network = nn.Sequential(*layers)
```

## 注意事项

1. **批归一化与偏置**：使用批归一化时，通常设置 `bias=False`
2. **Dropout 位置**：Dropout 在激活函数之后
3. **残差连接**：确保输入和输出维度匹配（或提供投影）
4. **谱归一化**：会增加少量计算开销，但能提高稳定性

## 相关文档

- [模块概览](overview.md)
- [谱归一化模块](spectral_norm.md)
- [TARNet 模块](tarnet.md)

