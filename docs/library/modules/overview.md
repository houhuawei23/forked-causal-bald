# 模块概览

## 概述

`causal_bald.library.modules` 模块提供了用于构建神经网络的各种组件和模块，包括全连接层、卷积层、高斯过程、变分推断等。

## 模块列表

### 1. 全连接层模块 (`dense.py`)

提供全连接神经网络的基础组件。

**主要类：**
- `Activation`: 激活函数模块
- `PreactivationDense`: 预激活全连接层
- `ResidualDense`: 残差全连接层
- `NeuralNetwork`: 完整的神经网络模块

**相关文档：** [全连接层模块](dense.md)

### 2. 卷积模块 (`convolution.py`)

提供卷积神经网络的基础组件。

**主要类：**
- `Activation`: 2D 激活函数模块
- `PreactivationConv`: 预激活卷积层
- `ResidualConv`: 残差卷积层
- `ResNet`: ResNet 架构

**相关文档：** [卷积模块](convolution.md)

### 3. 高斯过程模块 (`gaussian_process.py`)

提供变分高斯过程的实现。

**主要类：**
- `VariationalGP`: 变分高斯过程
- `DeepKernelGP`: 深度核高斯过程

**主要函数：**
- `initial_values_for_GP`: 初始化 GP 参数

**相关文档：** [高斯过程模块](gaussian_process.md)

### 4. 变分推断模块 (`variational.py`)

提供各种变分分布和输出层。

**主要类：**
- `Normal`: 正态分布输出层
- `SplitNormal`: 分离正态分布输出层
- `GMM`: 高斯混合模型输出层
- `SplitGMM`: 分离高斯混合模型输出层
- `Categorical`: 分类输出层

**相关文档：** [变分推断模块](variational.md)

### 5. 谱归一化模块 (`spectral_norm.py`)

提供谱归一化功能，用于稳定训练。

**主要类：**
- `SpectralNormFC`: 全连接层谱归一化
- `SpectralNormConv`: 卷积层谱归一化

**主要函数：**
- `spectral_norm_fc`: 应用全连接层谱归一化
- `spectral_norm_conv`: 应用卷积层谱归一化

**相关文档：** [谱归一化模块](spectral_norm.md)

### 6. TARNet 模块 (`tarnet.py`)

提供 TARNet 架构的实现。

**主要类：**
- `TARNet`: Treatment-Agnostic Representation Network

**相关文档：** [TARNet 模块](tarnet.md)

## 模块关系

```
NeuralNetwork (dense.py)
    ↓
TARNet (tarnet.py)
    ├── encoder (dense.NeuralNetwork 或 convolution.ResNet)
    ├── t0_encoder (dense.ResidualDense)
    ├── t1_encoder (dense.ResidualDense)
    └── outcome_density (variational.SplitGMM 或 variational.GMM)

DeepKernelGP (gaussian_process.py)
    ├── encoder (dense.NeuralNetwork 或 convolution.ResNet)
    └── gp (gaussian_process.VariationalGP)

所有模块
    ↓
spectral_norm (spectral_norm.py) - 可选
```

## 设计模式

### 1. 预激活设计

大多数模块采用预激活（Preactivation）设计：
```
BatchNorm → Activation → Dropout → Linear/Conv
```

### 2. 残差连接

支持残差连接，用于构建深层网络：
```
output = activation(linear(x)) + shortcut(x)
```

### 3. 谱归一化

可选地应用谱归一化，控制权重矩阵的谱范数：
```
weight = weight / max(1, spectral_norm / coeff)
```

### 4. 变分输出

使用变分分布作为输出层，提供不确定性估计：
```
output = VariationalDistribution(input)
```

## 使用示例

### 构建神经网络
```python
from causal_bald.library.modules import dense

network = dense.NeuralNetwork(
    architecture="resnet",
    dim_input=25,
    dim_hidden=200,
    depth=3,
    negative_slope=-1.0,  # 使用 ELU
    batch_norm=True,
    dropout_rate=0.1,
    spectral_norm=0.95,
    activate_output=True,
)
```

### 构建 TARNet
```python
from causal_bald.library.modules import tarnet

tarnet_model = tarnet.TARNet(
    architecture="resnet",
    dim_input=25,
    dim_hidden=200,
    dim_output=1,
    depth=3,
    negative_slope=-1.0,
    batch_norm=True,
    dropout_rate=0.1,
    spectral_norm=0.95,
)
```

### 构建 Deep Kernel GP
```python
from causal_bald.library.modules import dense, gaussian_process

encoder = dense.NeuralNetwork(...)
gp = gaussian_process.VariationalGP(...)
model = gaussian_process.DeepKernelGP(encoder=encoder, gp=gp)
```

## 架构选择

### 全连接架构 (`architecture="basic"`)
- 简单的预激活全连接层
- 适用于低维输入
- 计算效率高

### ResNet 架构 (`architecture="resnet"`)
- 带残差连接的全连接或卷积层
- 适用于深层网络
- 更好的梯度流动

## 相关文档

- [全连接层模块](dense.md)
- [卷积模块](convolution.md)
- [高斯过程模块](gaussian_process.md)
- [变分推断模块](variational.md)
- [谱归一化模块](spectral_norm.md)
- [TARNet 模块](tarnet.md)
- [库层概览](../overview.md)

