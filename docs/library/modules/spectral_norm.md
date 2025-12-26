# 谱归一化模块文档

## 概述

`causal_bald.library.modules.spectral_norm` 模块提供了谱归一化（Spectral Normalization）功能，用于稳定神经网络训练并提高泛化能力。

## 理论基础

谱归一化通过约束权重矩阵的谱范数（最大奇异值）来控制网络的 Lipschitz 常数，从而稳定训练过程。

**数学原理：**
- 权重矩阵的谱范数：`σ(W) = max_{||u||=1} ||Wu||`
- 归一化：`W_normalized = W / max(1, σ(W) / coeff)`
- 软归一化：只有当 `σ(W) > coeff` 时才归一化

## 主要类

### `SpectralNormFC`

全连接层的谱归一化实现。

**方法：**
- `compute_weight(module, do_power_iteration)`: 计算归一化后的权重
- `apply(module, coeff, name, n_power_iterations, dim, eps)`: 应用谱归一化

**特点：**
- 使用幂迭代（Power Iteration）估计谱范数
- 支持软归一化（soft normalization）
- 记录实际谱范数值

### `SpectralNormConv`

卷积层的谱归一化实现。

**方法：**
- `compute_weight(module, do_power_iteration)`: 计算归一化后的权重
- `apply(module, coeff, input_dim, name, n_power_iterations, eps)`: 应用谱归一化

**特点：**
- 针对卷积操作优化
- 考虑卷积的 stride 和 padding
- 支持转置卷积

## 主要函数

### `spectral_norm_fc(module, coeff, n_power_iterations, name, eps, dim)`

对全连接层应用谱归一化。

**参数：**
- `module` (nn.Module): 要归一化的模块（通常是 `nn.Linear`）
- `coeff` (float): 目标谱范数系数
- `n_power_iterations` (int, 默认=1): 幂迭代次数
- `name` (str, 默认="weight"): 权重参数名称
- `eps` (float, 默认=1e-12): 数值稳定性 epsilon
- `dim` (int, 可选): 输出维度（自动推断）

**返回：** 应用了谱归一化的模块

**使用示例：**
```python
from causal_bald.library.modules.spectral_norm import spectral_norm_fc
from torch import nn

# 创建线性层
linear = nn.Linear(100, 200)

# 应用谱归一化
linear_normalized = spectral_norm_fc(
    module=linear,
    coeff=0.95,  # 目标谱范数
    n_power_iterations=1,
)

# 使用
output = linear_normalized(inputs)
```

### `spectral_norm_conv(module, coeff, input_dim, n_power_iterations, name, eps)`

对卷积层应用谱归一化。

**参数：**
- `module` (nn.Module): 要归一化的模块（通常是 `nn.Conv2d`）
- `coeff` (float): 目标谱范数系数
- `input_dim` (tuple): 输入维度 `(C, H, W)`
- `n_power_iterations` (int, 默认=1): 幂迭代次数
- `name` (str, 默认="weight"): 权重参数名称
- `eps` (float, 默认=1e-12): 数值稳定性 epsilon

**返回：** 应用了谱归一化的模块

**使用示例：**
```python
from causal_bald.library.modules.spectral_norm import spectral_norm_conv
from torch import nn

# 创建卷积层
conv = nn.Conv2d(3, 64, kernel_size=3)

# 应用谱归一化
conv_normalized = spectral_norm_conv(
    module=conv,
    coeff=0.95,
    input_dim=(3, 32, 32),  # 输入图像维度
)
```

## 幂迭代算法

谱归一化使用幂迭代算法估计权重矩阵的最大奇异值：

```python
for _ in range(n_power_iterations):
    v = normalize(W^T @ u)
    u = normalize(W @ v)
sigma = u^T @ W @ v
```

**参数选择：**
- `n_power_iterations=1`: 快速但可能不够准确
- `n_power_iterations>1`: 更准确但计算更慢

## 软归一化

模块实现软归一化（Soft Normalization），只有当谱范数超过阈值时才归一化：

```python
factor = max(1, sigma / coeff)
weight_normalized = weight / factor
```

**优势：**
- 避免过度约束
- 保持网络的表达能力
- 更稳定的训练

## 使用场景

### 1. 全连接层
```python
from causal_bald.library.modules import dense
from causal_bald.library.modules.spectral_norm import spectral_norm_fc

linear = nn.Linear(100, 200)
if spectral_norm > 0.0:
    linear = spectral_norm_fc(linear, spectral_norm)
```

### 2. 卷积层
```python
from causal_bald.library.modules.spectral_norm import spectral_norm_conv

conv = nn.Conv2d(3, 64, 3)
if spectral_norm > 0.0:
    conv = spectral_norm_conv(conv, spectral_norm, input_dim=(3, 32, 32))
```

### 3. 在模块中使用
```python
from causal_bald.library.modules import dense

# 自动应用谱归一化
network = dense.NeuralNetwork(
    dim_input=25,
    dim_hidden=200,
    spectral_norm=0.95,  # 自动应用到所有线性层
    # ...
)
```

## 参数选择

### coeff（目标谱范数）

- **0.95**: 推荐值，平衡稳定性和表达能力
- **1.0**: 标准归一化
- **> 1.0**: 允许更大的谱范数
- **0.0**: 禁用谱归一化

### n_power_iterations（幂迭代次数）

- **1**: 默认值，快速估计
- **2-3**: 更准确的估计
- **> 3**: 通常不需要

## 监控谱范数

应用谱归一化后，可以访问实际的谱范数值：

```python
linear = spectral_norm_fc(nn.Linear(100, 200), coeff=0.95)
# 访问谱范数
sigma = linear.weight_u  # u 向量
sigma = linear.weight_v  # v 向量
sigma = linear.weight_sigma  # 实际谱范数
```

## 性能影响

### 计算开销
- **全连接层**：O(dim_output) 额外计算
- **卷积层**：O(output_size) 额外计算
- **总体**：通常 < 5% 的训练时间增加

### 内存开销
- 存储 `u` 和 `v` 向量
- 通常可以忽略不计

## 完整示例

### 在神经网络中应用
```python
from causal_bald.library.modules import dense
from causal_bald.library.modules.spectral_norm import spectral_norm_fc

# 方法1：在模块中自动应用
network = dense.NeuralNetwork(
    architecture="resnet",
    dim_input=25,
    dim_hidden=200,
    depth=3,
    spectral_norm=0.95,  # 自动应用
    # ...
)

# 方法2：手动应用
linear = nn.Linear(100, 200)
linear = spectral_norm_fc(linear, coeff=0.95)
```

### 监控训练过程
```python
# 训练后检查谱范数
for name, module in network.named_modules():
    if hasattr(module, 'weight_sigma'):
        sigma = module.weight_sigma.item()
        print(f"{name}: spectral norm = {sigma:.4f}")
```

## 注意事项

1. **设备一致性**：确保模块在正确的设备上
2. **状态保存**：谱归一化的状态会自动保存和加载
3. **训练模式**：只在训练时进行幂迭代
4. **数值稳定性**：使用 epsilon 避免除零

## 相关文档

- [模块概览](overview.md)
- [全连接层模块](dense.md)
- [卷积模块](convolution.md)
- [论文：Spectral Normalization](https://arxiv.org/abs/1802.05957)

