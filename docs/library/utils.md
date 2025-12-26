# 工具函数文档

## 概述

`causal_bald.library.utils` 模块提供了用于倾向性评分计算和数据处理的各种工具函数。

## 函数列表

### `lambda_top_func(mu, k, y, alpha)`

计算上界 lambda 函数值。

**参数：**
- `mu` (torch.Tensor): 均值参数
- `k` (int): 索引参数
- `y` (torch.Tensor): 数据张量，shape `(m, ...)`
- `alpha` (float): alpha 参数

**返回：** torch.Tensor - 计算得到的 lambda 值

**数学公式：**
```
m = y.shape[0]
r = sum(y[k:] - mu)
py = (k + 1) / m
lambda = mu + r / (m * (alpha + 1) - py)
```

**用途：** 用于倾向性评分的计算，特别是在处理上界情况时。

### `lambda_bottom_func(mu, k, y, alpha)`

计算下界 lambda 函数值。

**参数：**
- `mu` (torch.Tensor): 均值参数
- `k` (int): 索引参数
- `y` (torch.Tensor): 数据张量，shape `(m, ...)`
- `alpha` (float): alpha 参数

**返回：** torch.Tensor - 计算得到的 lambda 值

**数学公式：**
```
m = y.shape[0]
r = sum(y[:k+1] - mu)
py = k + 1
lambda = mu + r / (m * alpha + py)
```

**用途：** 用于倾向性评分的计算，特别是在处理下界情况时。

### `alpha_fn(pi, lambda_)`

计算 alpha 函数值。

**参数：**
- `pi` (float/torch.Tensor): 倾向性评分
- `lambda_` (float/torch.Tensor): lambda 参数

**返回：** float/torch.Tensor - alpha 值

**数学公式：**
```
alpha = (pi * lambda_) ** -1 + 1.0 - lambda_ ** -1
```

**用途：** 用于倾向性评分相关的计算，特别是在合成数据生成中。

### `beta_fn(pi, lambda_)`

计算 beta 函数值。

**参数：**
- `pi` (float/torch.Tensor): 倾向性评分
- `lambda_` (float/torch.Tensor): lambda 参数

**返回：** float/torch.Tensor - beta 值

**数学公式：**
```
beta = lambda_ * (pi) ** -1 + 1.0 - lambda_
```

**用途：** 用于倾向性评分相关的计算，特别是在合成数据生成中。

## 使用示例

### 基本使用
```python
import torch
from causal_bald.library import utils

# 示例数据
mu = torch.tensor(0.5)
k = 10
y = torch.randn(100, 1)
alpha = 0.75

# 计算 lambda
lambda_top = utils.lambda_top_func(mu, k, y, alpha)
lambda_bottom = utils.lambda_bottom_func(mu, k, y, alpha)

# 计算 alpha 和 beta
pi = 0.6
lambda_ = 1.5
alpha_val = utils.alpha_fn(pi, lambda_)
beta_val = utils.beta_fn(pi, lambda_)
```

### 在数据集生成中使用

这些函数主要用于合成数据集的生成，特别是在计算倾向性评分时：

```python
# 在 datasets/utils.py 中使用
from causal_bald.library import utils

def complete_propensity(x, u, lambda_, beta=0.75):
    nominal = nominal_propensity(x, beta=beta)
    alpha = utils.alpha_fn(nominal, lambda_)
    beta = utils.beta_fn(nominal, lambda_)
    return (u / alpha) + ((1 - u) / beta)
```

## 数学背景

这些函数与倾向性评分的计算相关，特别是在处理：

1. **上界和下界**：`lambda_top_func` 和 `lambda_bottom_func` 用于处理倾向性评分的边界情况
2. **参数转换**：`alpha_fn` 和 `beta_fn` 用于在不同参数空间之间转换

## 注意事项

1. **张量操作**：函数支持 PyTorch 张量，可以进行批量计算
2. **数值稳定性**：注意除零情况，确保分母不为零
3. **维度匹配**：确保输入张量的维度匹配

## 相关文档

- [数据集工具函数](../datasets/utils.md) - 数据集相关的工具函数
- [库层概览](overview.md)

