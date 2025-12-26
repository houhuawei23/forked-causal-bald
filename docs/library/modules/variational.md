# 变分推断模块文档

## 概述

`causal_bald.library.modules.variational` 模块提供了各种变分分布和输出层，用于在神经网络中建模不确定性。

## 主要类

### `Normal`

正态分布输出层，输出均值和标准差。

**初始化参数：**
- `dim_input` (int): 输入维度
- `dim_output` (int): 输出维度

**结构：**
```
Input
    ↓
Linear (μ) → μ
Linear (σ) → Softplus → σ
    ↓
Normal(μ, σ)
```

**使用示例：**
```python
from causal_bald.library.modules import variational

normal_layer = variational.Normal(
    dim_input=100,
    dim_output=1,
)

# 前向传播
output = normal_layer(inputs)  # Normal distribution
mean = output.mean  # 均值
std = output.stddev  # 标准差
```

### `SplitNormal`

分离正态分布输出层，为处理组和对照组分别建模。

**初始化参数：**
- `dim_input` (int): 输入维度
- `dim_output` (int): 输出维度

**结构：**
```
Input: [x, t]
    ↓
Linear (μ₀) → μ₀ (对照组)
Linear (σ₀) → Softplus → σ₀
Linear (μ₁) → μ₁ (处理组)
Linear (σ₁) → Softplus → σ₁
    ↓
Normal((1-t)μ₀ + tμ₁, (1-t)σ₀ + tσ₁)
```

**特点：**
- 根据处理变量 `t` 选择对应的均值和标准差
- 适用于因果推断场景

**使用示例：**
```python
split_normal = variational.SplitNormal(
    dim_input=100,
    dim_output=1,
)

# 输入包含处理变量
inputs = torch.cat([features, treatment], dim=-1)
output = split_normal(inputs)  # Normal distribution
```

### `GMM`

高斯混合模型输出层。

**初始化参数：**
- `dim_input` (int): 输入维度
- `dim_output` (int): 输出维度（混合成分数）

**结构：**
```
Input
    ↓
Linear (μ) → μ
Linear (σ) → Softplus → σ
Linear (π) → π (混合权重)
    ↓
MixtureSameFamily(
    Categorical(π),
    Normal(μ, σ)
)
```

**使用示例：**
```python
gmm = variational.GMM(
    dim_input=100,
    dim_output=3,  # 3 个混合成分
)

output = gmm(inputs)  # MixtureSameFamily distribution
```

### `SplitGMM`

分离高斯混合模型输出层，为处理组和对照组分别建模。

**初始化参数：**
- `dim_input` (int): 输入维度
- `dim_output` (int): 输出维度（混合成分数）

**结构：**
```
Input: [x, t]
    ↓
对照组: μ₀, σ₀, π₀
处理组: μ₁, σ₁, π₁
    ↓
MixtureSameFamily(
    Categorical((1-t)π₀ + tπ₁),
    Normal((1-t)μ₀ + tμ₁, (1-t)σ₀ + tσ₁)
)
```

**使用示例：**
```python
split_gmm = variational.SplitGMM(
    dim_input=100,
    dim_output=2,  # 2 个混合成分
)

inputs = torch.cat([features, treatment], dim=-1)
output = split_gmm(inputs)  # MixtureSameFamily distribution
```

### `Categorical`

分类输出层，支持二分类和多分类。

**初始化参数：**
- `dim_input` (int): 输入维度
- `dim_output` (int): 输出维度
  - `dim_output=1`: 二分类（Bernoulli）
  - `dim_output>1`: 多分类（Categorical）

**结构：**
```
Input
    ↓
Linear → logits
    ↓
Bernoulli(logits) 或 Categorical(logits)
```

**使用示例：**
```python
# 二分类
binary_classifier = variational.Categorical(
    dim_input=100,
    dim_output=1,
)

# 多分类
multi_classifier = variational.Categorical(
    dim_input=100,
    dim_output=10,
)

output = binary_classifier(inputs)  # Bernoulli distribution
```

### `MixtureSameFamily`

自定义混合分布类，重写了 `log_prob` 方法以提高数值稳定性。

**特点：**
- 继承自 `torch.distributions.MixtureSameFamily`
- 改进的 `log_prob` 计算
- 更好的数值稳定性

## 分布类型对比

### Normal vs SplitNormal

| 特性 | Normal | SplitNormal |
|------|--------|-------------|
| 输入格式 | `[x]` | `[x, t]` |
| 输出 | 单一分布 | 根据 t 选择分布 |
| 适用场景 | 一般回归 | 因果推断 |

### GMM vs SplitGMM

| 特性 | GMM | SplitGMM |
|------|-----|----------|
| 输入格式 | `[x]` | `[x, t]` |
| 输出 | 单一混合模型 | 根据 t 选择混合模型 |
| 适用场景 | 多模态回归 | 因果推断（多模态） |

## 使用场景

### 1. 简单回归（Normal）
```python
network = nn.Sequential(
    encoder,
    variational.Normal(dim_input=hidden_dim, dim_output=1),
)
```

### 2. 因果推断（SplitNormal）
```python
network = nn.Sequential(
    encoder,
    variational.SplitNormal(dim_input=hidden_dim, dim_output=1),
)
```

### 3. 多模态回归（GMM）
```python
network = nn.Sequential(
    encoder,
    variational.GMM(dim_input=hidden_dim, dim_output=3),  # 3 个成分
)
```

### 4. 倾向性评分（Categorical）
```python
network = nn.Sequential(
    encoder,
    variational.Categorical(dim_input=hidden_dim, dim_output=1),  # 二分类
)
```

## 数值稳定性

所有分布都包含数值稳定性措施：

1. **Softplus**：确保标准差为正
2. **Epsilon**：添加小的 epsilon（1e-7）避免数值问题
3. **Log Prob**：`MixtureSameFamily` 使用改进的 log_prob 计算

## 完整示例

### TARNet 中的使用
```python
from causal_bald.library.modules import variational

# 高维输入：使用 SplitGMM
if dim_input > 1:
    outcome_density = variational.SplitGMM(
        dim_input=hidden_dim,
        dim_output=1,
    )
# 1维输入：使用 GMM
else:
    outcome_density = nn.Sequential(
        encoder,
        variational.GMM(
            dim_input=hidden_dim,
            dim_output=1,
        ),
    )
```

### 预测和采样
```python
# 预测均值
mean = output.mean

# 采样
samples = output.sample(torch.Size([100]))

# 计算对数概率
log_prob = output.log_prob(targets)

# 计算熵（不确定性）
entropy = output.entropy()
```

## 注意事项

1. **输入格式**：`SplitNormal` 和 `SplitGMM` 需要输入包含处理变量
2. **维度匹配**：确保输入维度与 `dim_input` 匹配
3. **数值稳定性**：注意处理极端值
4. **混合成分数**：GMM 的 `dim_output` 表示混合成分数，不是输出维度

## 相关文档

- [模块概览](overview.md)
- [TARNet 模块](tarnet.md)
- [PyTorch Distributions](https://pytorch.org/docs/stable/distributions.html)

