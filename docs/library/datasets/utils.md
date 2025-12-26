# 数据集工具函数文档

## 概述

`causal_bald.library.datasets.utils` 模块提供了用于数据集生成和处理的工具函数，主要用于合成数据集的生成。

## 函数列表

### `nominal_propensity(x, beta=0.75)`

计算名义倾向性评分（未调整的倾向性评分）。

**参数：**
- `x` (np.ndarray): 协变量
- `beta` (float, 默认=0.75): x 对倾向性评分的影响系数

**返回：** np.ndarray - 名义倾向性评分，范围 [0, 1]

**数学公式：**
```
logit = beta * x + 0.5
pi_nominal = (1 + exp(-logit))^(-1)
```

**使用示例：**
```python
from causal_bald.library.datasets import utils
import numpy as np

x = np.array([[0.5], [1.0], [-1.0]])
pi_nominal = utils.nominal_propensity(x, beta=0.75)
# 输出: array([0.73, 0.82, 0.38])
```

### `complete_propensity(x, u, lambda_, beta=0.75)`

计算完整倾向性评分（调整后的倾向性评分）。

**参数：**
- `x` (np.ndarray): 协变量
- `u` (float/np.ndarray): 未观测混淆变量（通常为 0）
- `lambda_` (float): lambda 参数
- `beta` (float, 默认=0.75): x 对倾向性评分的影响系数

**返回：** np.ndarray - 完整倾向性评分

**数学公式：**
```
pi_nominal = nominal_propensity(x, beta)
alpha = alpha_fn(pi_nominal, lambda_)
beta_val = beta_fn(pi_nominal, lambda_)
pi_complete = u / alpha + (1 - u) / beta_val
```

**使用示例：**
```python
x = np.array([[0.5], [1.0], [-1.0]])
pi = utils.complete_propensity(x, u=0.0, lambda_=1.0, beta=0.75)
```

### `f_mu(x, t, u, gamma=4.0)`

计算潜在结果函数。

**参数：**
- `x` (np.ndarray): 协变量
- `t` (float): 处理变量（0 或 1）
- `u` (float): 未观测混淆变量（通常为 0）
- `gamma` (float, 默认=4.0): gamma 参数

**返回：** np.ndarray - 潜在结果

**数学公式：**
```
mu = (2*t - 1) * x
    + (2.0*t - 1)
    - 2 * sin((4*t - 2) * x)
    - (gamma * u - 2) * (1 + 0.5 * x)
```

**使用示例：**
```python
x = np.array([[0.5], [1.0], [-1.0]])

# 对照组潜在结果
mu0 = utils.f_mu(x, t=0.0, u=0.0, gamma=4.0)

# 处理组潜在结果
mu1 = utils.f_mu(x, t=1.0, u=0.0, gamma=4.0)

# 治疗效应
tau = mu1 - mu0
```

### `linear_normalization(x, new_min, new_max)`

线性归一化函数，将数据缩放到指定范围。

**参数：**
- `x` (np.ndarray): 输入数据
- `new_min` (float): 新范围的最小值
- `new_max` (float): 新范围的最大值

**返回：** np.ndarray - 归一化后的数据

**数学公式：**
```
x_normalized = (x - x.min()) * (new_max - new_min) / (x.max() - x.min()) + new_min
```

**使用示例：**
```python
x = np.array([1, 2, 3, 4, 5])
x_norm = utils.linear_normalization(x, new_min=0.0, new_max=1.0)
# 输出: array([0.0, 0.25, 0.5, 0.75, 1.0])
```

## 函数关系

### 倾向性评分计算流程

```
x (协变量)
    ↓
nominal_propensity(x, beta)
    ↓
pi_nominal (名义倾向性评分)
    ↓
alpha_fn(pi_nominal, lambda_)
beta_fn(pi_nominal, lambda_)
    ↓
complete_propensity(x, u, lambda_, beta)
    ↓
pi (完整倾向性评分)
```

### 潜在结果计算流程

```
x (协变量), t (处理变量)
    ↓
f_mu(x, t, u, gamma)
    ↓
mu (潜在结果)
    ↓
mu0 = f_mu(x, t=0, u, gamma)
mu1 = f_mu(x, t=1, u, gamma)
    ↓
tau = mu1 - mu0 (治疗效应)
```

## 参数说明

### `beta`（x 对倾向性评分的影响系数）

- **默认值**：0.75
- **作用**：控制协变量对处理分配的影响强度
- **范围**：通常 0.5 - 2.0
- **影响**：较大的 beta 导致更强的选择偏差

### `lambda_`（lambda 参数）

- **作用**：调整倾向性评分的分布
- **默认值**：1.0
- **影响**：影响倾向性评分的调整程度

### `gamma`（gamma 参数）

- **默认值**：4.0
- **作用**：控制未观测混淆变量的影响
- **影响**：影响潜在结果的生成

### `u`（未观测混淆变量）

- **默认值**：0.0
- **作用**：模拟未观测混淆
- **注意**：在标准设置中通常为 0

## 使用示例

### 生成合成数据
```python
import numpy as np
from causal_bald.library.datasets import utils

# 生成协变量
n_samples = 1000
x = np.random.randn(n_samples, 1).astype("float32")

# 计算倾向性评分
pi = utils.complete_propensity(x, u=0.0, lambda_=1.0, beta=0.75)

# 生成处理变量
t = np.random.binomial(1, pi).astype("float32")

# 计算潜在结果
mu0 = utils.f_mu(x, t=0.0, u=0.0, gamma=4.0)
mu1 = utils.f_mu(x, t=1.0, u=0.0, gamma=4.0)

# 生成观测结果
sigma_y = 1.0
eps = sigma_y * np.random.randn(n_samples).astype("float32")
y0 = mu0 + eps
y1 = mu1 + eps
y = t * y1 + (1 - t) * y0

# 计算治疗效应
tau = mu1 - mu0
```

### 可视化函数
```python
import matplotlib.pyplot as plt

# 生成测试点
x_test = np.linspace(-3, 3, 100).reshape(-1, 1)

# 计算倾向性评分
pi_test = utils.complete_propensity(x_test, u=0.0, lambda_=1.0, beta=0.75)

# 计算潜在结果
mu0_test = utils.f_mu(x_test, t=0.0, u=0.0, gamma=4.0)
mu1_test = utils.f_mu(x_test, t=1.0, u=0.0, gamma=4.0)
tau_test = mu1_test - mu0_test

# 绘制
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x_test, pi_test)
plt.xlabel('x')
plt.ylabel('π(x)')
plt.title('Propensity Score')

plt.subplot(1, 3, 2)
plt.plot(x_test, mu0_test, label='μ₀')
plt.plot(x_test, mu1_test, label='μ₁')
plt.xlabel('x')
plt.ylabel('Potential Outcome')
plt.legend()
plt.title('Potential Outcomes')

plt.subplot(1, 3, 3)
plt.plot(x_test, tau_test)
plt.xlabel('x')
plt.ylabel('τ(x)')
plt.title('Treatment Effect')

plt.tight_layout()
plt.show()
```

## 数学背景

### 倾向性评分

倾向性评分表示给定协变量下接受处理的概率：
```
π(x) = P(T=1|X=x)
```

### 潜在结果框架

在潜在结果框架中：
- `Y(0)`: 对照组的潜在结果
- `Y(1)`: 处理组的潜在结果
- `τ = Y(1) - Y(0)`: 个体治疗效应

### 治疗效应

平均治疗效应（ATE）：
```
ATE = E[Y(1) - Y(0)] = E[τ]
```

条件平均治疗效应（CATE）：
```
CATE(x) = E[Y(1) - Y(0)|X=x] = E[τ|X=x]
```

## 注意事项

1. **数据类型**：函数返回 numpy 数组，注意类型转换
2. **维度匹配**：确保输入维度匹配
3. **数值稳定性**：注意处理极端值
4. **随机性**：使用固定随机种子确保可重复性

## 相关文档

- [数据集概览](overview.md)
- [Synthetic 数据集](synthetic.md)
- [工具函数文档](../utils.md)

