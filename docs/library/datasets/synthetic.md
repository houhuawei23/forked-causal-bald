# Synthetic 数据集文档

## 概述

`causal_bald.library.datasets.synthetic` 模块实现了合成数据集的生成器，用于方法验证和调试。该数据集具有已知的真实治疗效应，便于评估方法的准确性。

## 数据集信息

- **协变量维度**：1 维
- **样本数**：可配置（默认 10000）
- **特点**：已知真实治疗效应，支持双峰分布

## 类定义

### `Synthetic`

合成数据集类，继承自 `torch.utils.data.Dataset`。

**初始化参数：**
- `num_examples` (int): 样本数量
- `mode` (str): 模式（`"mu"` 用于结果预测，`"pi"` 用于倾向性评分）
- `beta` (float, 默认=0.75): x 对 t 的影响系数
- `sigma_y` (float, 默认=1.0): 结果噪声标准差
- `bimodal` (bool, 默认=False): 是否使用双峰分布
- `seed` (int, 默认=1331): 随机种子
- `split` (str, 可选): 数据集划分（未使用）

**属性：**
- `x` (np.ndarray): 协变量，shape `(n_samples, 1)`
- `t` (np.ndarray): 处理指示变量，shape `(n_samples,)`
- `y` (np.ndarray): 观测结果，shape `(n_samples,)`
- `mu0` (np.ndarray): 对照组潜在结果
- `mu1` (np.ndarray): 处理组潜在结果
- `tau` (np.ndarray): 真实治疗效应（μ₁ - μ₀）
- `pi` (np.ndarray): 倾向性评分
- `dim_input` (int): 输入维度（1）
- `dim_treatment` (int): 处理维度（1）
- `dim_output` (int): 输出维度（1）

## 数据生成过程

### 1. 生成协变量

#### 单峰分布（默认）
```python
x ~ N(0, 1)  # 标准正态分布
```

#### 双峰分布
```python
x ~ 0.5 * N(-2, 0.7²) + 0.5 * N(2, 0.7²)  # 混合正态分布
```

### 2. 生成倾向性评分

```python
pi = complete_propensity(x, u=0, lambda_=1.0, beta=beta)
```

### 3. 生成处理变量

```python
t ~ Bernoulli(pi)
```

### 4. 生成潜在结果

```python
mu0 = f_mu(x, t=0.0, u=0, gamma=0.0)
mu1 = f_mu(x, t=1.0, u=0, gamma=0.0)
```

### 5. 生成观测结果

```python
eps ~ N(0, sigma_y²)
y0 = mu0 + eps
y1 = mu1 + eps
y = t * y1 + (1 - t) * y0
```

### 6. 计算治疗效应

```python
tau = mu1 - mu0
```

## 使用示例

### 基本使用
```python
from causal_bald.library.datasets import Synthetic

# 生成数据集
ds = Synthetic(
    num_examples=10000,
    mode="mu",
    beta=0.75,
    sigma_y=1.0,
    bimodal=False,
    seed=1331,
)
```

### 双峰分布
```python
ds_bimodal = Synthetic(
    num_examples=10000,
    mode="mu",
    beta=2.0,
    sigma_y=1.0,
    bimodal=True,  # 使用双峰分布
    seed=1331,
)
```

### 在数据加载器中使用
```python
from torch.utils.data import DataLoader

loader = DataLoader(
    ds,
    batch_size=100,
    shuffle=True,
)

for batch in loader:
    (x, t), y = batch
    # x: (batch_size, 1)
    # t: (batch_size,)
    # y: (batch_size,)
```

### 访问真实治疗效应
```python
# 真实治疗效应
tau_true = ds.tau  # shape: (n_samples,)

# 可视化
import matplotlib.pyplot as plt
plt.scatter(ds.x, ds.tau)
plt.xlabel('x')
plt.ylabel('τ(x)')
plt.show()
```

### 倾向性评分模式
```python
ds_pi = Synthetic(
    num_examples=10000,
    mode="pi",  # 倾向性评分模式
    beta=0.75,
    seed=1331,
)

# 输入是协变量，目标是处理变量
for batch in DataLoader(ds_pi):
    x, t = batch
    # x: (batch_size, 1)
    # t: (batch_size,)
```

## 参数说明

### `beta`（x 对 t 的影响系数）

- **默认值**：0.75
- **作用**：控制协变量对处理分配的影响强度
- **范围**：通常 0.5 - 2.0
- **影响**：较大的 beta 导致更强的选择偏差

### `sigma_y`（结果噪声标准差）

- **默认值**：1.0
- **作用**：控制结果的噪声水平
- **影响**：较大的 sigma_y 增加学习难度

### `bimodal`（双峰分布）

- **默认值**：False
- **作用**：是否使用双峰分布生成协变量
- **用途**：测试方法在非标准分布下的表现

## 真实治疗效应函数

数据集提供了真实治疗效应函数：

```python
def tau_fn(self, x):
    """计算真实治疗效应"""
    return utils.f_mu(x=x, t=1.0, u=1.0, gamma=0.0) - utils.f_mu(
        x=x, t=0.0, u=1.0, gamma=0.0
    )
```

**使用示例：**
```python
# 计算任意 x 的治疗效应
x_new = np.array([[0.5], [1.0], [-1.0]])
tau_new = ds.tau_fn(x_new)
```

## 可视化

### 可视化数据集
```python
import matplotlib.pyplot as plt
import numpy as np

# 绘制协变量分布
plt.hist(ds.x, bins=50, alpha=0.5, label='x')
plt.xlabel('x')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# 绘制治疗效应
x_sorted = np.sort(ds.x, axis=0)
tau_sorted = ds.tau[np.argsort(ds.x, axis=0)]
plt.plot(x_sorted, tau_sorted)
plt.xlabel('x')
plt.ylabel('τ(x)')
plt.title('True Treatment Effect')
plt.show()
```

## 注意事项

1. **随机种子**：影响所有随机生成的数据
2. **模式选择**：`mode="mu"` 用于结果预测，`mode="pi"` 用于倾向性评分
3. **数据格式**：所有数据都是 float32 类型
4. **维度**：协变量是 1 维的，便于可视化

## 相关文档

- [数据集概览](overview.md)
- [IHDP 数据集](ihdp.md)
- [数据集工具函数](utils.md)

