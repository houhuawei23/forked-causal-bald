# 获取函数文档

## 概述

`causal_bald.library.acquisitions` 模块实现了多种基于信息论的获取函数，用于主动学习中的样本选择。这些函数专门针对因果推断场景设计，考虑了处理组和对照组的重叠区域。

## 函数签名

所有获取函数遵循统一的签名：

```python
def acquisition_function(mu_0, mu_1, t, pt, temperature):
    """
    参数:
        mu_0: np.ndarray, shape (num_samples, batch_size)
              对照组均值预测（来自多个模型样本）
        mu_1: np.ndarray, shape (num_samples, batch_size)
              处理组均值预测（来自多个模型样本）
        t: np.ndarray, shape (batch_size,)
           处理指示变量（0 或 1）
        pt: np.ndarray, shape (batch_size,) 或 None
           倾向性评分
        temperature: float
                    温度参数，用于缩放获取分数
    
    返回:
        scores: np.ndarray, shape (batch_size,)
                获取分数
    """
```

## 获取函数列表

### `random(mu_0, mu_1, t, pt, temperature)`

随机获取函数，返回零分数（用于随机采样）。

**数学公式：**
```
score = 0
```

**使用场景：** 基线对比、首次迭代

### `mu(mu_0, mu_1, t, pt, temperature)`

μ-BALD：基于均值不确定性的获取函数。

**数学公式：**
```
score = (1 / temperature) * log(Var[μ_t] + ε)
```

其中 `μ_t = t * μ₁ + (1-t) * μ₀`，`Var[μ_t]` 是模型样本间的方差。

**特点：**
- 关注当前处理组的不确定性
- 不考虑重叠区域
- 简单高效

### `rho(mu_0, mu_1, t, pt, temperature)`

ρ-BALD：关注重叠区域的获取函数。

**数学公式：**
```
score = τ-BALD - μ(1-t)-BALD
      = log(Var[τ] + ε) - log(Var[μ_{1-t}] + ε)
```

其中 `τ = μ₁ - μ₀` 是治疗效应。

**特点：**
- 关注重叠区域（两个处理组都存在的区域）
- 平衡处理组和对照组的不确定性
- 有助于识别因果效应

### `mu_rho(mu_0, mu_1, t, pt, temperature)`

μρ-BALD：结合均值不确定性和重叠区域的获取函数（推荐）。

**数学公式：**
```
score = μ-BALD + ρ-BALD
      = log(Var[μ_t] + ε) + (log(Var[τ] + ε) - log(Var[μ_{1-t}] + ε))
```

**特点：**
- 结合了 μ-BALD 和 ρ-BALD 的优势
- 既关注当前处理组的不确定性，也关注重叠区域
- 在实验中表现最好

### `pi(mu_0, mu_1, t, pt, temperature)`

π-BALD：基于倾向性评分的获取函数。

**数学公式：**
```
score = log(t * (1 - pt) + (1 - t) * pt + ε)
```

**特点：**
- 关注倾向性评分接近 0.5 的样本（重叠区域）
- 需要预先训练倾向性评分模型
- 不直接使用模型不确定性

### `mu_pi(mu_0, mu_1, t, pt, temperature)`

μπ-BALD：结合均值不确定性和倾向性评分。

**数学公式：**
```
score = μ-BALD + π-BALD
      = log(Var[μ_t] + ε) + log(t * (1 - pt) + (1 - t) * pt + ε)
```

**特点：**
- 结合模型不确定性和倾向性评分
- 需要倾向性评分模型
- 平衡信息获取和重叠区域

### `tau(mu_0, mu_1, t, pt, temperature)`

τ-BALD：直接关注治疗效应不确定性的获取函数。

**数学公式：**
```
score = (1 / temperature) * log(Var[τ] + ε)
```

其中 `τ = μ₁ - μ₀`。

**特点：**
- 直接关注治疗效应的不确定性
- 不考虑重叠区域
- 可能偏向非重叠区域

### `sundin(mu_0, mu_1, t, pt, temperature)`

Sundin 方法：基于互信息的获取函数。

**数学公式：**
```
τ = μ₁ - μ₀
γ = Φ(-|τ| / √2)  # 标准正态分布的 CDF
predictive_entropy = H(Bernoulli(γ̄))
conditional_entropy = E[H(Bernoulli(γ))]
MI = predictive_entropy - conditional_entropy
score = MI
```

**特点：**
- 基于互信息理论
- 考虑治疗效应的不确定性
- 需要温度参数为 1.0

## 函数字典

所有获取函数注册在 `FUNCTIONS` 字典中：

```python
FUNCTIONS = {
    "random": random,
    "tau": tau,
    "mu": mu,
    "rho": rho,
    "mu-rho": mu_rho,
    "pi": pi,
    "mu-pi": mu_pi,
    "sundin": sundin,
}
```

## 使用示例

### 基本使用
```python
import numpy as np
from causal_bald.library import acquisitions

# 假设已有模型预测
mu_0 = np.random.randn(100, 1000)  # 100 个模型样本，1000 个数据点
mu_1 = np.random.randn(100, 1000)
t = np.random.binomial(1, 0.5, 1000)
pt = np.random.uniform(0.1, 0.9, 1000)

# 计算获取分数
scores = acquisitions.mu_rho(
    mu_0=mu_0,
    mu_1=mu_1,
    t=t,
    pt=pt,
    temperature=0.25,
)

# 选择 Top-K
top_k = 10
indices = np.argsort(scores)[-top_k:]
```

### 在主动学习中使用
```python
# 预测池集
mu_0, mu_1 = model.predict_mus(ds_pool)

# 计算获取分数
acquisition_function = acquisitions.FUNCTIONS["mu-rho"]
scores = acquisition_function(
    mu_0=mu_0,
    mu_1=mu_1,
    t=ds_pool.t,
    pt=pt,
    temperature=0.25,
)

# 选择样本（池集索引）
pool_indices = np.argsort(scores)[-batch_size:]
```

### 温度缩放
```python
# 温度 > 0：使用温度缩放 + 采样
if temperature > 0.0:
    scores_exp = np.exp(scores / temperature)
    probs = scores_exp / scores_exp.sum()
    indices = np.random.choice(len(probs), size=batch_size, p=probs, replace=False)
else:
    # 温度 = 0：确定性 Top-K
    indices = np.argsort(scores)[-batch_size:]
```

## 数学原理

### 信息论基础

获取函数基于信息论中的熵和互信息概念：

1. **熵（Entropy）**：衡量随机变量的不确定性
   ```
   H(X) = -Σ p(x) log p(x)
   ```

2. **互信息（Mutual Information）**：衡量两个随机变量之间的相关性
   ```
   I(X; Y) = H(X) - H(X|Y)
   ```

3. **BALD**：Bayesian Active Learning by Disagreement
   ```
   BALD = H(y|x, D) - E_θ[H(y|x, θ)]
   ```

### 因果推断特殊性

在因果推断中，需要特别关注：

1. **重叠区域**：处理组和对照组都存在的协变量区域
2. **治疗效应**：τ = μ₁ - μ₀ 的不确定性
3. **倾向性评分**：P(T=1|X=x)，用于识别重叠区域

## 选择指南

### 推荐使用

- **μρ-BALD**：在大多数情况下表现最好，推荐作为默认选择

### 根据场景选择

- **关注重叠区域**：使用 `rho` 或 `mu-rho`
- **简单快速**：使用 `mu`
- **有倾向性评分**：使用 `mu-pi` 或 `pi`
- **直接关注治疗效应**：使用 `tau`
- **理论保证**：使用 `sundin`

### 参数设置

- **temperature**：
  - `0.0`：确定性选择（Top-K）
  - `0.25`：推荐值，平衡探索和利用
  - `1.0`：标准温度，用于 `sundin`

## 实现细节

### 数值稳定性

所有函数都使用小的 epsilon (`_eps = 1e-7`) 避免数值问题：

```python
score = log(variance + _eps)
```

### 方差计算

方差通过模型样本间的差异计算：

```python
variance = mu_t.var(axis=0)  # 沿模型样本维度计算方差
```

### 温度缩放

温度参数用于控制获取函数的锐度：

```python
scaled_score = score / temperature
```

较小的温度值使函数更锐利（更确定性），较大的温度值使函数更平滑（更随机）。

## 注意事项

1. **输入格式**：`mu_0` 和 `mu_1` 必须是 2D 数组，第一维是模型样本数
2. **倾向性评分**：`pi` 和 `mu-pi` 需要倾向性评分，其他函数可以传入 `None`
3. **温度参数**：不同函数对温度的敏感度不同
4. **计算效率**：所有函数都是向量化实现，计算高效

## 相关文档

- [库层概览](overview.md)
- [主动学习工作流](../../application/workflows/active_learning.md)
- [模型文档](models/overview.md)

