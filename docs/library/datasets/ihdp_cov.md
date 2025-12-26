# IHDP-Cov 数据集文档

## 概述

`causal_bald.library.datasets.ihdp_cov` 模块实现了 IHDP-Cov 数据集，这是 IHDP 数据集的一个变体，用于测试协变量选择的影响。

## 数据集信息

- **基础数据**：IHDP 数据集
- **样本数**：约 672（观测数据，过滤后）
- **协变量数**：25（与 IHDP 相同）
- **特点**：使用特定的协变量子集进行训练

## 类定义

### `IHDPCov`

IHDP-Cov 数据集类，继承自 `torch.utils.data.Dataset`。

**初始化参数：**
- `root` (str/Path): 数据根目录
- `split` (str): 数据集划分（`"train"`, `"test"`）
- `mode` (str): 模式（`"mu"` 用于结果预测，`"pi"` 用于倾向性评分）
- `seed` (int): 随机种子

**属性：**
- `x` (np.ndarray): 协变量，shape `(n_samples, 25)`
- `t` (np.ndarray): 处理指示变量，shape `(n_samples,)`
- `y` (np.ndarray): 观测结果，shape `(n_samples,)`
- `mu0` (np.ndarray): 对照组潜在结果
- `mu1` (np.ndarray): 处理组潜在结果
- `tau` (np.ndarray): 真实治疗效应（μ₁ - μ₀）
- `pi` (np.ndarray): 倾向性评分
- `dim_input` (int): 输入维度（25）

## 数据预处理

### 1. 数据加载和清理

与 IHDP 相同：
- 下载数据（如需要）
- 移除特定子集（Hill 2011 方法）
- 选择相关协变量

### 2. 生成响应面

与 IHDP 相同：
- 使用随机系数生成潜在结果
- `mu0 = exp((x + 0.5) · β_x)`
- `mu1 = (x + 0.5) · β_x - ω`

### 3. 数据集划分

```python
# 训练/测试划分
df_train, df_test = model_selection.train_test_split(
    df, test_size=0.1, random_state=seed
)

# 只使用 b.marr == 1 的样本作为训练集
df_train = df_train[df["b.marr"] == 1]
```

**关键区别：**
- 训练集只包含 `b.marr == 1` 的样本
- 测试集包含所有样本
- 没有验证集划分

### 4. 标准化

```python
# 基于训练集计算均值和标准差
self.y_mean = df_train["y"].mean() if mode == "mu" else 0.0
self.y_std = df_train["y"].std() if mode == "mu" else 1.0

# 标准化结果
self.y = (df["y"] - self.y_mean) / self.y_std
```

## 使用示例

### 基本使用
```python
from causal_bald.library.datasets import IHDPCov

# 加载训练集
ds_train = IHDPCov(
    root="assets/",
    split="train",
    mode="mu",
    seed=0,
)

# 加载测试集
ds_test = IHDPCov(
    root="assets/",
    split="test",
    mode="mu",
    seed=0,
)
```

### 在数据加载器中使用
```python
from torch.utils.data import DataLoader

loader = DataLoader(
    ds_train,
    batch_size=100,
    shuffle=True,
)

for batch in loader:
    (x, t), y = batch
    # x: (batch_size, 25)
    # t: (batch_size,)
    # y: (batch_size,)
```

### 访问属性
```python
# 协变量
x = ds_train.x  # shape: (n_samples, 25)

# 处理变量
t = ds_train.t  # shape: (n_samples,)

# 观测结果（已标准化）
y = ds_train.y  # shape: (n_samples,)

# 潜在结果
mu0 = ds_train.mu0  # 对照组
mu1 = ds_train.mu1  # 处理组

# 真实治疗效应
tau = ds_train.tau  # shape: (n_samples,)

# 倾向性评分
pi = ds_train.pi  # shape: (n_samples,)
```

## 与 IHDP 的区别

| 特性 | IHDP | IHDP-Cov |
|------|------|----------|
| 训练集 | 全部数据 | 只包含 b.marr==1 |
| 验证集 | 有 | 无 |
| 测试集 | 有 | 有 |
| 标准化 | 全局 | 基于训练集 |
| 用途 | 标准基准 | 协变量选择测试 |

## 使用场景

IHDP-Cov 主要用于：
1. **协变量选择研究**：测试不同协变量子集的影响
2. **选择偏差分析**：研究训练集选择偏差的影响
3. **方法对比**：与其他数据集对比方法性能

## 注意事项

1. **训练集限制**：训练集只包含 `b.marr == 1` 的样本，样本数较少
2. **无验证集**：没有独立的验证集，需要从训练集中划分
3. **标准化**：结果基于训练集标准化，注意反标准化
4. **随机种子**：影响响应面生成和数据集划分

## 相关文档

- [数据集概览](overview.md)
- [IHDP 数据集](ihdp.md)
- [Synthetic 数据集](synthetic.md)

