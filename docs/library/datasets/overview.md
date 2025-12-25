# 数据集概览

## 概述

`causal_bald.library.datasets` 模块实现了多种用于因果推断的数据集加载器。所有数据集都继承自 `torch.utils.data.Dataset`，遵循统一的接口。

## 数据集列表

### 1. IHDP (`ihdp.py`)

Infant Health and Development Program 数据集，半合成数据集。

**特点：**
- 25 个协变量
- 747 个样本
- 真实世界的协变量分布
- 合成的治疗效应

**相关文档：** [IHDP 数据集](ihdp.md)

### 2. Synthetic (`synthetic.py`)

合成数据集，用于方法验证和调试。

**特点：**
- 1 维协变量
- 可配置的参数（β, σ, 双峰分布等）
- 已知真实治疗效应
- 快速生成

**相关文档：** [Synthetic 数据集](synthetic.md)

### 3. CMNIST (`hcmnist.py`)

Causal MNIST 数据集，基于 MNIST 的因果推断数据集。

**特点：**
- 图像数据（28x28）
- 高维协变量
- 复杂的治疗分配机制
- 用于测试深度学习方法

**相关文档：** [CMNIST 数据集](hcmnist.md)

### 4. Active Learning Dataset (`active_learning.py`)

主动学习数据集包装器，将数据集分为训练集和池集。

**特点：**
- 动态样本获取
- 训练集和池集管理
- 支持增量学习

**相关文档：** [Active Learning 数据集](active_learning.md)

## 数据集字典

所有数据集注册在 `DATASETS` 字典中：

```python
from causal_bald.library.datasets import DATASETS

ds = DATASETS["ihdp"](root="assets/", split="train", mode="mu", seed=0)
```

## 统一接口

所有数据集都提供以下接口：

### 初始化
```python
dataset = Dataset(
    root="path/to/data",  # 数据路径
    split="train",        # 数据集划分
    mode="mu",           # 模式（mu/pi）
    seed=0,              # 随机种子
)
```

### 属性
- `x`: 协变量
- `t`: 处理指示变量（0 或 1）
- `y`: 观测结果
- `mu0`: 对照组潜在结果
- `mu1`: 处理组潜在结果
- `tau`: 真实治疗效应（μ₁ - μ₀）
- `pi`: 倾向性评分
- `dim_input`: 输入维度

### 访问
```python
# 通过索引访问
x, t, y = dataset[0]

# 通过属性访问
all_x = dataset.x
all_t = dataset.t
all_y = dataset.y
```

## 数据集模式

### `mode="mu"`

用于结果预测任务，返回：
- `y`: 观测结果
- `mu0`, `mu1`: 潜在结果
- `tau`: 治疗效应

### `mode="pi"`

用于倾向性评分预测任务，返回：
- `pi`: 倾向性评分
- `t`: 处理指示变量

## 使用示例

### 加载数据集
```python
from causal_bald.library import datasets

# IHDP
ds_train = datasets.IHDP(
    root="assets/",
    split="train",
    mode="mu",
    seed=0,
)

# Synthetic
ds_synthetic = datasets.Synthetic(
    num_examples=10000,
    mode="mu",
    beta=2.0,
    sigma_y=1.0,
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
    # 训练模型
```

### 主动学习中使用
```python
from causal_bald.library.datasets import ActiveLearningDataset

ds_active = ActiveLearningDataset(ds_train)
ds_active.acquire([0, 1, 2, ...])  # 获取样本

# 访问训练集和池集
train_ds = ds_active.training_dataset
pool_ds = ds_active.pool_dataset
```

## 数据预处理

数据集可能包含以下预处理步骤：

1. **标准化**：协变量标准化
2. **划分**：训练/验证/测试集划分
3. **采样**：根据模式采样不同的数据

## 相关文档

- [库层概览](../overview.md)
- [IHDP 数据集](ihdp.md)
- [Synthetic 数据集](synthetic.md)
- [CMNIST 数据集](hcmnist.md)
- [Active Learning 数据集](active_learning.md)

