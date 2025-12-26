# Active Learning 数据集文档

## 概述

`causal_bald.library.datasets.active_learning` 模块提供了主动学习数据集包装器，用于将数据集动态分为训练集和池集，支持增量式样本获取。

## 核心类

### `ActiveLearningDataset`

主动学习数据集包装器，将数据集分为训练集和池集。

**初始化参数：**
- `dataset` (Dataset): 原始数据集
- `start_indices` (np.ndarray, 可选): 初始训练样本索引

**属性：**
- `dataset`: 原始数据集
- `training_dataset`: 训练数据集（Subset）
- `pool_dataset`: 池数据集（Subset）
- `acquired_indices`: 已获取的样本索引

**使用示例：**
```python
from causal_bald.library.datasets import IHDP, ActiveLearningDataset

# 加载原始数据集
ds_full = IHDP(root="assets/", split="train", mode="mu", seed=0)

# 创建主动学习数据集
ds_active = ActiveLearningDataset(ds_full)

# 初始获取一些样本
initial_indices = ds_active.get_random_pool_indices(100)
ds_active.acquire(initial_indices)

# 访问训练集和池集
train_ds = ds_active.training_dataset
pool_ds = ds_active.pool_dataset
```

## 主要方法

### `acquire(pool_indices)`

从池集中获取样本到训练集。

**参数：**
- `pool_indices` (np.ndarray/torch.Tensor): 池集中的索引（相对于池集）

**功能：**
1. 将池集索引转换为数据集索引
2. 更新训练掩码和池掩码
3. 更新训练集和池集的索引

**使用示例：**
```python
# 获取池集中的前 10 个样本
pool_indices = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ds_active.acquire(pool_indices)

# 或者使用 numpy 数组
pool_indices = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
ds_active.acquire(pool_indices)
```

### `is_empty()`

检查池集是否为空。

**返回：** bool

**使用示例：**
```python
if ds_active.is_empty():
    print("Pool is empty, all samples acquired!")
```

### `get_random_pool_indices(size)`

随机获取池集中的索引。

**参数：**
- `size` (int): 要获取的样本数量

**返回：** torch.Tensor - 池集中的随机索引

**使用示例：**
```python
# 随机获取 10 个样本
random_indices = ds_active.get_random_pool_indices(10)
ds_active.acquire(random_indices)
```

### `get_dataset_indices(pool_indices)`

将池集索引转换为数据集索引。

**参数：**
- `pool_indices` (np.ndarray/torch.Tensor): 池集索引

**返回：** np.ndarray - 数据集索引

**使用示例：**
```python
pool_indices = torch.tensor([0, 1, 2])
dataset_indices = ds_active.get_dataset_indices(pool_indices)
# 返回原始数据集中的实际索引
```

## 内部实现

### 掩码机制

使用布尔掩码跟踪样本状态：

```python
self.training_mask = np.full((len(dataset),), False)
self.pool_mask = np.full((len(dataset),), True)
```

### 索引更新

```python
def _update_indices(self):
    self.training_dataset.indices = np.nonzero(self.training_mask)[0]
    self.pool_dataset.indices = np.nonzero(self.pool_mask)[0]
```

## 使用场景

### 1. 主动学习循环
```python
ds_active = ActiveLearningDataset(ds_full)

# 初始获取
initial_indices = ds_active.get_random_pool_indices(100)
ds_active.acquire(initial_indices)

# 主动学习循环
for step in range(max_steps):
    # 训练模型
    model.fit(ds_active.training_dataset, ...)
    
    # 预测池集
    scores = model.predict_scores(ds_active.pool_dataset)
    
    # 选择样本
    top_indices = select_top_k(scores, k=10)
    
    # 获取样本
    ds_active.acquire(top_indices)
    
    # 检查是否完成
    if ds_active.is_empty():
        break
```

### 2. 批量获取
```python
# 获取多个批次
for batch_idx in range(num_batches):
    batch_indices = ds_active.get_random_pool_indices(batch_size)
    ds_active.acquire(batch_indices)
```

### 3. 自定义初始集
```python
# 使用预定义的初始索引
initial_indices = np.array([0, 10, 20, 30, 40])
ds_active = ActiveLearningDataset(ds_full, start_indices=initial_indices)
```

## 辅助类

### `RandomFixedLengthSampler`

固定长度的随机采样器，用于小数据集上的多次采样。

**初始化参数：**
- `dataset` (Dataset): 数据集
- `target_length` (int): 目标长度

**功能：**
- 如果 `target_length < len(dataset)`：返回随机排列
- 否则：重复采样以达到目标长度

**使用示例：**
```python
from causal_bald.library.datasets.active_learning import RandomFixedLengthSampler

sampler = RandomFixedLengthSampler(ds_train, target_length=10000)
loader = DataLoader(ds_train, batch_size=100, sampler=sampler)
```

## 注意事项

1. **索引转换**：`acquire()` 接收的是池集索引，不是数据集索引
2. **内存效率**：使用掩码机制，内存效率高
3. **线程安全**：不支持多线程同时获取
4. **状态一致性**：每次获取后会自动更新索引

## 相关文档

- [数据集概览](overview.md)
- [主动学习工作流](../../../application/workflows/active_learning.md)

