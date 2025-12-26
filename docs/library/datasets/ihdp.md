# IHDP 数据集文档

## 概述

`causal_bald.library.datasets.ihdp` 模块实现了 IHDP（Infant Health and Development Program）数据集的加载器。这是一个半合成数据集，广泛用于因果推断方法的评估。

## 数据集信息

- **来源**：Infant Health and Development Program 真实数据
- **样本数**：747（观测数据）
- **协变量数**：25（6 个连续变量 + 19 个二元变量）
- **特点**：真实世界的协变量分布 + 合成的治疗效应

## 类定义

### `IHDP`

IHDP 数据集类，继承自 `torch.utils.data.Dataset`。

**初始化参数：**
- `root` (str/Path): 数据根目录
- `split` (str): 数据集划分（`"train"`, `"valid"`, `"test"`）
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

### 1. 数据下载

如果数据文件不存在，会自动下载：

```python
data_path = root / "ihdp.RData"
if not data_path.exists():
    # 从 GitHub 下载
    r = requests.get("https://github.com/vdorie/npci/raw/master/examples/ihdp_sim/data/ihdp.RData")
    with open(data_path, "wb") as f:
        f.write(r.content)
```

### 2. 数据清理

- 移除观测数据中的特定子集（Hill 2011 方法）
- 选择相关协变量

### 3. 标准化

连续协变量进行标准化：
```python
df[_CONTINUOUS_COVARIATES] = preprocessing.StandardScaler().fit_transform(
    df[_CONTINUOUS_COVARIATES]
)
```

### 4. 生成响应面

使用随机系数生成潜在结果：
- `mu0 = exp((x + 0.5) · β_x)`
- `mu1 = (x + 0.5) · β_x - ω`
- `tau = mu1 - mu0`

## 协变量列表

### 连续协变量（6 个）
- `bw`: 出生体重
- `b.head`: 出生头围
- `preterm`: 早产
- `birth.o`: 出生顺序
- `nnhealth`: 新生儿健康
- `momage`: 母亲年龄

### 二元协变量（19 个）
- `sex`: 性别
- `twin`: 双胞胎
- `b.marr`: 出生时婚姻状况
- `mom.lths`: 母亲教育 < 高中
- `mom.hs`: 母亲教育 = 高中
- `mom.scoll`: 母亲教育 = 大学
- `cig`: 吸烟
- `first`: 第一胎
- `booze`: 饮酒
- `drugs`: 药物
- `work.dur`: 工作持续时间
- `prenatal`: 产前护理
- `ark`, `ein`, `har`, `mia`, `pen`, `tex`, `was`: 州标识

## 使用示例

### 基本使用
```python
from causal_bald.library.datasets import IHDP

# 加载训练集
ds_train = IHDP(
    root="assets/",
    split="train",
    mode="mu",
    seed=0,
)

# 加载验证集
ds_valid = IHDP(
    root="assets/",
    split="valid",
    mode="mu",
    seed=0,
)

# 加载测试集
ds_test = IHDP(
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

# 观测结果
y = ds_train.y  # shape: (n_samples,)

# 潜在结果
mu0 = ds_train.mu0  # 对照组
mu1 = ds_train.mu1  # 处理组

# 真实治疗效应
tau = ds_train.tau  # shape: (n_samples,)

# 倾向性评分
pi = ds_train.pi  # shape: (n_samples,)
```

### 倾向性评分模式
```python
# 用于训练倾向性评分模型
ds_pi = IHDP(
    root="assets/",
    split="train",
    mode="pi",  # 倾向性评分模式
    seed=0,
)

# 输入是协变量，目标是处理变量
for batch in DataLoader(ds_pi):
    x, t = batch
    # x: (batch_size, 25)
    # t: (batch_size,)
```

## 数据集划分

数据集使用固定的随机种子进行划分，确保可重复性：

```python
# 使用相同的 seed 确保划分一致
ds_train = IHDP(root="assets/", split="train", mode="mu", seed=0)
ds_valid = IHDP(root="assets/", split="valid", mode="mu", seed=0)
ds_test = IHDP(root="assets/", split="test", mode="mu", seed=0)
```

## 注意事项

1. **数据下载**：首次使用会自动下载数据文件（~1MB）
2. **随机种子**：影响响应面的生成，不同种子会产生不同的治疗效应
3. **模式选择**：`mode="mu"` 用于结果预测，`mode="pi"` 用于倾向性评分
4. **数据格式**：所有数据都是 float32 类型

## 相关文档

- [数据集概览](overview.md)
- [Synthetic 数据集](synthetic.md)
- [Hill 2011 论文](https://www.tandfonline.com/doi/abs/10.1198/jcgs.2011.09220)

