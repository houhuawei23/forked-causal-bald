# CMNIST 数据集文档

## 概述

`causal_bald.library.datasets.hcmnist` 模块实现了 CMNIST（Causal MNIST）数据集，这是一个基于 MNIST 的因果推断数据集，用于测试深度学习方法。

## 数据集信息

- **基础数据**：MNIST 手写数字数据集
- **协变量维度**：`[1, 28, 28]`（图像）
- **特点**：高维图像数据 + 复杂的治疗分配机制
- **用途**：测试深度学习方法在图像数据上的表现

## 类定义

### `HCMNIST`

CMNIST 数据集类，继承自 `torchvision.datasets.MNIST`。

**初始化参数：**
- `root` (str): 数据根目录
- `split` (str): 数据集划分（`"train"`, `"valid"`, `"test"`）
- `mode` (str): 模式（`"mu"` 用于结果预测，`"pi"` 用于倾向性评分）
- `beta` (float, 默认=2.0): x 对 t 的影响系数
- `sigma_y` (float, 默认=0.01): 结果噪声标准差
- `domain` (float, 默认=3.0): 域范围
- `subsample` (dict, 可选): 子采样字典，格式 `{digit: frequency}`
- `seed` (int, 默认=1331): 随机种子
- `transform` (Callable, 可选): 图像变换
- `target_transform` (Callable, 可选): 目标变换
- `download` (bool, 默认=True): 是否自动下载

**属性：**
- `x` (np.ndarray): 预处理后的图像数据，shape `(n_samples, 1, 28, 28)`
- `t` (np.ndarray): 处理指示变量，shape `(n_samples,)`
- `y` (np.ndarray): 观测结果，shape `(n_samples,)`
- `mu0` (np.ndarray): 对照组潜在结果
- `mu1` (np.ndarray): 处理组潜在结果
- `tau` (np.ndarray): 真实治疗效应（μ₁ - μ₀）
- `pi` (np.ndarray): 倾向性评分
- `phi` (np.ndarray): 协变量嵌入（1维表示）
- `dim_input` (list): 输入维度 `[1, 28, 28]`
- `dim_treatment` (int): 处理维度（1）
- `dim_output` (int): 输出维度（1）

## 数据生成过程

### 1. 加载 MNIST 数据

```python
# 继承自 torchvision.datasets.MNIST
super(HCMNIST, self).__init__(
    root, train=(split == "train" or split == "valid"), ...
)
```

### 2. 数据预处理

```python
# 展平图像
self.data = self.data.view(len(self.targets), -1).numpy()

# 标准化
self.x = ((self.data.astype("float32") / 255.0) - 0.1307) / 0.3081
```

### 3. 数据集划分

```python
if train:
    data_train, data_valid, targets_train, targets_valid = train_test_split(
        self.data, self.targets, test_size=0.3, random_state=seed
    )
    self.data = data_train if split == "train" else data_valid
    self.targets = targets_train if split == "train" else targets_valid
```

### 4. 子采样（可选）

```python
if subsample is not None:
    for digit, frequency in subsample.items():
        idx = np.where(self.targets == digit)[0]
        idx_delete = rng.choice(
            idx, size=int(len(idx) * (1 - frequency)), replace=False
        )
        self.data = np.delete(self.data, idx_delete, axis=0)
        self.targets = np.delete(self.targets, idx_delete, axis=0)
```

### 5. 生成协变量嵌入（phi）

使用预训练的 phi 模型将图像映射到 1 维表示：

```python
self.phi_model = fit_phi_model(
    root=root,
    edges=torch.arange(-domain, domain + 0.1, (2 * domain) / 10),
)
phi = self.phi
```

### 6. 生成倾向性评分

```python
self.pi = utils.complete_propensity(
    x=phi, u=0.0, lambda_=1.0, beta=beta
).astype("float32").ravel()
```

### 7. 生成处理变量

```python
self.t = rng.binomial(1, self.pi).astype("float32")
```

### 8. 生成潜在结果

```python
self.mu0 = utils.f_mu(x=phi, t=0.0, u=0.0, gamma=0.0).astype("float32").ravel()
self.mu1 = utils.f_mu(x=phi, t=1.0, u=0.0, gamma=0.0).astype("float32").ravel()
```

### 9. 生成观测结果

```python
eps = (sigma_y * rng.normal(size=self.t.shape)).astype("float32")
self.y0 = self.mu0 + eps
self.y1 = self.mu1 + eps
self.y = self.t * self.y1 + (1 - self.t) * self.y0
```

## 使用示例

### 基本使用
```python
from causal_bald.library.datasets import HCMNIST

# 加载训练集
ds_train = HCMNIST(
    root="assets/",
    split="train",
    mode="mu",
    beta=2.0,
    sigma_y=0.01,
    domain=3.0,
    seed=0,
)

# 加载验证集
ds_valid = HCMNIST(
    root="assets/",
    split="valid",
    mode="mu",
    beta=2.0,
    sigma_y=0.01,
    domain=3.0,
    seed=0,
)

# 加载测试集
ds_test = HCMNIST(
    root="assets/",
    split="test",
    mode="mu",
    beta=2.0,
    sigma_y=0.01,
    domain=3.0,
    seed=0,
)
```

### 子采样
```python
# 只保留部分数字
ds_train = HCMNIST(
    root="assets/",
    split="train",
    mode="mu",
    subsample={
        0: 0.5,  # 保留 50% 的数字 0
        1: 0.8,  # 保留 80% 的数字 1
        2: 1.0,  # 保留 100% 的数字 2
        # ...
    },
    seed=0,
)
```

### 在数据加载器中使用
```python
from torch.utils.data import DataLoader

loader = DataLoader(
    ds_train,
    batch_size=64,
    shuffle=True,
)

for batch in loader:
    (x, t), y = batch
    # x: (batch_size, 1, 28, 28) - 图像
    # t: (batch_size,) - 处理变量
    # y: (batch_size,) - 结果
```

### 访问属性
```python
# 图像数据
x = ds_train.x  # shape: (n_samples, 1, 28, 28)

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

# 协变量嵌入
phi = ds_train.phi  # shape: (n_samples,)
```

### 倾向性评分模式
```python
# 用于训练倾向性评分模型
ds_pi = HCMNIST(
    root="assets/",
    split="train",
    mode="pi",  # 倾向性评分模式
    beta=2.0,
    seed=0,
)

# 输入是图像，目标是处理变量
for batch in DataLoader(ds_pi):
    x, t = batch
    # x: (batch_size, 1, 28, 28)
    # t: (batch_size,)
```

## 参数说明

### `beta`（x 对 t 的影响系数）

- **默认值**：2.0
- **作用**：控制协变量嵌入对处理分配的影响强度
- **影响**：较大的 beta 导致更强的选择偏差

### `sigma_y`（结果噪声标准差）

- **默认值**：0.01
- **作用**：控制结果的噪声水平
- **注意**：CMNIST 使用较小的噪声（相比 Synthetic）

### `domain`（域范围）

- **默认值**：3.0
- **作用**：定义 phi 的域范围 `[-domain, domain]`
- **用途**：用于生成 phi 模型的边缘

### `subsample`（子采样）

- **类型**：dict，格式 `{digit: frequency}`
- **作用**：控制每个数字的保留比例
- **用途**：创建不平衡数据集

## 可视化

### 可视化数据集
```python
import matplotlib.pyplot as plt

# 显示图像样本
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i in range(10):
    ax = axes[i // 5, i % 5]
    ax.imshow(ds_train.x[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"t={ds_train.t[i]:.0f}, y={ds_train.y[i]:.2f}")
    ax.axis('off')
plt.tight_layout()
plt.show()
```

## 注意事项

1. **数据下载**：首次使用会自动下载 MNIST 数据
2. **phi 模型**：需要预训练的 phi 模型（会自动处理）
3. **内存使用**：图像数据占用较多内存
4. **处理时间**：phi 模型拟合可能需要一些时间
5. **随机种子**：影响所有随机生成的数据

## 与其他数据集对比

| 特性 | CMNIST | IHDP | Synthetic |
|------|--------|------|-----------|
| 输入类型 | 图像 | 表格 | 1维 |
| 输入维度 | [1,28,28] | 25 | 1 |
| 复杂度 | 高 | 中 | 低 |
| 用途 | 深度学习测试 | 标准基准 | 方法验证 |

## 相关文档

- [数据集概览](overview.md)
- [IHDP 数据集](ihdp.md)
- [Synthetic 数据集](synthetic.md)
- [数据集工具函数](utils.md)

