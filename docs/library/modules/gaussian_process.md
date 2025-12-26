# 高斯过程模块文档

## 概述

`causal_bald.library.modules.gaussian_process` 模块提供了变分高斯过程和深度核高斯过程的实现，用于在深度学习中引入不确定性估计。

## 主要类

### `VariationalGP`

变分高斯过程，基于 GPyTorch 实现。

**初始化参数：**
- `num_outputs` (int): 输出数量
- `initial_lengthscale` (float/torch.Tensor): 初始长度尺度
- `initial_inducing_points` (torch.Tensor): 初始诱导点，shape `(n_inducing_points, dim)`
- `separate_inducing_points` (bool): 是否为每个输出使用独立的诱导点
- `kernel` (str): 核函数类型（`"RBF"`, `"Matern12"`, `"Matern32"`, `"Matern52"`, `"RQ"`）
- `ard` (int, optional): ARD（自动相关性确定）维度数
- `lengthscale_prior` (bool): 是否使用长度尺度先验

**支持的核函数：**
- `RBF`: 径向基函数核（平方指数核）
- `Matern12`: Matern 核（ν=1/2）
- `Matern32`: Matern 核（ν=3/2）
- `Matern52`: Matern 核（ν=5/2）
- `RQ`: 有理二次核

**结构：**
```
VariationalStrategy
    ↓
VariationalDistribution (Cholesky)
    ↓
Kernel (RBF/Matern/RQ)
    ↓
Mean Module (ConstantMean)
```

**使用示例：**
```python
from causal_bald.library.modules import gaussian_process
import torch

# 初始化诱导点
n_inducing = 100
dim = 50
initial_inducing_points = torch.randn(n_inducing, dim)
initial_lengthscale = 1.0

# 创建变分 GP
gp = gaussian_process.VariationalGP(
    num_outputs=1,
    initial_lengthscale=initial_lengthscale,
    initial_inducing_points=initial_inducing_points,
    separate_inducing_points=False,
    kernel="Matern32",
    ard=None,
    lengthscale_prior=False,
)

# 前向传播
x = torch.randn(10, dim)
output = gp(x)  # MultivariateNormal distribution
```

### `DeepKernelGP`

深度核高斯过程，结合深度编码器和高斯过程。

**初始化参数：**
- `encoder` (nn.Module): 深度编码器（特征提取器）
- `gp` (VariationalGP): 变分高斯过程

**结构：**
```
Input: [covariates, treatment]
    ↓
Encoder (提取特征)
    ↓
Concatenate [features, treatment]
    ↓
GP (高斯过程)
    ↓
Output: MultivariateNormal
```

**使用示例：**
```python
from causal_bald.library.modules import dense, gaussian_process

# 创建编码器
encoder = dense.NeuralNetwork(
    dim_input=25,
    dim_hidden=200,
    depth=3,
    # ... 其他参数
)

# 创建 GP
gp = gaussian_process.VariationalGP(...)

# 创建 Deep Kernel GP
model = gaussian_process.DeepKernelGP(
    encoder=encoder,
    gp=gp,
)

# 前向传播
inputs = torch.cat([covariates, treatment], dim=-1)
output = model(inputs)  # MultivariateNormal distribution
```

## 工具函数

### `initial_values_for_GP(train_dataset, feature_extractor, n_inducing_points, device)`

初始化 GP 的诱导点和长度尺度。

**参数：**
- `train_dataset` (Dataset): 训练数据集
- `feature_extractor` (nn.Module): 特征提取器（编码器）
- `n_inducing_points` (int): 诱导点数量
- `device` (torch.device): 计算设备

**返回：** tuple - `(initial_inducing_points, initial_lengthscale)`

**功能：**
1. 从训练集中采样特征
2. 使用 K-means 聚类初始化诱导点
3. 计算初始长度尺度（基于特征间距离）

**使用示例：**
```python
initial_inducing_points, initial_lengthscale = gaussian_process.initial_values_for_GP(
    train_dataset=ds_train,
    feature_extractor=encoder,
    n_inducing_points=100,
    device=torch.device("cuda"),
)
```

### `_get_initial_inducing_points(f_X_sample, n_inducing_points)`

使用 K-means 聚类获取初始诱导点。

**参数：**
- `f_X_sample` (np.ndarray): 特征样本
- `n_inducing_points` (int): 诱导点数量

**返回：** torch.Tensor - 初始诱导点

**实现：**
```python
kmeans = cluster.MiniBatchKMeans(n_clusters=n_inducing_points)
kmeans.fit(f_X_sample)
initial_inducing_points = torch.from_numpy(kmeans.cluster_centers_)
```

### `_get_initial_lengthscale(f_X_samples)`

计算初始长度尺度。

**参数：**
- `f_X_samples` (torch.Tensor): 特征样本

**返回：** torch.Tensor - 初始长度尺度

**实现：**
```python
initial_lengthscale = torch.pdist(f_X_samples).mean()
```

## 变分推断

### 变分策略

使用 `VariationalStrategy` 进行变分推断：
- **变分分布**：`CholeskyVariationalDistribution`
- **诱导点**：可学习的参数
- **多任务支持**：`IndependentMultitaskVariationalStrategy`

### ELBO 损失

训练时使用变分下界（ELBO）作为损失函数：

```python
from gpytorch import mlls

loss = mlls.VariationalELBO(
    likelihood=likelihood,
    model=gp,
    num_data=len(train_dataset),
)
```

## 核函数选择

### RBF 核
- **特点**：平滑、无限可微
- **适用**：平滑函数
- **参数**：长度尺度

### Matern 核
- **Matern12**：一次可微
- **Matern32**：一次可微，更平滑
- **Matern52**：两次可微
- **适用**：不同平滑度的函数

### RQ 核
- **特点**：有理二次形式
- **适用**：多尺度模式

**推荐：** `Matern32`（平衡平滑度和灵活性）

## 诱导点选择

### 数量选择
- **少量数据**：`n_inducing_points = min(100, len(train_dataset) // 10)`
- **大量数据**：`n_inducing_points = 100-200`
- **权衡**：更多诱导点 → 更准确但计算更慢

### 初始化方法
- **K-means 聚类**：默认方法，基于数据分布
- **随机采样**：简单但可能不够好
- **均匀采样**：适用于均匀分布的数据

## 完整示例

### 创建 Deep Kernel GP 模型
```python
import torch
from causal_bald.library.modules import dense, gaussian_process
from gpytorch import mlls, likelihoods

# 1. 创建编码器
encoder = dense.NeuralNetwork(
    architecture="resnet",
    dim_input=25,
    dim_hidden=200,
    depth=3,
    negative_slope=-1.0,
    batch_norm=True,
    dropout_rate=0.1,
    spectral_norm=0.95,
    activate_output=False,
)

# 2. 初始化 GP 参数
initial_inducing_points, initial_lengthscale = gaussian_process.initial_values_for_GP(
    train_dataset=ds_train,
    feature_extractor=encoder,
    n_inducing_points=100,
    device=torch.device("cuda"),
)

# 3. 创建变分 GP
gp = gaussian_process.VariationalGP(
    num_outputs=1,
    initial_lengthscale=initial_lengthscale,
    initial_inducing_points=initial_inducing_points,
    kernel="Matern32",
)

# 4. 创建 Deep Kernel GP
model = gaussian_process.DeepKernelGP(
    encoder=encoder,
    gp=gp,
)

# 5. 创建似然函数
likelihood = likelihoods.GaussianLikelihood()

# 6. 创建损失函数
loss = mlls.VariationalELBO(
    likelihood=likelihood,
    model=gp,
    num_data=len(ds_train),
)
```

### 预测不确定性
```python
model.eval()
likelihood.eval()

with torch.no_grad():
    # 预测
    output = model(inputs)
    
    # 采样（用于不确定性估计）
    samples = output.sample(torch.Size([1000]))  # 1000 个样本
    
    # 计算均值和方差
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
```

## 注意事项

1. **内存使用**：诱导点数量影响内存使用
2. **计算复杂度**：O(n_inducing_points²)
3. **初始化**：好的初始化对训练很重要
4. **设备管理**：确保编码器和 GP 在同一设备上

## 相关文档

- [模块概览](overview.md)
- [Deep Kernel GP 模型](../models/deep_kernel.md)
- [GPyTorch 文档](https://docs.gpytorch.ai/)

