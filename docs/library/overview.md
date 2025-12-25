# 库层概览

## 概述

`causal_bald.library` 模块包含了 Causal-BALD 项目的核心算法和模型实现，提供了从数据加载、模型定义、获取函数到可视化的完整功能。

## 模块组织

### 1. 获取函数 (`acquisitions.py`)

实现了多种基于信息论的获取函数，用于主动学习中的样本选择。

**主要功能：**
- 计算样本的信息价值
- 支持多种 BALD 变体
- 考虑因果推断的特殊性（重叠区域）

**相关文档：** [获取函数文档](acquisitions.md)

### 2. 模型 (`models/`)

实现了多种深度学习模型架构。

**模型类型：**
- **Deep Kernel GP**：结合深度学习和高斯过程
- **TARNet**：Treatment-Agnostic Representation Network
- **Neural Network**：标准神经网络

**相关文档：** [模型文档](models/overview.md)

### 3. 模块 (`modules/`)

提供了神经网络构建块和组件。

**模块类型：**
- **Dense**：全连接层模块
- **Convolution**：卷积模块
- **Gaussian Process**：高斯过程模块
- **Variational**：变分推断模块
- **Spectral Norm**：谱归一化模块
- **TARNet**：TARNet 特定模块

**相关文档：** [模块文档](modules/overview.md)

### 4. 数据集 (`datasets/`)

实现了多种数据集加载器。

**数据集类型：**
- **IHDP**：Infant Health and Development Program 数据集
- **Synthetic**：合成数据集
- **CMNIST**：Causal MNIST 数据集
- **Active Learning**：主动学习数据集包装器

**相关文档：** [数据集文档](datasets/overview.md)

### 5. 可视化 (`plotting.py`)

提供了结果可视化和分析功能。

**可视化类型：**
- 数据集可视化
- 获取分布可视化
- 收敛曲线
- 误差分析

**相关文档：** [可视化文档](plotting.md)

### 6. 工具函数 (`utils.py`)

提供了各种工具函数。

**功能：**
- 倾向性评分计算
- 数据预处理
- 数学工具函数

## 模块关系

```
acquisitions.py
    ↓ (使用)
models/
    ├── core.py (基类)
    ├── deep_kernel.py
    ├── tarnet.py
    └── neural_network.py
        ↓ (使用)
    modules/
        ├── dense.py
        ├── convolution.py
        ├── gaussian_process.py
        ├── variational.py
        └── spectral_norm.py
        ↓ (使用)
    datasets/
        ├── ihdp.py
        ├── synthetic.py
        ├── hcmnist.py
        └── active_learning.py
```

## 核心概念

### 获取函数

获取函数用于评估样本的信息价值，在主动学习中用于选择最有价值的样本。

**数学原理：**
- 基于信息论（熵、互信息）
- 考虑模型不确定性
- 关注因果推断中的重叠区域

### 模型架构

#### Deep Kernel GP
- 深度编码器提取特征
- 变分高斯过程建模不确定性
- 支持多种核函数

#### TARNet
- 共享特征提取器
- 独立的处理组和对照组头
- 支持集成多个模型

### 数据集格式

所有数据集遵循统一接口：

```python
class Dataset:
    def __getitem__(self, idx):
        return (x, t), y
    
    @property
    def x(self):  # 协变量
    @property
    def t(self):  # 处理指示变量
    @property
    def y(self):  # 结果变量
    @property
    def mu0(self):  # 对照组潜在结果
    @property
    def mu1(self):  # 处理组潜在结果
    @property
    def tau(self):  # 真实治疗效应
    @property
    def dim_input(self):  # 输入维度
```

## 使用流程

### 1. 加载数据集
```python
from causal_bald.library import datasets

ds = datasets.IHDP(root="assets/", split="train", mode="mu", seed=0)
```

### 2. 初始化模型
```python
from causal_bald.library import models

model = models.DeepKernelGP(
    job_dir="checkpoints/",
    kernel="Matern32",
    num_inducing_points=100,
    # ... 更多参数
)
```

### 3. 训练模型
```python
model.fit(ds_train, ds_valid)
```

### 4. 预测
```python
mu_0, mu_1 = model.predict_mus(ds_test)
```

### 5. 计算获取分数
```python
from causal_bald.library import acquisitions

scores = acquisitions.mu_rho(
    mu_0=mu_0,
    mu_1=mu_1,
    t=ds.t,
    pt=pt,
    temperature=0.25,
)
```

## 扩展指南

### 添加新获取函数

1. 在 `acquisitions.py` 中实现函数：
```python
def new_acquisition(mu_0, mu_1, t, pt, temperature):
    # 实现获取函数逻辑
    return scores
```

2. 添加到 `FUNCTIONS` 字典：
```python
FUNCTIONS["new-acquisition"] = new_acquisition
```

### 添加新模型

1. 继承 `core.PyTorchModel`：
```python
class NewModel(core.PyTorchModel):
    def __init__(self, ...):
        super().__init__(...)
        # 初始化网络
    
    def train_step(self, engine, batch):
        # 实现训练步骤
    
    def predict_mus(self, ds):
        # 实现预测
```

2. 在 `models/__init__.py` 中导出

### 添加新数据集

1. 继承 `torch.utils.data.Dataset`：
```python
class NewDataset(data.Dataset):
    def __init__(self, ...):
        # 初始化数据
    
    def __getitem__(self, idx):
        return (x, t), y
```

2. 添加到 `datasets.DATASETS` 字典

## 相关文档

- [获取函数](acquisitions.md)
- [模型文档](models/overview.md)
- [模块文档](modules/overview.md)
- [数据集文档](datasets/overview.md)
- [可视化文档](plotting.md)

