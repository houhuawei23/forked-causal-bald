# TARNet 模块文档

## 概述

`causal_bald.library.modules.tarnet` 模块实现了 TARNet（Treatment-Agnostic Representation Network）架构，用于因果效应估计。

## 核心思想

TARNet 使用共享的特征提取器和独立的处理组/对照组头来估计因果效应：
- **共享编码器**：提取与处理无关的特征表示
- **独立头**：分别为处理组和对照组建模结果分布

## 主要类

### `TARNet`

TARNet 网络架构。

**初始化参数：**
- `architecture` (str): 编码器架构（`"basic"` 或 `"resnet"`）
- `dim_input` (int/list): 输入维度（整数或图像维度列表）
- `dim_hidden` (int): 隐藏层维度
- `dim_output` (int): 输出维度
- `depth` (int): 网络深度
- `negative_slope` (float): LeakyReLU 负斜率
- `batch_norm` (bool): 是否使用批归一化
- `dropout_rate` (float): Dropout 率
- `spectral_norm` (float): 谱归一化系数

**结构：**
```
Input: [covariates, treatment]
    ↓
Encoder (共享特征提取器)
    ├── 1D 输入: dense.NeuralNetwork
    ├── 图像输入: convolution.ResNet
    └── 1维输入: Identity
    ↓
Treatment-Specific Encoders
    ├── t0_encoder (对照组)
    └── t1_encoder (处理组)
    ↓
Outcome Density
    ├── 高维: variational.SplitGMM
    └── 1维: dense.NeuralNetwork → variational.GMM
    ↓
Output: Distribution
```

**属性：**
- `dim_output`: 输出维度

**使用示例：**
```python
from causal_bald.library.modules import tarnet

# 1D 输入
model = tarnet.TARNet(
    architecture="resnet",
    dim_input=25,
    dim_hidden=200,
    dim_output=1,
    depth=3,
    negative_slope=-1.0,
    batch_norm=True,
    dropout_rate=0.1,
    spectral_norm=0.95,
)

# 前向传播
inputs = torch.cat([covariates, treatment], dim=-1)
output = model(inputs)  # Distribution
```

## 架构细节

### 编码器选择

根据输入维度自动选择编码器：

#### 1. 图像输入（`isinstance(dim_input, list)`）
```python
encoder = convolution.ResNet(
    dim_input=dim_input,
    layers=[2] * depth,
    base_width=dim_hidden // 8,
    # ...
)
```

#### 2. 1维输入（`dim_input == 1`）
```python
encoder = nn.Identity()
encoder.dim_output = 1
```

#### 3. 其他情况
```python
encoder = dense.NeuralNetwork(
    architecture=architecture,
    dim_input=dim_input,
    dim_hidden=dim_hidden,
    depth=depth,
    # ...
)
```

### 处理特定编码器

#### t0_encoder（对照组）
```python
t0_encoder = nn.Sequential(
    dense.ResidualDense(...),  # 第一层
    dense.ResidualDense(...),  # 第二层
    dense.Activation(...),     # 激活
)
```

#### t1_encoder（处理组）
结构同 `t0_encoder`，但参数独立。

**特殊处理：**
- 如果 `encoder.dim_output == 1`，使用 `nn.Identity()`
- 否则使用残差层序列

### 结果密度

根据编码器输出维度选择：

#### 高维（`encoder.dim_output != 1`）
```python
outcome_density = variational.SplitGMM(
    dim_input=dim_hidden,
    dim_output=dim_output,
)
```

#### 1维（`encoder.dim_output == 1`）
```python
outcome_density = nn.Sequential(
    dense.NeuralNetwork(
        dim_input=dim_input + 1,  # 包含 treatment
        dim_hidden=dim_hidden,
        depth=depth,
        # ...
    ),
    variational.GMM(
        dim_input=dim_hidden,
        dim_output=dim_output,
    ),
)
```

## 前向传播

```python
def forward(self, inputs):
    # 分离协变量和处理变量
    phi = self.encoder(inputs[:, :-1])  # 特征提取
    t = inputs[:, -1:]                   # 处理变量
    
    # 根据处理变量选择编码器
    phi = (1 - t) * self.t0_encoder(phi) + t * self.t1_encoder(phi)
    
    # 预测结果分布
    return self.outcome_density(torch.cat([phi, t], dim=-1))
```

## 使用场景

### 1. 标准因果推断
```python
model = tarnet.TARNet(
    dim_input=25,
    dim_hidden=200,
    dim_output=1,
    depth=3,
    # ...
)

# 训练
output = model(inputs)
loss = -output.log_prob(targets).mean()

# 预测
mu_0, mu_1 = predict_mus(model, dataset)
tau = mu_1 - mu_0  # 治疗效应
```

### 2. 图像数据
```python
model = tarnet.TARNet(
    dim_input=[3, 28, 28],  # 图像维度
    dim_hidden=200,
    dim_output=1,
    depth=3,
    # ...
)
```

### 3. 1维数据
```python
model = tarnet.TARNet(
    dim_input=1,
    dim_hidden=100,
    dim_output=1,
    depth=2,
    # ...
)
```

## 预测治疗效应

### 预测 μ₀ 和 μ₁
```python
def predict_mus(model, dataset):
    model.eval()
    mu_0 = []
    mu_1 = []
    
    with torch.no_grad():
        for batch in dataloader:
            covariates = batch[0][:, :-1]
            treatment = batch[0][:, -1:]
            
            # 预测对照组
            inputs_0 = torch.cat([covariates, torch.zeros_like(treatment)], dim=-1)
            output_0 = model(inputs_0)
            mu_0.append(output_0.mean)
            
            # 预测处理组
            inputs_1 = torch.cat([covariates, torch.ones_like(treatment)], dim=-1)
            output_1 = model(inputs_1)
            mu_1.append(output_1.mean)
    
    return torch.cat(mu_0), torch.cat(mu_1)
```

## 设计优势

1. **共享表示**：编码器学习与处理无关的特征
2. **灵活建模**：独立的头允许不同的处理效应
3. **不确定性**：使用变分分布提供不确定性估计
4. **多模态支持**：GMM 可以建模多模态分布

## 注意事项

1. **输入格式**：输入必须包含处理变量作为最后一维
2. **维度匹配**：确保所有维度匹配
3. **训练模式**：训练时使用 `model.train()`，预测时使用 `model.eval()`
4. **1维特殊情况**：1维输入有特殊的处理逻辑

## 相关文档

- [模块概览](overview.md)
- [全连接层模块](dense.md)
- [卷积模块](convolution.md)
- [变分推断模块](variational.md)
- [TARNet 模型](../models/tarnet.md)

