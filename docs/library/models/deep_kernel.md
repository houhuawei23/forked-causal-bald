# Deep Kernel GP 模型文档

## 概述

`causal_bald.library.models.deep_kernel` 模块实现了 Deep Kernel GP（深度核高斯过程）模型，结合了深度学习和高斯过程的优势，能够量化预测不确定性。

## 核心思想

Deep Kernel GP 使用深度神经网络提取特征，然后在高维特征空间中使用高斯过程建模，从而：
- 利用深度学习的特征提取能力
- 通过高斯过程提供不确定性估计
- 适用于因果效应估计任务

## 类定义

### `DeepKernelGP`

Deep Kernel GP 模型类，继承自 `core.PyTorchModel`。

**初始化参数：**
- `job_dir` (str): 工作目录，用于保存检查点
- `kernel` (str): GP 核函数类型（`"RBF"`, `"Matern12"`, `"Matern32"`, `"Matern52"`）
- `num_inducing_points` (int): 诱导点数量
- `inducing_point_dataset` (Dataset): 用于初始化诱导点的数据集
- `architecture` (str): 编码器架构（`"basic"` 或 `"resnet"`）
- `dim_input` (int/list): 输入维度（整数或图像维度列表）
- `dim_hidden` (int): 隐藏层维度
- `dim_output` (int): 输出维度
- `depth` (int): 网络深度
- `negative_slope` (float): LeakyReLU 负斜率（< 0 时使用 ELU）
- `batch_norm` (bool): 是否使用批归一化
- `spectral_norm` (float): 谱归一化系数
- `dropout_rate` (float): Dropout 率
- `weight_decay` (float): 权重衰减
- `learning_rate` (float): 学习率
- `batch_size` (int): 批次大小
- `epochs` (int): 训练轮数
- `patience` (int): 早停耐心值
- `num_workers` (int): 数据加载工作进程数
- `seed` (int): 随机种子

## 架构组成

### 1. 编码器（Encoder）

根据输入类型选择编码器：

#### 图像输入（`isinstance(dim_input, list)`）
```python
encoder = convolution.ResNet(
    dim_input=dim_input,
    layers=[2] * depth,
    base_width=dim_hidden // 8,
    # ...
)
```

#### 其他输入
```python
encoder = nn.Sequential(
    dense.NeuralNetwork(
        architecture=architecture,
        dim_input=dim_input,
        dim_hidden=dim_hidden,
        depth=depth,
        # ...
    ),
    dense.Activation(...),
)
```

### 2. 变分高斯过程（Variational GP）

```python
gp = gaussian_process.VariationalGP(
    num_outputs=dim_output,
    initial_lengthscale=initial_lengthscale,
    initial_inducing_points=initial_inducing_points,
    kernel=kernel,
    # ...
)
```

### 3. Deep Kernel GP 网络

```python
network = gaussian_process.DeepKernelGP(
    encoder=encoder,
    gp=gp,
)
```

### 4. 似然函数和优化器

```python
likelihood = likelihoods.GaussianLikelihood()

optimizer = optim.Adam(
    params=[
        {"params": encoder.parameters(), "lr": learning_rate},
        {"params": gp.parameters(), "lr": 2 * learning_rate},
        {"params": likelihood.parameters(), "lr": 2 * learning_rate},
    ],
    weight_decay=weight_decay,
)
```

## 核心方法

### `train_step(engine, batch)`

训练步骤实现。

**流程：**
1. 设置网络和似然函数为训练模式
2. 预处理批次数据
3. 前向传播
4. 计算变分 ELBO 损失
5. 反向传播和参数更新

**代码：**
```python
def train_step(self, engine, batch):
    self.network.train()
    self.likelihood.train()
    inputs, targets = self.preprocess(batch)
    self.optimizer.zero_grad()
    outputs = self.network(inputs)
    loss = -self.loss(outputs, targets.squeeze()).mean()
    loss.backward()
    self.optimizer.step()
    return {"outputs": outputs, "targets": targets}
```

### `tune_step(engine, batch)`

验证步骤实现。

**流程：**
1. 设置网络和似然函数为评估模式
2. 预处理批次数据
3. 前向传播（无梯度）
4. 返回输出和目标

### `predict_mus(ds, batch_size=None)`

预测 μ₀ 和 μ₁（对照组和处理组的均值）。

**参数：**
- `ds` (Dataset): 待预测的数据集
- `batch_size` (int, 可选): 批次大小（默认 `2 * self.batch_size`）

**返回：** tuple - `(mu_0, mu_1)`

**流程：**
1. 创建数据加载器
2. 对每个批次：
   - 复制协变量（一次用于 t=0，一次用于 t=1）
   - 创建对应的处理变量
   - 预测后验分布
   - 采样 1000 次
   - 分离 μ₀ 和 μ₁
3. 拼接所有批次的结果

**返回值格式：**
- `mu_0`: shape `(1000, n_samples)` - 1000 个模型样本的预测
- `mu_1`: shape `(1000, n_samples)` - 1000 个模型样本的预测

## 损失函数

使用变分下界（Variational ELBO）作为损失函数：

```python
loss = mlls.VariationalELBO(
    likelihood=self.likelihood,
    model=self.network.gp,
    num_data=len(inducing_point_dataset),
)
```

**特点：**
- 平衡数据拟合和正则化
- 考虑变分分布的不确定性
- 适用于大规模数据

## 评估指标

使用负对数似然作为评估指标：

```python
metrics = {
    "loss": metrics.Average(
        output_transform=lambda x: -self.likelihood.expected_log_prob(
            x["targets"].squeeze(), x["outputs"]
        ).mean(),
        device=self.device,
    )
}
```

## 使用示例

### 基本使用
```python
from causal_bald.library.models import DeepKernelGP
from causal_bald.library.datasets import IHDP

# 加载数据集
ds_train = IHDP(root="assets/", split="train", mode="mu", seed=0)
ds_valid = IHDP(root="assets/", split="valid", mode="mu", seed=0)

# 创建模型
model = DeepKernelGP(
    job_dir="checkpoints/",
    kernel="Matern32",
    num_inducing_points=100,
    inducing_point_dataset=ds_train,
    architecture="resnet",
    dim_input=25,
    dim_hidden=200,
    dim_output=1,
    depth=3,
    negative_slope=-1.0,
    batch_norm=True,
    spectral_norm=0.95,
    dropout_rate=0.1,
    weight_decay=0.0001,
    learning_rate=0.001,
    batch_size=100,
    epochs=500,
    patience=5,
    num_workers=0,
    seed=0,
)

# 训练
model.fit(ds_train, ds_valid)

# 预测
mu_0, mu_1 = model.predict_mus(ds_test)
tau_pred = mu_1.mean(0) - mu_0.mean(0)  # 取均值
```

### 图像数据
```python
# 使用 ResNet 编码器处理图像
model = DeepKernelGP(
    job_dir="checkpoints/",
    kernel="Matern32",
    num_inducing_points=100,
    inducing_point_dataset=ds_train,
    architecture="resnet",
    dim_input=[3, 28, 28],  # 图像维度
    dim_hidden=200,
    dim_output=1,
    depth=3,
    # ...
)
```

## 参数选择指南

### 核函数选择

- **RBF**：平滑函数，无限可微
- **Matern12**：一次可微
- **Matern32**：一次可微，更平滑（推荐）
- **Matern52**：两次可微

**推荐：** `Matern32`（平衡平滑度和灵活性）

### 诱导点数量

- **小数据集**（< 1000）：`num_inducing_points = 50-100`
- **中等数据集**（1000-10000）：`num_inducing_points = 100-200`
- **大数据集**（> 10000）：`num_inducing_points = 200-500`

**权衡：** 更多诱导点 → 更准确但计算更慢

### 学习率设置

- **编码器**：`learning_rate`（标准学习率）
- **GP 参数**：`2 * learning_rate`（更快收敛）
- **似然参数**：`2 * learning_rate`

**推荐：** `learning_rate = 0.001`

## 不确定性估计

模型通过采样提供不确定性估计：

```python
mu_0, mu_1 = model.predict_mus(ds_test)

# 均值预测
mu_0_mean = mu_0.mean(0)
mu_1_mean = mu_1.mean(0)

# 不确定性（标准差）
mu_0_std = mu_0.std(0)
mu_1_std = mu_1.std(0)

# 治疗效应不确定性
tau_pred = mu_1_mean - mu_0_mean
tau_std = np.sqrt(mu_0_std**2 + mu_1_std**2)
```

## 优势与局限

### 优势

1. **不确定性量化**：提供预测的不确定性估计
2. **特征学习**：深度编码器自动学习特征
3. **小数据友好**：变分推断适用于中等规模数据
4. **灵活性**：支持多种核函数和架构

### 局限

1. **计算开销**：比纯神经网络慢
2. **内存使用**：诱导点占用内存
3. **超参数敏感**：需要仔细调整超参数

## 注意事项

1. **诱导点初始化**：使用 `initial_values_for_GP` 进行良好初始化
2. **设备一致性**：确保编码器和 GP 在同一设备上
3. **批次大小**：预测时使用较大的批次大小以提高效率
4. **采样数量**：`predict_mus` 固定采样 1000 次，可根据需要调整

## 相关文档

- [模型概览](overview.md)
- [基础模型类](core.md)
- [高斯过程模块](../modules/gaussian_process.md)
- [全连接层模块](../modules/dense.md)
- [卷积模块](../modules/convolution.md)

