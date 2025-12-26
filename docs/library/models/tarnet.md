# TARNet 模型文档

## 概述

`causal_bald.library.models.tarnet` 模块实现了 TARNet（Treatment-Agnostic Representation Network）模型，用于因果效应估计。

## 核心思想

TARNet 使用共享的特征提取器和独立的处理组/对照组头来估计因果效应：
- **共享编码器**：学习与处理无关的特征表示
- **独立头**：分别为处理组和对照组建模结果分布
- **变分输出**：使用变分分布提供不确定性估计

## 类定义

### `TARNet`

TARNet 模型类，继承自 `core.PyTorchModel`。

**初始化参数：**
- `job_dir` (str): 工作目录，用于保存检查点
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

### 网络结构

模型使用 `tarnet.TARNet` 模块构建网络：

```python
self.network = tarnet.TARNet(
    architecture=architecture,
    dim_input=dim_input,
    dim_hidden=dim_hidden,
    dim_output=dim_output,
    depth=depth,
    negative_slope=negative_slope,
    batch_norm=batch_norm,
    dropout_rate=dropout_rate,
    spectral_norm=spectral_norm,
)
```

**结构详情：**
- **编码器**：共享特征提取器（根据输入类型选择）
- **t0_encoder**：对照组编码器
- **t1_encoder**：处理组编码器
- **outcome_density**：结果分布（SplitGMM 或 GMM）

## 核心方法

### `train_step(engine, batch)`

训练步骤实现。

**流程：**
1. 设置网络为训练模式
2. 预处理批次数据
3. 前向传播得到分布
4. 计算负对数似然损失
5. 反向传播和参数更新

**代码：**
```python
def train_step(self, engine, batch):
    self.network.train()
    inputs, targets = self.preprocess(batch)
    self.optimizer.zero_grad()
    outputs = self.network(inputs)
    loss = -outputs.log_prob(targets).mean()
    loss.backward()
    self.optimizer.step()
    return {"outputs": outputs, "targets": targets}
```

### `tune_step(engine, batch)`

验证步骤实现。

**流程：**
1. 设置网络为评估模式
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
   - 提取均值
   - 分离 μ₀ 和 μ₁
3. 拼接所有批次的结果

**返回值格式：**
- `mu_0`: shape `(n_samples,)` - 均值预测
- `mu_1`: shape `(n_samples,)` - 均值预测

## 损失函数

使用负对数似然作为损失函数：

```python
loss = -outputs.log_prob(targets).mean()
```

其中 `outputs` 是变分分布（SplitGMM 或 GMM），`targets` 是观测结果。

## 评估指标

使用平均负对数似然作为评估指标：

```python
metrics = {
    "loss": metrics.Average(
        output_transform=lambda x: -x["outputs"].log_prob(x["targets"]).mean(),
        device=self.device,
    )
}
```

## 使用示例

### 基本使用
```python
from causal_bald.library.models import TARNet
from causal_bald.library.datasets import IHDP

# 加载数据集
ds_train = IHDP(root="assets/", split="train", mode="mu", seed=0)
ds_valid = IHDP(root="assets/", split="valid", mode="mu", seed=0)

# 创建模型
model = TARNet(
    job_dir="checkpoints/",
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
    batch_size=200,
    epochs=500,
    patience=20,
    num_workers=0,
    seed=0,
)

# 训练
model.fit(ds_train, ds_valid)

# 预测
mu_0, mu_1 = model.predict_mus(ds_test)
tau_pred = mu_1 - mu_0  # 治疗效应
```

### 图像数据
```python
# 使用 ResNet 编码器处理图像
model = TARNet(
    job_dir="checkpoints/",
    architecture="resnet",
    dim_input=[3, 28, 28],  # 图像维度
    dim_hidden=200,
    dim_output=1,
    depth=3,
    # ...
)
```

### 集成模型

TARNet 常用于构建集成模型：

```python
ensemble_size = 5
models = []

for i in range(ensemble_size):
    model = TARNet(
        job_dir=f"checkpoints/model-{i}/",
        # ... 参数
    )
    model.fit(ds_train, ds_valid)
    models.append(model)

# 集成预测
mu_0_list = []
mu_1_list = []
for model in models:
    mu_0, mu_1 = model.predict_mus(ds_test)
    mu_0_list.append(mu_0)
    mu_1_list.append(mu_1)

# 平均预测
mu_0_ensemble = np.mean(mu_0_list, axis=0)
mu_1_ensemble = np.mean(mu_1_list, axis=0)
tau_ensemble = mu_1_ensemble - mu_0_ensemble
```

## 参数选择指南

### 架构选择

- **basic**：简单预激活层，适用于浅层网络
- **resnet**：残差连接，适用于深层网络（推荐）

### 隐藏层维度

- **小数据集**（< 1000）：`dim_hidden = 100-200`
- **中等数据集**（1000-10000）：`dim_hidden = 200-400`
- **大数据集**（> 10000）：`dim_hidden = 400-800`

### 深度选择

- **简单任务**：`depth = 2-3`
- **复杂任务**：`depth = 3-5`
- **图像任务**：`depth = 3-4`

**推荐：** `depth = 3`

## 优势与局限

### 优势

1. **快速训练**：比 Deep Kernel GP 快得多
2. **灵活建模**：独立的头允许不同的处理效应
3. **集成支持**：易于构建集成模型
4. **不确定性**：通过变分分布提供不确定性估计

### 局限

1. **不确定性有限**：不如 Deep Kernel GP 的不确定性估计丰富
2. **需要集成**：单个模型的不确定性可能不够可靠

## 与 Deep Kernel GP 对比

| 特性 | TARNet | Deep Kernel GP |
|------|--------|----------------|
| 训练速度 | 快 | 慢 |
| 不确定性 | 有限 | 丰富 |
| 计算开销 | 低 | 高 |
| 集成 | 容易 | 困难 |
| 适用数据量 | 大 | 中小 |

## 注意事项

1. **输入格式**：输入必须包含处理变量作为最后一维
2. **批次大小**：可以使用较大的批次大小（如 200）
3. **早停**：建议使用较大的 patience（如 20）
4. **集成**：对于重要应用，建议使用集成模型

## 相关文档

- [模型概览](overview.md)
- [基础模型类](core.md)
- [TARNet 模块](../modules/tarnet.md)
- [变分推断模块](../modules/variational.md)

