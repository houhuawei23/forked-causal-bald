# Neural Network 模型文档

## 概述

`causal_bald.library.models.neural_network` 模块实现了标准神经网络模型，主要用于辅助任务（如倾向性评分预测）。

## 核心思想

Neural Network 模型是一个简单的神经网络，用于：
- **倾向性评分预测**：预测 P(T=1|X=x)
- **辅助任务**：其他分类或回归任务
- **快速原型**：快速测试和验证

## 类定义

### `NeuralNetwork`

神经网络模型类，继承自 `core.PyTorchModel`。

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
encoder = dense.NeuralNetwork(
    architecture=architecture,
    dim_input=dim_input,
    dim_hidden=dim_hidden,
    depth=depth,
    # ...
)
```

### 输出层

使用分类变分分布作为输出层：

```python
self.network = nn.Sequential(
    encoder,
    variational.Categorical(
        dim_input=encoder.dim_output,
        dim_output=dim_output,
    ),
)
```

**输出类型：**
- `dim_output=1`：Bernoulli 分布（二分类）
- `dim_output>1`：Categorical 分布（多分类）

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

### `predict_mean(ds, batch_size=None)`

预测均值。

**参数：**
- `ds` (Dataset): 待预测的数据集
- `batch_size` (int, 可选): 批次大小（默认 `2 * self.batch_size`）

**返回：** np.ndarray - 均值预测，shape `(n_samples, dim_output)`

**流程：**
1. 创建数据加载器
2. 对每个批次：
   - 预处理数据
   - 预测分布
   - 提取均值
3. 拼接所有批次的结果

## 损失函数

使用负对数似然作为损失函数：

```python
loss = -outputs.log_prob(targets).mean()
```

其中 `outputs` 是分类分布（Bernoulli 或 Categorical），`targets` 是标签。

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

### 倾向性评分预测
```python
from causal_bald.library.models import NeuralNetwork
from causal_bald.library.datasets import IHDP

# 加载数据集（倾向性评分模式）
ds_train = IHDP(root="assets/", split="train", mode="pi", seed=0)
ds_valid = IHDP(root="assets/", split="valid", mode="pi", seed=0)

# 创建模型
model = NeuralNetwork(
    job_dir="checkpoints/pi/",
    architecture="resnet",
    dim_input=25,
    dim_hidden=200,
    dim_output=1,  # 二分类（Bernoulli）
    depth=3,
    negative_slope=-1.0,
    batch_norm=False,  # 倾向性评分通常不使用批归一化
    spectral_norm=0.95,
    dropout_rate=0.1,
    weight_decay=0.0001,
    learning_rate=0.001,
    batch_size=200,
    epochs=500,
    patience=10,
    num_workers=0,
    seed=0,
)

# 训练
model.fit(ds_train, ds_valid)

# 预测倾向性评分
pi_pred = model.predict_mean(ds_test)
```

### 多分类任务
```python
# 10 分类任务
model = NeuralNetwork(
    job_dir="checkpoints/",
    architecture="resnet",
    dim_input=784,
    dim_hidden=200,
    dim_output=10,  # 10 分类
    depth=3,
    # ...
)
```

### 图像分类
```python
# 图像输入
model = NeuralNetwork(
    job_dir="checkpoints/",
    architecture="resnet",
    dim_input=[3, 28, 28],  # 图像维度
    dim_hidden=200,
    dim_output=10,  # 10 分类
    depth=3,
    # ...
)
```

## 在主动学习中使用

Neural Network 模型主要用于训练倾向性评分模型：

```python
from causal_bald.application.workflows.active_learning import get_propensities

# 在主动学习循环中获取倾向性评分
pt = get_propensities(trial_dir=trial_dir, config=config)

# get_propensities 内部会：
# 1. 创建 NeuralNetwork 模型
# 2. 训练模型（如果检查点不存在）
# 3. 预测倾向性评分
# 4. 返回预测结果
```

## 参数选择指南

### 架构选择

- **basic**：简单快速，适用于简单任务
- **resnet**：更强大，适用于复杂任务（推荐）

### 隐藏层维度

- **小任务**：`dim_hidden = 100-200`
- **中等任务**：`dim_hidden = 200-400`
- **复杂任务**：`dim_hidden = 400-800`

### 批归一化

- **倾向性评分**：通常不使用批归一化（`batch_norm=False`）
- **其他任务**：可以使用批归一化（`batch_norm=True`）

## 优势与局限

### 优势

1. **简单快速**：训练和预测都很快
2. **灵活**：支持多种输入类型和任务
3. **易于使用**：接口简单直观

### 局限

1. **不确定性有限**：只提供点估计
2. **表达能力**：不如专门的因果推断模型

## 与其他模型对比

| 特性 | Neural Network | TARNet | Deep Kernel GP |
|------|----------------|--------|----------------|
| 用途 | 辅助任务 | 因果推断 | 因果推断 |
| 不确定性 | 无 | 有限 | 丰富 |
| 速度 | 最快 | 快 | 慢 |
| 复杂度 | 低 | 中 | 高 |

## 注意事项

1. **模式选择**：确保数据集使用正确的模式（`mode="pi"` 用于倾向性评分）
2. **输出维度**：`dim_output=1` 用于二分类，`dim_output>1` 用于多分类
3. **权重衰减**：倾向性评分模型通常使用较小的权重衰减
4. **检查点**：如果检查点已存在，训练会跳过

## 相关文档

- [模型概览](overview.md)
- [基础模型类](core.md)
- [全连接层模块](../modules/dense.md)
- [卷积模块](../modules/convolution.md)
- [变分推断模块](../modules/variational.md)
- [主动学习工作流](../../application/workflows/active_learning.md)

