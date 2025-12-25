# 基础模型类

## 概述

`causal_bald.library.models.core` 模块定义了所有模型的基础类，提供了统一的接口和通用的训练基础设施。

## 类定义

### `BaseModel`

所有模型的抽象基类。

```python
class BaseModel(ABC):
    def __init__(self, job_dir, seed):
        self.job_dir = job_dir
        self.seed = seed
    
    @abstractmethod
    def fit(self, train_dataset, tune_dataset):
        """训练模型"""
        pass
    
    @abstractmethod
    def save(self):
        """保存模型"""
        pass
    
    @abstractmethod
    def load(self):
        """加载模型"""
        pass
```

### `PyTorchModel`

基于 PyTorch 的模型基类，继承自 `BaseModel`。

**初始化参数：**
- `job_dir` (str): 工作目录，用于保存检查点
- `learning_rate` (float): 学习率
- `batch_size` (int): 批次大小
- `epochs` (int): 训练轮数
- `num_workers` (int): 数据加载工作进程数
- `seed` (int): 随机种子

**属性：**
- `network`: 神经网络（通过 property 访问）
- `optimizer`: 优化器（通过 property 访问）
- `metrics`: 评估指标字典（通过 property 访问）
- `device`: 计算设备（CPU 或 GPU）
- `best_state`: 最佳模型状态
- `best_loss`: 最佳验证损失
- `counter`: 早停计数器

## 核心方法

### `fit(train_dataset, tune_dataset)`

执行模型训练。

**参数：**
- `train_dataset` (Dataset): 训练数据集
- `tune_dataset` (Dataset): 验证数据集

**返回：** dict - 最终评估指标

**流程：**
1. 创建数据加载器
2. 设置训练和验证引擎
3. 注册事件处理器
4. 运行训练循环

**数据加载器配置：**
- 训练集使用 `RandomFixedLengthSampler`（固定长度采样）
- 验证集使用标准采样
- 支持多进程数据加载

### `train_step(engine, batch)`

训练步骤，需子类实现。

**参数：**
- `engine`: PyTorch Ignite 引擎
- `batch`: 批次数据

**返回：** dict - 包含 `outputs` 和 `targets` 的字典

### `tune_step(engine, batch)`

验证步骤，需子类实现。

**参数：** 同上

**返回：** dict - 包含 `outputs` 和 `targets` 的字典

### `preprocess(batch)`

预处理批次数据，将数据移动到正确的设备。

**参数：**
- `batch`: 批次数据，格式为 `(inputs, targets)`

**返回：** tuple - `(inputs, targets)`，已移动到设备

**功能：**
- 自动检测输入和目标的数据类型
- 支持列表和单个张量
- 移动到指定设备（CPU/GPU）

### `save()`

保存模型检查点。

**保存内容：**
- 模型状态字典
- 优化器状态字典
- 训练引擎状态
- 似然函数状态（如适用）

**保存位置：** `{job_dir}/best_checkpoint.pt`

### `load()`

加载模型检查点。

**功能：**
- 支持 Ray Tune 和标准检查点
- 恢复模型、优化器和引擎状态
- 如果检查点不存在，记录日志但不报错

### `update()`

更新最佳模型状态。

**功能：**
- 深拷贝当前最佳状态
- 仅在非 Ray Tune 模式下执行
- 保存模型、优化器和似然函数状态

## 事件处理

### `on_epoch_completed(engine, train_loader, tune_loader)`

每个 epoch 完成时调用。

**功能：**
1. 打印训练指标
2. 运行验证评估
3. 打印验证指标
4. 检查是否改善
5. 更新最佳状态或增加早停计数器
6. 如果达到早停条件，终止训练

### `on_training_completed(engine, loader)`

训练完成时调用。

**功能：**
1. 保存最佳模型
2. 加载最佳模型
3. 运行最终评估
4. 打印最佳指标

## 早停机制

通过 `patience` 参数控制早停：

```python
if tune_metrics["loss"] < self.best_loss:
    self.best_loss = tune_metrics["loss"]
    self.counter = 0
    self.update()
else:
    self.counter += 1
    if self.counter == self.patience:
        engine.terminate()  # 早停
```

## 使用示例

### 继承 PyTorchModel

```python
from causal_bald.library.models import core

class MyModel(core.PyTorchModel):
    def __init__(self, job_dir, learning_rate, batch_size, epochs, num_workers, seed):
        super().__init__(
            job_dir=job_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            num_workers=num_workers,
            seed=seed,
        )
        # 初始化网络、优化器等
        self.network = ...
        self.optimizer = ...
        self.metrics = {...}
    
    def train_step(self, engine, batch):
        self.network.train()
        inputs, targets = self.preprocess(batch)
        self.optimizer.zero_grad()
        outputs = self.network(inputs)
        loss = compute_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return {"outputs": outputs, "targets": targets}
    
    def tune_step(self, engine, batch):
        self.network.eval()
        inputs, targets = self.preprocess(batch)
        with torch.no_grad():
            outputs = self.network(inputs)
        return {"outputs": outputs, "targets": targets}
```

## 集成 PyTorch Ignite

`PyTorchModel` 使用 PyTorch Ignite 框架管理训练循环：

- **Engine**: 管理训练和验证循环
- **Metrics**: 自动计算和聚合指标
- **Events**: 事件驱动的训练流程
- **Handlers**: 自定义事件处理器

## 相关文档

- [模型概览](overview.md)
- [Deep Kernel GP](deep_kernel.md)
- [TARNet](tarnet.md)

