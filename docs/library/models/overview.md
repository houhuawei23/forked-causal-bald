# 模型概览

## 概述

`causal_bald.library.models` 模块实现了用于因果效应估计的深度学习模型。所有模型都继承自 `core.BaseModel` 或 `core.PyTorchModel`，提供统一的接口。

## 模型架构

### 1. Deep Kernel GP (`deep_kernel.py`)

结合深度学习和高斯过程的模型，能够量化预测不确定性。

**特点：**
- 深度编码器提取特征
- 变分高斯过程建模不确定性
- 支持多种核函数（RBF, Matern12, Matern32, Matern52）
- 通过采样提供不确定性估计

**相关文档：** [Deep Kernel GP 文档](deep_kernel.md)

### 2. TARNet (`tarnet.py`)

Treatment-Agnostic Representation Network，使用共享特征提取器和独立的处理组/对照组头。

**特点：**
- 共享特征表示
- 独立的处理效应预测头
- 支持集成多个模型
- 快速训练和预测

**相关文档：** [TARNet 文档](tarnet.md)

### 3. Neural Network (`neural_network.py`)

标准神经网络模型，用于倾向性评分预测等任务。

**特点：**
- 灵活的架构（ResNet 或线性）
- 支持卷积和全连接层
- 变分输出层
- 用于辅助任务（如倾向性评分）

**相关文档：** [Neural Network 文档](neural_network.md)

## 基础类

### `BaseModel` (`core.py`)

所有模型的抽象基类。

**方法：**
- `fit(train_dataset, tune_dataset)`: 训练模型
- `save()`: 保存模型
- `load()`: 加载模型

### `PyTorchModel` (`core.py`)

基于 PyTorch 的模型基类，提供训练基础设施。

**属性：**
- `network`: 神经网络
- `optimizer`: 优化器
- `metrics`: 评估指标
- `device`: 计算设备

**方法：**
- `train_step(engine, batch)`: 训练步骤（需子类实现）
- `tune_step(engine, batch)`: 验证步骤（需子类实现）
- `fit(train_dataset, tune_dataset)`: 训练循环
- `save()`: 保存检查点
- `load()`: 加载检查点

**功能：**
- 自动设备管理
- TensorBoard 日志记录
- 早停机制
- 检查点管理

## 统一接口

所有模型都提供以下接口：

### 训练
```python
model.fit(ds_train, ds_valid)
```

### 预测
```python
# Deep Kernel GP 和 TARNet
mu_0, mu_1 = model.predict_mus(ds_test)

# Neural Network
mean = model.predict_mean(ds_test)
```

### 保存和加载
```python
model.save()
model.load()
```

## 模型选择指南

### 使用 Deep Kernel GP 当：
- 需要不确定性量化
- 数据量中等（< 10K）
- 需要贝叶斯推断
- 计算资源充足

### 使用 TARNet 当：
- 需要快速训练和预测
- 数据量较大
- 需要集成多个模型
- 计算资源有限

### 使用 Neural Network 当：
- 用于辅助任务（如倾向性评分）
- 不需要不确定性估计
- 简单的回归或分类任务

## 模型配置

### Deep Kernel GP 配置
```python
config = {
    "kernel": "Matern32",
    "num_inducing_points": 100,
    "dim_hidden": 200,
    "dim_output": 1,
    "depth": 3,
    "learning_rate": 0.001,
    "batch_size": 100,
    "epochs": 500,
}
```

### TARNet 配置
```python
config = {
    "dim_hidden": 400,
    "dim_output": 2,
    "depth": 3,
    "learning_rate": 0.001,
    "batch_size": 200,
    "epochs": 500,
    "ensemble_size": 5,  # 用于集成
}
```

## 相关文档

- [基础模型类](core.md)
- [Deep Kernel GP](deep_kernel.md)
- [TARNet](tarnet.md)
- [Neural Network](neural_network.md)

