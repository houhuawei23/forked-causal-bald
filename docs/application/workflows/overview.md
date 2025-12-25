# 工作流概览

## 概述

`causal_bald.application.workflows` 模块包含了 Causal-BALD 项目的核心工作流实现。这些工作流提供了从数据加载、模型训练、主动学习到结果评估的完整流程。

## 工作流模块

### 1. 主动学习工作流 (`active_learning.py`)

实现主动学习循环，包括样本获取、模型训练和迭代更新。

**主要函数：**
- `active_learner()`: 执行主动学习主循环
- `get_propensities()`: 获取倾向性评分（如需要）

**相关文档：** [主动学习工作流](active_learning.md)

### 2. 训练工作流 (`training.py`)

实现标准模型训练流程。

**主要函数：**
- `trainer()`: 执行模型训练

**相关文档：** [训练工作流](training.md)

### 3. 评估工作流 (`evaluation.py`)

实现模型评估和结果可视化。

**主要函数：**
- `pehe()`: 计算 PEHE 指标
- `plot_convergence()`: 绘制收敛曲线
- `plot_evolution()`: 绘制演化过程
- `plot_distribution()`: 绘制分布图
- `plot_dataset()`: 绘制数据集可视化
- `plot_errorbars()`: 绘制误差条

**相关文档：** [评估工作流](evaluation.md)

### 4. 超参数调优工作流 (`tuning.py`)

实现基于 Ray Tune 的超参数优化。

**主要函数：**
- `deep_kernel_gp_tuner()`: Deep Kernel GP 超参数调优
- `tarnet_tuner()`: TARNet 超参数调优

**相关文档：** [超参数调优工作流](tuning.md)

### 5. 工具函数 (`utils.py`)

提供工作流中使用的工具函数。

**主要函数：**
- `train_deep_kernel_gp()`: 训练 Deep Kernel GP 模型
- `train_ensemble()`: 训练集成模型
- `predict_deep_kernel_gp()`: 使用 Deep Kernel GP 预测
- `predict_ensemble()`: 使用集成模型预测
- `directory_deep_kernel_gp()`: 生成 Deep Kernel GP 目录结构
- `directory_ensemble()`: 生成集成模型目录结构

**相关文档：** [工具函数](utils.md)

## 工作流关系

```
CLI (main.py)
    │
    ├─── active-learning ───> active_learning.active_learner()
    │                              │
    │                              ├──> utils.TRAIN_FUNCTIONS
    │                              └──> utils.PREDICT_FUNCTIONS
    │
    ├─── train ──────────────> training.trainer()
    │                              │
    │                              └──> utils.TRAIN_FUNCTIONS
    │
    ├─── tune ───────────────> tuning.*_tuner()
    │                              │
    │                              └──> utils.TRAIN_FUNCTIONS
    │
    └─── evaluate ───────────> evaluation.*()
                                      │
                                      └──> utils.PREDICT_FUNCTIONS
```

## 配置字典结构

所有工作流函数都接收一个 `config` 字典参数，包含以下字段：

### 数据集配置
```python
config = {
    "dataset_name": "ihdp",  # 数据集名称
    "ds_train": {
        "root": "assets/",
        "split": "train",
        "mode": "mu",
        "seed": 1331,
    },
    "ds_valid": {...},
    "ds_test": {...},
}
```

### 主动学习配置
```python
config = {
    "step_size": 10,              # 每次获取的样本数
    "warm_start_size": 100,       # 初始样本数
    "max_acquisitions": 38,       # 最大获取步数
    "acquisition_function": "mu-rho",  # 获取函数
    "temperature": 0.25,          # 温度参数
    "use_gumbel": True,           # 是否使用 Gumbel 采样
}
```

### 模型配置
```python
config = {
    "model_name": "deep_kernel_gp",
    "kernel": "Matern32",
    "num_inducing_points": 100,
    "dim_hidden": 200,
    "dim_output": 1,
    "depth": 3,
    "learning_rate": 0.001,
    "batch_size": 100,
    "epochs": 500,
    # ... 更多参数
}
```

### 资源配置
```python
config = {
    "gpu_per_trial": 0.2,
    "cpu_per_trial": 1.0,
    "num_trials": 5,
    "seed": 1331,
}
```

## 目录结构约定

工作流会根据配置自动生成结构化的目录路径：

### 主动学习目录
```
experiments/active_learning/
    ss-{step_size}_ws-{warm_start_size}_ma-{max_acquisitions}_af-{acquisition_function}_temp-{temperature}_gumb-{use_gumbel}/
        {dataset_name}/
            {model_name}/
                {model_config}/
                    trial-{trial:03d}/
                        config.json
                        acquisition-{i:03d}/
                            aquired.json
                            best_checkpoint.pt
```

### 训练目录
```
experiments/training/
    {dataset_name}/
        {model_name}/
            {model_config}/
                trial-{trial:03d}/
                    config.json
                    checkpoints/
                        best_checkpoint.pt
```

## 数据流

### 主动学习数据流
```
数据集初始化
    ↓
创建 ActiveLearningDataset
    ↓
循环 (max_acquisitions 次):
    ├── 计算获取分数
    ├── 选择样本
    ├── 更新训练集
    ├── 训练模型
    └── 保存检查点
```

### 训练数据流
```
数据集加载
    ↓
模型初始化
    ↓
训练循环:
    ├── 前向传播
    ├── 计算损失
    ├── 反向传播
    ├── 更新参数
    └── 验证评估
    ↓
保存最佳模型
```

### 评估数据流
```
加载实验目录
    ↓
遍历试验:
    ├── 加载配置
    ├── 加载模型
    ├── 预测结果
    └── 计算指标
    ↓
聚合结果
    ↓
生成可视化
```

## 扩展工作流

要添加新的工作流：

1. 在 `workflows/` 目录下创建新模块
2. 实现工作流函数
3. 在 `utils.py` 中添加必要的工具函数
4. 在 `main.py` 中添加对应的 CLI 命令
5. 更新相关文档

## 相关文档

- [CLI 接口文档](../main.md)
- [主动学习工作流](active_learning.md)
- [训练工作流](training.md)
- [评估工作流](evaluation.md)
- [超参数调优工作流](tuning.md)
- [工具函数](utils.md)

