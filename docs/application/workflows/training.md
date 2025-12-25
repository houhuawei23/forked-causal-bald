# 训练工作流

## 概述

`causal_bald.application.workflows.training` 模块实现了标准的模型训练流程，用于在完整数据集上训练模型。

## 核心函数

### `trainer(config, experiment_dir, trial, model_name)`

执行模型训练。

**参数：**
- `config` (dict): 配置字典，包含数据集和模型参数
- `experiment_dir` (Path): 实验根目录
- `trial` (int): 试验编号
- `model_name` (str): 模型名称

**返回：** int（固定返回 -1）

**功能：**
1. 设置数据集随机种子
2. 加载训练和验证数据集
3. 创建实验目录
4. 保存配置
5. 调用训练函数训练模型

## 训练流程

### 流程步骤

```python
# 1. 设置随机种子
config["ds_train"]["seed"] = trial
config["ds_valid"]["seed"] = trial + 1 if dataset_name == "synthetic" else trial
config["ds_test"]["seed"] = trial + 2 if dataset_name == "synthetic" else trial

# 2. 加载数据集
ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))

# 3. 创建实验目录
experiment_dir = utils.DIRECTORIES[model_name](base_dir=experiment_dir, config=config)
experiment_dir = experiment_dir / f"trial-{trial:03d}"
experiment_dir.mkdir(parents=True, exist_ok=True)

# 4. 保存配置
config_path = experiment_dir / "config.json"
with config_path.open(mode="w") as cp:
    json.dump(config, cp)

# 5. 训练模型
out_dir = experiment_dir / "checkpoints"
utils.TRAIN_FUNCTIONS[model_name](
    ds_train=ds_train,
    ds_valid=ds_valid,
    job_dir=out_dir,
    config=config,
    dim_input=ds_train.dim_input,
)
```

## 配置参数

### 数据集配置
```python
config = {
    "dataset_name": "ihdp",
    "ds_train": {
        "root": "assets/",
        "split": "train",
        "mode": "mu",
        "seed": 0,
    },
    "ds_valid": {
        "root": "assets/",
        "split": "valid",
        "mode": "mu",
        "seed": 0,
    },
}
```

### 模型配置
根据模型类型不同，配置参数也不同：

#### Deep Kernel GP
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

#### Ensemble
```python
config = {
    "model_name": "ensemble",
    "dim_hidden": 400,
    "dim_output": 2,
    "depth": 3,
    "ensemble_size": 5,
    "learning_rate": 0.001,
    "batch_size": 200,
    "epochs": 500,
    # ... 更多参数
}
```

## 目录结构

训练会创建以下目录结构：

```
experiment_dir/
    {model_name}/
        {model_config}/
            trial-{trial:03d}/
                config.json                    # 配置保存
                checkpoints/
                    best_checkpoint.pt         # 最佳模型检查点
                    events.out.tfevents.*      # TensorBoard 日志
```

对于 Ensemble 模型：
```
experiment_dir/
    ensemble/
        {model_config}/
            trial-{trial:03d}/
                config.json
                checkpoints/
                    model-0/
                        best_checkpoint.pt
                    model-1/
                        best_checkpoint.pt
                    ...
                    model-{ensemble_size-1}/
                        best_checkpoint.pt
```

## 训练函数

训练函数通过 `utils.TRAIN_FUNCTIONS` 字典调用：

### `train_deep_kernel_gp(ds_train, ds_valid, job_dir, config, dim_input)`

训练 Deep Kernel GP 模型。

**参数：**
- `ds_train` (Dataset): 训练数据集
- `ds_valid` (Dataset): 验证数据集
- `job_dir` (Path): 工作目录
- `config` (dict): 配置字典
- `dim_input` (int): 输入维度

**功能：**
1. 检查检查点是否存在
2. 从配置中提取模型参数
3. 初始化 DeepKernelGP 模型
4. 调用 `model.fit()` 训练模型

### `train_ensemble(ds_train, ds_valid, job_dir, config, dim_input)`

训练集成模型。

**参数：** 同上

**功能：**
1. 从配置中提取集成大小和模型参数
2. 循环训练多个 TARNet 模型
3. 每个模型保存到独立的子目录

## 使用示例

### 基本使用
```python
from causal_bald.application.workflows import training
from pathlib import Path

config = {
    "dataset_name": "ihdp",
    "ds_train": {"root": "assets/", "split": "train", "mode": "mu", "seed": 0},
    "ds_valid": {"root": "assets/", "split": "valid", "mode": "mu", "seed": 0},
    "model_name": "deep_kernel_gp",
    "kernel": "Matern32",
    "num_inducing_points": 100,
    "dim_hidden": 200,
    "learning_rate": 0.001,
    "batch_size": 100,
    "epochs": 500,
    # ... 更多参数
}

training.trainer(
    config=config,
    experiment_dir=Path("experiments/training/"),
    trial=0,
    model_name="deep_kernel_gp",
)
```

### CLI 使用
```bash
causal-bald \
    train \
        --job-dir experiments/ \
        --num-trials 5 \
    ihdp --root assets/ \
    deep-kernel-gp \
        --kernel Matern32 \
        --epochs 500
```

## 模型训练细节

### Deep Kernel GP 训练

1. **模型初始化**：
   - 根据输入类型选择编码器（卷积或全连接）
   - 初始化变分高斯过程
   - 设置优化器和损失函数

2. **训练循环**：
   - 使用 VariationalELBO 作为损失函数
   - 编码器和 GP 使用不同的学习率
   - 支持早停机制

3. **检查点保存**：
   - 在验证损失改善时保存最佳模型
   - 保存模型状态、优化器状态和引擎状态

### Ensemble 训练

1. **独立训练**：
   - 每个模型独立训练
   - 使用不同的随机初始化
   - 可以并行训练（通过 Ray）

2. **模型保存**：
   - 每个模型保存到独立目录
   - 文件名格式：`model-{ensemble_id}/best_checkpoint.pt`

## 随机种子处理

- **合成数据集**：训练、验证、测试使用不同的种子（trial, trial+1, trial+2）
- **其他数据集**：训练和验证使用相同种子（trial）

## 注意事项

1. **检查点**：如果检查点已存在，训练会跳过
2. **早停**：模型支持早停机制，通过 `patience` 参数控制
3. **TensorBoard**：训练过程会记录到 TensorBoard 日志
4. **资源配置**：通过 Ray 可以配置 GPU/CPU 资源

## 相关文档

- [工作流概览](overview.md)
- [Deep Kernel GP 模型](../../library/models/deep_kernel.md)
- [TARNet 模型](../../library/models/tarnet.md)
- [工具函数](utils.md)

