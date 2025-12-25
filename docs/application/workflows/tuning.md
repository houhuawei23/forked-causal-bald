# 超参数调优工作流

## 概述

`causal_bald.application.workflows.tuning` 模块实现了基于 Ray Tune 的超参数优化功能，支持 Deep Kernel GP 和 TARNet 模型的超参数搜索。

## 核心函数

### `deep_kernel_gp_tuner(config)`

Deep Kernel GP 模型的超参数调优器。

**参数：**
- `config` (dict): 配置字典，包含数据集和调优参数

**返回：** None

**功能：**
1. 定义超参数搜索空间
2. 设置优化算法（HyperOpt）
3. 设置调度器（AsyncHyperBandScheduler）
4. 运行超参数搜索
5. 输出最佳超参数

### `tarnet_tuner(config)`

TARNet 模型的超参数调优器。

**参数：**
- `config` (dict): 配置字典

**返回：** None

**功能：** 类似 `deep_kernel_gp_tuner`，但针对 TARNet 模型

## 超参数搜索空间

### Deep Kernel GP 搜索空间

```python
space = {
    "kernel": tune.choice(["RBF", "Matern12", "Matern32", "Matern52"]),
    "num_inducing_points": tune.choice([20, 50, 100, 200]),
    "dim_hidden": tune.choice([100, 200, 400]),
    "depth": tune.choice([2, 3, 4]),
    "negative_slope": tune.choice([-1.0, 0.0, 0.1, 0.2]),
    "dropout_rate": tune.choice([0.05, 0.1, 0.2, 0.5]),
    "spectral_norm": tune.choice([0.0, 0.95, 1.5, 3.0]),
    "learning_rate": tune.choice([2e-4, 5e-4, 1e-3]),
    "batch_size": tune.choice([32, 64, 100, 200]),
}
```

### TARNet 搜索空间

```python
space = {
    "dim_hidden": tune.choice([100, 200, 400]),
    "depth": tune.choice([2, 3, 4]),
    "negative_slope": tune.choice([-1.0, 0.0, 0.1, 0.2]),
    "dropout_rate": tune.choice([0.05, 0.1, 0.2, 0.5]),
    "spectral_norm": tune.choice([0.0, 0.95, 1.5, 3.0]),
    "learning_rate": tune.choice([2e-4, 5e-4, 1e-3]),
    "batch_size": tune.choice([32, 64, 100, 200]),
}
```

## 优化算法

### HyperOpt 搜索算法

```python
algorithm = hyperopt.HyperOptSearch(
    space,
    metric="mean_loss",
    mode="min",
    n_initial_points=100,
)
```

**特点：**
- 基于 Tree-structured Parzen Estimator (TPE)
- 支持离散和连续超参数
- 初始随机采样 100 个点

### AsyncHyperBandScheduler 调度器

#### Deep Kernel GP
```python
scheduler = schedulers.AsyncHyperBandScheduler(
    grace_period=100,      # 最小训练轮数
    max_t=config.get("epochs"),  # 最大训练轮数
)
```

#### TARNet
```python
scheduler = schedulers.AsyncHyperBandScheduler(
    grace_period=20,       # 更小的耐心值（训练更快）
    max_t=config.get("epochs"),
)
```

**特点：**
- 异步多臂老虎机算法
- 早停表现不佳的试验
- 资源高效利用

## 训练函数

### Deep Kernel GP 训练函数

```python
def func(config):
    dataset_name = config.get("dataset_name")
    ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))
    
    utils.TRAIN_FUNCTIONS["deep_kernel_gp"](
        ds_train=ds_train,
        ds_valid=ds_valid,
        job_dir=None,  # Ray Tune 管理检查点
        config=config,
        dim_input=ds_train.dim_input,
    )
```

### TARNet 训练函数

```python
def func(config):
    dataset_name = config.get("dataset_name")
    ds_train = datasets.DATASETS.get(dataset_name)(**config.get("ds_train"))
    ds_valid = datasets.DATASETS.get(dataset_name)(**config.get("ds_valid"))
    
    utils.TRAIN_FUNCTIONS["ensemble"](
        ds_train=ds_train,
        ds_valid=ds_valid,
        job_dir=None,
        config=config,
        dim_input=ds_train.dim_input,
    )
```

## Ray Tune 配置

### 运行配置

```python
analysis = tune.run(
    run_or_experiment=func,
    metric="mean_loss",
    mode="min",
    name="hyperopt_deep_kernel_gp",
    resources_per_trial={
        "cpu": config.get("cpu_per_trial"),
        "gpu": config.get("gpu_per_trial"),
    },
    num_samples=config.get("max_samples"),
    search_alg=algorithm,
    scheduler=scheduler,
    local_dir=config.get("experiment_dir"),
    config=config,
)
```

**参数说明：**
- `metric`: 优化指标（验证损失）
- `mode`: 优化方向（最小化）
- `num_samples`: 搜索空间采样数量
- `resources_per_trial`: 每个试验的资源分配

## 配置参数

### 必需参数
```python
config = {
    "dataset_name": "ihdp",
    "ds_train": {...},
    "ds_valid": {...},
    "experiment_dir": "experiments/tuning/",
    "max_samples": 200,
    "epochs": 500,
    "cpu_per_trial": 1.0,
    "gpu_per_trial": 0.2,
}
```

## 使用示例

### 基本使用
```python
from causal_bald.application.workflows import tuning

config = {
    "dataset_name": "ihdp",
    "ds_train": {"root": "assets/", "split": "train", "mode": "mu", "seed": 0},
    "ds_valid": {"root": "assets/", "split": "valid", "mode": "mu", "seed": 0},
    "experiment_dir": "experiments/tuning/",
    "max_samples": 200,
    "epochs": 500,
    "cpu_per_trial": 1.0,
    "gpu_per_trial": 0.2,
}

tuning.deep_kernel_gp_tuner(config)
```

### CLI 使用
```bash
# Deep Kernel GP 调优
causal-bald \
    tune \
        --job-dir experiments/ \
        --max-samples 200 \
        --gpu-per-trial 0.2 \
    ihdp --root assets/ \
    deep-kernel-gp

# TARNet 调优
causal-bald \
    tune \
        --job-dir experiments/ \
        --max-samples 200 \
    ihdp --root assets/ \
    ensemble
```

## 结果输出

### 最佳超参数

调优完成后会输出最佳超参数：

```python
print("Best hyperparameters found were: ", analysis.best_config)
```

### 结果目录结构

```
experiment_dir/
    hyperopt_{model_name}/
        {trial_id}/
            checkpoint.pt
            events.out.tfevents.*
        ...
```

### 访问结果

```python
# 最佳配置
best_config = analysis.best_config

# 最佳试验结果
best_trial = analysis.get_best_trial("mean_loss", "min", "last")

# 所有试验结果
df = analysis.results_df
```

## 优化策略

### 1. 搜索空间设计

- **离散参数**：使用 `tune.choice()` 定义候选值
- **连续参数**：可以使用 `tune.uniform()` 或 `tune.loguniform()`
- **平衡探索和利用**：初始随机采样 + TPE 优化

### 2. 早停策略

- **Grace Period**：给每个试验最小训练轮数
- **AsyncHyperBand**：自动停止表现不佳的试验
- **资源效率**：节省计算资源

### 3. 并行化

- **Ray Tune**：自动管理并行试验
- **资源分配**：通过 `resources_per_trial` 控制
- **分布式**：支持多机分布式调优

## 注意事项

1. **计算资源**：超参数调优需要大量计算资源
2. **时间成本**：可能需要数小时或数天
3. **搜索空间**：过大的搜索空间会导致调优困难
4. **验证集**：确保验证集足够大且代表性好
5. **随机性**：不同运行可能得到不同结果

## 最佳实践

1. **从小开始**：先用小搜索空间和少量样本测试
2. **逐步扩展**：根据初步结果调整搜索空间
3. **固定随机种子**：确保结果可重复
4. **监控资源**：注意 GPU/CPU 使用情况
5. **保存结果**：定期保存中间结果

## 相关文档

- [工作流概览](overview.md)
- [训练工作流](training.md)
- [Ray Tune 文档](https://docs.ray.io/en/latest/tune/index.html)
- [HyperOpt 文档](https://hyperopt.github.io/hyperopt/)

