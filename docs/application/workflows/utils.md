# 工作流工具函数

## 概述

`causal_bald.application.workflows.utils` 模块提供了工作流中使用的工具函数，包括模型训练、预测和目录管理功能。

## 目录管理函数

### `directory_deep_kernel_gp(base_dir, config)`

生成 Deep Kernel GP 模型的目录路径。

**参数：**
- `base_dir` (str/Path): 基础目录
- `config` (dict): 配置字典

**返回：** Path

**目录格式：**
```
{base_dir}/deep_kernel_gp/kernel-{kernel}_ip-{num_inducing_points}-dh-{dim_hidden}_do-{dim_output}_dp-{depth}_ns-{negative_slope}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}
```

**示例：**
```
experiments/deep_kernel_gp/kernel-Matern32_ip-100-dh-200_do-1_dp-3_ns--1.0_dr-0.1_sn-0.95_lr-0.001_bs-100_ep-500
```

### `directory_ensemble(base_dir, config)`

生成集成模型的目录路径。

**参数：** 同上

**返回：** Path

**目录格式：**
```
{base_dir}/ensemble/dh-{dim_hidden}_do-{dim_output}_dp-{depth}_ns-{negative_slope}_dr-{dropout_rate}_sn-{spectral_norm}_lr-{learning_rate}_bs-{batch_size}_ep-{epochs}
```

### `DIRECTORIES`

目录生成函数字典。

```python
DIRECTORIES = {
    "deep_kernel_gp": directory_deep_kernel_gp,
    "ensemble": directory_ensemble,
}
```

## 训练函数

### `train_deep_kernel_gp(ds_train, ds_valid, job_dir, config, dim_input)`

训练 Deep Kernel GP 模型。

**参数：**
- `ds_train` (Dataset): 训练数据集
- `ds_valid` (Dataset): 验证数据集
- `job_dir` (Path): 工作目录（检查点保存位置）
- `config` (dict): 配置字典
- `dim_input` (int): 输入维度

**返回：** None

**功能：**
1. 检查检查点是否存在，如果存在则跳过训练
2. 从配置中提取模型参数
3. 初始化 DeepKernelGP 模型
4. 调用 `model.fit()` 训练模型

**实现逻辑：**
```python
if not (job_dir / "best_checkpoint.pt").exists():
    # 提取参数
    kernel = config.get("kernel")
    num_inducing_points = config.get("num_inducing_points")
    # ... 更多参数
    
    # 初始化模型
    model = models.DeepKernelGP(
        job_dir=job_dir,
        kernel=kernel,
        num_inducing_points=num_inducing_points,
        inducing_point_dataset=ds_train,
        # ... 更多参数
    )
    
    # 训练
    model.fit(ds_train, ds_valid)
```

### `train_ensemble(ds_train, ds_valid, job_dir, config, dim_input)`

训练集成模型。

**参数：** 同上

**返回：** None

**功能：**
1. 从配置中提取集成大小和模型参数
2. 循环训练多个 TARNet 模型
3. 每个模型保存到独立的子目录

**实现逻辑：**
```python
ensemble_size = config.get("ensemble_size")
for ensemble_id in range(ensemble_size):
    out_dir = job_dir / f"model-{ensemble_id}"
    
    # 跳过已训练的模型
    if (out_dir / "best_checkpoint.pt").exists():
        continue
    
    # 初始化模型
    model = models.TARNet(
        job_dir=out_dir,
        # ... 参数
    )
    
    # 训练
    model.fit(ds_train, ds_valid)
```

### `TRAIN_FUNCTIONS`

训练函数字典。

```python
TRAIN_FUNCTIONS = {
    "deep_kernel_gp": train_deep_kernel_gp,
    "ensemble": train_ensemble,
}
```

## 预测函数

### `predict_deep_kernel_gp(dataset, job_dir, config)`

使用 Deep Kernel GP 模型进行预测。

**参数：**
- `dataset` (Dataset): 待预测的数据集
- `job_dir` (Path): 模型检查点目录
- `config` (dict): 配置字典

**返回：** tuple - (mu_0, mu_1)

**功能：**
1. 从配置中提取模型参数
2. 初始化模型（不训练）
3. 加载检查点
4. 预测 μ₀ 和 μ₁

**实现逻辑：**
```python
# 提取参数
kernel = config.get("kernel")
num_inducing_points = config.get("num_inducing_points")
# ... 更多参数

# 初始化模型
model = models.DeepKernelGP(
    job_dir=job_dir,
    kernel=kernel,
    num_inducing_points=num_inducing_points,
    inducing_point_dataset=dataset,
    # ... 更多参数
)

# 加载检查点
model.load()

# 预测
return model.predict_mus(dataset)
```

**返回值：**
- `mu_0` (np.ndarray): 对照组均值预测，shape: `(num_samples, batch_size)`
- `mu_1` (np.ndarray): 处理组均值预测，shape: `(num_samples, batch_size)`

### `predict_ensemble(dataset, job_dir, config)`

使用集成模型进行预测。

**参数：** 同上

**返回：** tuple - (mu_0, mu_1)

**功能：**
1. 从配置中提取集成大小和模型参数
2. 循环加载每个模型
3. 聚合所有模型的预测结果

**实现逻辑：**
```python
ensemble_size = config.get("ensemble_size")
mu_0 = []
mu_1 = []

for ensemble_id in range(ensemble_size):
    out_dir = job_dir / f"model-{ensemble_id}"
    
    # 初始化模型
    model = models.TARNet(
        job_dir=out_dir,
        # ... 参数
    )
    
    # 加载检查点
    model.load()
    
    # 预测
    mus = model.predict_mus(dataset)
    mu_0.append(mus[0])
    mu_1.append(mus[1])

return np.asarray(mu_0), np.asarray(mu_1)
```

**返回值：**
- `mu_0` (np.ndarray): 对照组均值预测，shape: `(ensemble_size, batch_size)`
- `mu_1` (np.ndarray): 处理组均值预测，shape: `(ensemble_size, batch_size)`

### `PREDICT_FUNCTIONS`

预测函数字典。

```python
PREDICT_FUNCTIONS = {
    "deep_kernel_gp": predict_deep_kernel_gp,
    "ensemble": predict_ensemble,
}
```

## 使用示例

### 训练模型
```python
from causal_bald.application.workflows import utils
from causal_bald.library import datasets
from pathlib import Path

# 加载数据集
ds_train = datasets.IHDP(root="assets/", split="train", mode="mu", seed=0)
ds_valid = datasets.IHDP(root="assets/", split="valid", mode="mu", seed=0)

# 配置
config = {
    "kernel": "Matern32",
    "num_inducing_points": 100,
    "dim_hidden": 200,
    "learning_rate": 0.001,
    "batch_size": 100,
    "epochs": 500,
    # ... 更多参数
}

# 训练
utils.train_deep_kernel_gp(
    ds_train=ds_train,
    ds_valid=ds_valid,
    job_dir=Path("experiments/checkpoints/"),
    config=config,
    dim_input=ds_train.dim_input,
)
```

### 预测
```python
# 加载测试集
ds_test = datasets.IHDP(root="assets/", split="test", mode="mu", seed=0)

# 预测
mu_0, mu_1 = utils.predict_deep_kernel_gp(
    dataset=ds_test,
    job_dir=Path("experiments/checkpoints/"),
    config=config,
)

# 计算治疗效应
tau_pred = mu_1.mean(0) - mu_0.mean(0)  # 取均值
```

### 目录生成
```python
config = {
    "kernel": "Matern32",
    "num_inducing_points": 100,
    "dim_hidden": 200,
    # ... 更多参数
}

experiment_dir = utils.directory_deep_kernel_gp(
    base_dir="experiments/",
    config=config,
)
# 输出: experiments/deep_kernel_gp/kernel-Matern32_ip-100-dh-200_...
```

## 扩展新模型

要添加新模型的训练和预测函数：

1. **实现训练函数**：
```python
def train_new_model(ds_train, ds_valid, job_dir, config, dim_input):
    # 检查检查点
    if (job_dir / "best_checkpoint.pt").exists():
        return
    
    # 初始化模型
    model = models.NewModel(...)
    
    # 训练
    model.fit(ds_train, ds_valid)
```

2. **实现预测函数**：
```python
def predict_new_model(dataset, job_dir, config):
    # 初始化模型
    model = models.NewModel(...)
    
    # 加载检查点
    model.load()
    
    # 预测
    return model.predict_mus(dataset)
```

3. **添加到字典**：
```python
TRAIN_FUNCTIONS["new_model"] = train_new_model
PREDICT_FUNCTIONS["new_model"] = predict_new_model
```

4. **添加目录函数**（可选）：
```python
def directory_new_model(base_dir, config):
    # 生成目录路径
    return Path(base_dir) / "new_model" / f"config-{...}"

DIRECTORIES["new_model"] = directory_new_model
```

## 注意事项

1. **检查点检查**：训练函数会检查检查点是否存在，避免重复训练
2. **参数提取**：所有函数都需要从配置字典中提取参数
3. **目录结构**：目录路径基于配置参数生成，确保一致性
4. **模型初始化**：预测时需要重新初始化模型（不训练）
5. **返回值格式**：预测函数返回的格式需要与获取函数兼容

## 相关文档

- [工作流概览](overview.md)
- [训练工作流](training.md)
- [主动学习工作流](active_learning.md)
- [Deep Kernel GP 模型](../../library/models/deep_kernel.md)
- [TARNet 模型](../../library/models/tarnet.md)

