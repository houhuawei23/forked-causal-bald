# 主动学习工作流

## 概述

`causal_bald.application.workflows.active_learning` 模块实现了主动学习循环，用于高效地选择最有价值的样本进行标注，从而提高模型学习效率。

## 核心函数

### `active_learner(model_name, config, experiment_dir, trial)`

执行主动学习主循环。

**参数：**
- `model_name` (str): 模型名称，如 `"deep_kernel_gp"` 或 `"ensemble"`
- `config` (dict): 配置字典，包含数据集、模型和训练参数
- `experiment_dir` (Path): 实验根目录
- `trial` (int): 试验编号

**返回：** None

**功能：**
1. 设置数据集随机种子
2. 初始化主动学习数据集
3. 创建实验目录结构
4. 获取或训练倾向性评分模型（如需要）
5. 执行主动学习循环

## 主动学习循环

### 循环流程

```python
for i in range(max_acquisitions):
    # 1. 确定批次大小
    batch_size = warm_start_size if i == 0 else step_size
    
    # 2. 计算获取分数
    if i == 0:
        # 首次使用随机获取
        scores = random(...)
    else:
        # 使用模型预测计算获取分数
        mu_0, mu_1 = predict_pool_set(...)
        scores = acquisition_function(mu_0, mu_1, ...)
    
    # 3. 选择样本
    if temperature > 0.0:
        if use_gumbel:
            # Gumbel Top-K 采样
            idx = gumbel_top_k(scores, batch_size)
        else:
            # 温度缩放 + 多项式采样
            idx = temperature_sampling(scores, batch_size)
    else:
        # 确定性选择（Top-K）
        idx = top_k(scores, batch_size)
    
    # 4. 获取样本
    ds_active.acquire(idx)
    
    # 5. 训练模型
    train_model(ds_active.training_dataset, ...)
    
    # 6. 保存获取的样本索引
    save_acquired_indices(idx)
```

### 获取策略

#### 1. 随机获取（首次迭代）
```python
scores = acquisitions.random(
    mu_0=None, 
    mu_1=None, 
    t=ds_active.dataset.t, 
    pt=pt, 
    temperature=None
)
```

#### 2. 基于模型的获取
```python
# 预测池集
mu_0, mu_1 = utils.PREDICT_FUNCTIONS[model_name](
    dataset=ds_active.dataset,
    job_dir=trial_dir / f"acquisition-{i-1:03d}",
    config=config,
)

# 计算获取分数
scores = acquisition_function(
    mu_0=mu_0,
    mu_1=mu_1,
    t=ds_active.dataset.t,
    pt=pt,
    temperature=temperature,
)
```

### 样本选择方法

#### Gumbel Top-K 采样
```python
if use_gumbel:
    p = scores + scipy.stats.gumbel_r.rvs(
        loc=0, scale=1, size=len(scores), random_state=None,
    )
    idx = np.argpartition(p, -batch_size)[-batch_size:]
```

#### 温度缩放采样
```python
else:
    scores = np.exp(scores)
    p = scores / scores.sum()
    idx = np.random.choice(
        range(len(p)), replace=False, p=p, size=batch_size,
    )
```

#### 确定性 Top-K
```python
else:
    idx = np.argsort(scores)[-batch_size:]
```

## 倾向性评分获取

### `get_propensities(trial_dir, config)`

获取或训练倾向性评分模型。

**参数：**
- `trial_dir` (Path): 试验目录
- `config` (dict): 配置字典

**返回：** np.ndarray 或 None

**功能：**
- 如果获取函数需要倾向性评分（`pi` 或 `mu-pi`），则训练一个神经网络模型来预测倾向性评分
- 否则返回 None

**实现逻辑：**
```python
if config.get("acquisition_function") in ["pi", "mu-pi"]:
    # 1. 准备倾向性评分数据集
    config_pi_train = deepcopy(config.get("ds_train"))
    config_pi_train["mode"] = "pi"
    ds_pi_train = datasets.DATASETS.get(dataset_name)(**config_pi_train)
    
    # 2. 初始化模型
    pi_model = models.NeuralNetwork(...)
    
    # 3. 训练（如果检查点不存在）
    if not checkpoint_exists:
        pi_model.fit(ds_pi_train, ds_pi_valid)
    
    # 4. 加载并预测
    pi_model.load()
    return pi_model.predict_mean(ds_pi_train).ravel()
else:
    return None
```

## 配置参数

### 必需参数
- `dataset_name`: 数据集名称
- `ds_train`: 训练数据集配置
- `ds_valid`: 验证数据集配置
- `model_name`: 模型名称
- `step_size`: 每次获取的样本数
- `warm_start_size`: 初始样本数
- `max_acquisitions`: 最大获取步数
- `acquisition_function`: 获取函数名称

### 可选参数
- `temperature`: 温度参数（默认 1.0）
- `use_gumbel`: 是否使用 Gumbel 采样（默认 True）
- `seed`: 随机种子

## 目录结构

主动学习会创建以下目录结构：

```
experiment_dir/
    trial-{trial:03d}/
        config.json                    # 配置保存
        pi/                            # 倾向性评分模型（如需要）
            best_checkpoint.pt
        acquisition-{i:03d}/
            aquired.json               # 获取的样本索引
            best_checkpoint.pt         # 模型检查点
            events.out.tfevents.*      # TensorBoard 日志
```

## 使用示例

### 基本使用
```python
from causal_bald.application.workflows import active_learning
from pathlib import Path

config = {
    "dataset_name": "ihdp",
    "ds_train": {"root": "assets/", "split": "train", "mode": "mu", "seed": 0},
    "ds_valid": {"root": "assets/", "split": "valid", "mode": "mu", "seed": 0},
    "model_name": "deep_kernel_gp",
    "step_size": 10,
    "warm_start_size": 100,
    "max_acquisitions": 38,
    "acquisition_function": "mu-rho",
    "temperature": 0.25,
    "use_gumbel": True,
    # ... 模型配置
}

active_learning.active_learner(
    model_name="deep_kernel_gp",
    config=config,
    experiment_dir=Path("experiments/"),
    trial=0,
)
```

### CLI 使用
```bash
causal-bald \
    active-learning \
        --job-dir experiments/ \
        --num-trials 5 \
        --step-size 10 \
        --warm-start-size 100 \
        --max-acquisitions 38 \
        --acquisition-function mu-rho \
        --temperature 0.25 \
    ihdp --root assets/ \
    deep-kernel-gp
```

## 获取函数

支持的获取函数（详见 [获取函数文档](../../library/acquisitions.md)）：

- `random`: 随机获取
- `mu`: μ-BALD
- `rho`: ρ-BALD
- `mu-rho`: μρ-BALD（推荐）
- `pi`: π-BALD
- `mu-pi`: μπ-BALD
- `tau`: τ-BALD
- `sundin`: Sundin 方法

## 注意事项

1. **首次迭代**：首次迭代总是使用随机获取，因为此时还没有训练好的模型
2. **检查点**：如果某个获取步骤的检查点已存在，会跳过该步骤
3. **倾向性评分**：某些获取函数（`pi`, `mu-pi`）需要倾向性评分，会自动训练一个神经网络模型
4. **随机种子**：每个试验使用不同的随机种子，确保结果的可重复性
5. **内存管理**：大量获取步骤可能会占用较多内存，注意监控

## 相关文档

- [工作流概览](overview.md)
- [获取函数文档](../../library/acquisitions.md)
- [数据集文档](../../library/datasets/active_learning.md)
- [工具函数](utils.md)

