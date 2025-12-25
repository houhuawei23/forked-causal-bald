# 评估工作流

## 概述

`causal_bald.application.workflows.evaluation` 模块实现了模型评估和结果可视化功能，包括 PEHE 指标计算、收敛曲线绘制、演化过程可视化等。

## 核心函数

### `pehe(experiment_dir, output_dir)`

计算 PEHE（Precision in Estimation of Heterogeneous Effect）指标。

**参数：**
- `experiment_dir` (Path): 实验目录，包含多个试验
- `output_dir` (Path): 结果输出目录

**返回：** None

**功能：**
1. 遍历所有试验目录
2. 对每个获取步骤计算 PEHE
3. 聚合所有试验的结果
4. 保存到 JSON 文件

**输出文件：**
- `{acquisition_function}_pehe.json`: 包含所有试验的 PEHE 结果

**JSON 格式：**
```json
{
    "acquisition_function": "mu-rho",
    "trial-000": {
        "value": [0.5, 0.4, 0.3, ...],
        "num_acquired": [100, 110, 120, ...]
    },
    "trial-001": {...},
    ...
}
```

### `plot_convergence(experiment_dir, methods)`

绘制收敛曲线，比较不同方法的性能。

**参数：**
- `experiment_dir` (Path): 实验目录
- `methods` (list): 要绘制的方法列表，如 `["mu-rho", "random", "mu"]`

**返回：** None

**功能：**
1. 加载每个方法的 PEHE 结果
2. 计算均值和标准误差
3. 绘制带误差带的收敛曲线
4. 保存到 `convergence.png`

**可视化特点：**
- 不同方法使用不同颜色和标记
- 显示标准误差带（±SEM）
- 自动图例

### `plot_evolution(experiment_dir, trial, num_steps)`

绘制单个试验的演化过程。

**参数：**
- `experiment_dir` (Path): 实验目录
- `trial` (int): 试验编号
- `num_steps` (int): 要绘制的步数

**返回：** None

**功能：**
1. 加载指定试验的配置和数据
2. 对每个获取步骤：
   - 加载获取的样本
   - 预测治疗效应
   - 生成分布图
3. 保存到 `evolution/` 目录

**输出文件：**
- `evolution/distribution_{step:02d}.png`: 每个步骤的分布图

### `plot_distribution(experiment_dir, acquisition_step)`

绘制指定获取步骤的分布图，聚合所有试验。

**参数：**
- `experiment_dir` (Path): 实验目录
- `acquisition_step` (int): 获取步骤编号

**返回：** None

**功能：**
1. 遍历所有试验
2. 加载指定步骤的获取样本和预测结果
3. 聚合所有试验的数据
4. 生成分布图

**输出文件：**
- `distribution_{acquisition_step:02d}.png`

### `plot_dataset(config, output_dir)`

绘制数据集可视化。

**参数：**
- `config` (dict): 配置字典，包含数据集信息
- `output_dir` (Path): 输出目录

**返回：** None

**功能：**
1. 加载数据集
2. 根据数据集类型选择可视化函数
3. 保存可视化结果

**支持的数据集：**
- `cmnist`: 使用 `plotting.mnist()`
- `synthetic`: 使用 `plotting.dataset()`

**输出文件：**
- `{dataset_name}_dataset.png`

### `plot_errorbars(experiment_dir)`

绘制误差条图。

**参数：**
- `experiment_dir` (Path): 实验目录

**返回：** None

**功能：**
1. 遍历所有试验
2. 加载模型预测结果
3. 计算真实值和预测值的误差
4. 绘制误差条图

**输出文件：**
- `trial-{trial:03d}/scatter.png`

## PEHE 计算

### 计算公式

```python
def rmse_fn(y_pred, y):
    return np.sqrt(np.mean(np.square(y_pred - y)))

tau_pred = (mu_1 - mu_0) * ds_test.y_std[0]
tau_true = ds_test.mu1 - ds_test.mu0
pehe = rmse_fn(tau_pred.mean(0), tau_true)
```

### 计算流程

```python
for i in range(max_acquisitions):
    # 1. 加载获取的样本索引
    acquired_path = trial_dir / f"acquisition-{i:03d}" / "aquired.json"
    acquired_indices = load_acquired_indices(acquired_path)
    
    # 2. 加载模型并预测
    mu_0, mu_1 = utils.PREDICT_FUNCTIONS[model_name](
        dataset=ds_test,
        job_dir=acquisition_dir,
        config=config,
    )
    
    # 3. 计算预测的治疗效应
    tau_pred = (mu_1 - mu_0) * ds_test.y_std[0]
    
    # 4. 获取真实治疗效应
    tau_true = ds_test.mu1 - ds_test.mu0
    
    # 5. 计算 PEHE
    pehe_value = rmse_fn(tau_pred.mean(0), tau_true)
    
    # 6. 记录结果
    trial_pehe["value"].append(pehe_value)
    trial_pehe["num_acquired"].append(len(acquired_indices))
```

## 可视化样式

### 方法样式配置

```python
styles = {
    "random": ("C5", "--", "X"),
    "mu": ("C9", "-", "x"),
    "tau": ("C3", "-", "+"),
    "rho": ("C2", "-", "|"),
    "mu-pi": ("C4", "-", "^"),
    "mu-rho": ("C0", "-", "o"),
    "pi": ("C8", "-", "*"),
    "sundin": ("C1", "-", "s"),
}
```

格式：`(颜色, 线型, 标记)`

## 使用示例

### 计算 PEHE
```python
from causal_bald.application.workflows import evaluation
from pathlib import Path

evaluation.pehe(
    experiment_dir=Path("experiments/active_learning/..."),
    output_dir=Path("experiments/results/"),
)
```

### CLI 使用
```bash
# 计算 PEHE
causal-bald \
    evaluate \
        --experiment-dir experiments/active_learning/... \
        --output-dir experiments/results \
    pehe

# 绘制收敛曲线
causal-bald \
    evaluate \
        --experiment-dir experiments/results \
    plot-convergence \
        -m mu-rho -m random -m mu

# 绘制演化过程
causal-bald \
    evaluate \
        --experiment-dir experiments/active_learning/... \
    plot-evolution \
        --trial 0 --num-steps 10

# 绘制分布
causal-bald \
    evaluate \
        --experiment-dir experiments/active_learning/... \
    plot-distribution \
        --acquisition-step 5

# 绘制数据集
causal-bald \
    plot-dataset \
        --output-dir assets/ \
    synthetic
```

## 目录结构

评估结果保存在以下目录结构：

```
output_dir/
    {acquisition_function}_pehe.json      # PEHE 结果
    convergence.png                        # 收敛曲线
    distribution_{step:02d}.png            # 分布图

experiment_dir/
    trial-{trial:03d}/
        pehe.json                          # 单个试验的 PEHE
        scatter.png                        # 误差条图
        evolution/
            distribution_{step:02d}.png    # 演化过程
```

## 统计计算

### 均值和标准误差

```python
pehes = np.asarray(pehes)  # shape: (num_trials, num_steps)
mean_pehe = pehes.mean(0)  # 均值
sem_pehe = stats.sem(pehes, axis=0)  # 标准误差
```

### 误差带

```python
plt.fill_between(
    x=x,
    y1=mean_pehe - sem_pehe,
    y2=mean_pehe + sem_pehe,
    color=color,
    alpha=0.3,
)
```

## 注意事项

1. **文件存在性**：如果 PEHE 结果已存在，会直接加载而不重新计算
2. **数据标准化**：预测结果需要乘以 `y_std` 进行反标准化
3. **试验数量**：确保有足够的试验以获得可靠的统计结果
4. **内存使用**：大量试验和步骤可能占用较多内存

## 相关文档

- [工作流概览](overview.md)
- [可视化函数文档](../../library/plotting.md)
- [工具函数](utils.md)

