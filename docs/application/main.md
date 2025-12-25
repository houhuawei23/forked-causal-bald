# CLI 接口文档

## 概述

`causal_bald.application.main` 模块提供了 Causal-BALD 项目的命令行接口（CLI）。该接口基于 Click 框架构建，支持链式命令组合，提供了完整的实验配置和运行功能。

## 主命令组

### `cli()`

主命令行接口组，所有命令都挂载在此组下。

**使用方式：**
```bash
causal-bald [命令] [选项] [子命令] [子选项]
```

## 实验模式命令

### `tune`

启动超参数调优模式。

**选项：**
- `--job-dir` (str, 必需): 检查点和结果保存位置
- `--max-samples` (int, 默认=200): 搜索空间最大采样数
- `--gpu-per-trial` (float, 默认=0.0): 每个试验的 GPU 数量
- `--cpu-per-trial` (float, 默认=1.0): 每个试验的 CPU 数量
- `--object-memory-store` (int, 默认=8000000000): Ray 对象存储内存大小（字节）
- `--seed` (int, 默认=1331): 随机数生成器种子

**示例：**
```bash
causal-bald tune --job-dir experiments/ --max-samples 100 ihdp --root assets/ deep-kernel-gp
```

### `train`

启动模型训练模式。

**选项：**
- `--job-dir` (str, 必需): 检查点和结果保存位置
- `--num-trials` (int, 默认=1): 试验次数
- `--gpu-per-trial` (float, 默认=0.0): 每个试验的 GPU 数量
- `--cpu-per-trial` (float, 默认=1.0): 每个试验的 CPU 数量
- `--object-memory-store` (int, 默认=8000000000): Ray 对象存储内存大小
- `--verbose` (bool, 默认=False): 是否显示详细信息
- `--seed` (int, 默认=1331): 随机数生成器种子

**示例：**
```bash
causal-bald train --job-dir experiments/ --num-trials 5 ihdp --root assets/ deep-kernel-gp
```

### `active-learning`

启动主动学习模式。

**选项：**
- `--job-dir` (str, 必需): 检查点和结果保存位置
- `--num-trials` (int, 默认=1): 试验次数
- `--step-size` (int, 默认=10): 每次获取的样本数量
- `--warm-start-size` (int, 默认=50): 初始获取的样本数量
- `--max-acquisitions` (int, 默认=100): 最大获取步数
- `--acquisition-function` (str, 默认="mu-rho"): 获取函数名称
  - 可选值: `random`, `mu`, `rho`, `mu-rho`, `pi`, `mu-pi`, `tau`, `sundin`
- `--temperature` (float, 默认=1.0): 获取函数温度参数
- `--use-gumbel` (bool, 默认=True): 是否使用 Gumbel Top-K 采样
- `--gpu-per-trial` (float, 默认=0.0): 每个试验的 GPU 数量
- `--cpu-per-trial` (float, 默认=1.0): 每个试验的 CPU 数量
- `--object-memory-store` (int, 默认=8000000000): Ray 对象存储内存大小
- `--verbose` (bool, 默认=False): 是否显示详细信息
- `--seed` (int, 默认=1331): 随机数生成器种子

**示例：**
```bash
causal-bald active-learning \
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

### `evaluate`

启动评估模式。

**选项：**
- `--experiment-dir` (str, 必需): 读取检查点的实验目录
- `--output-dir` (str, 可选): 结果输出目录（默认与实验目录相同）

**示例：**
```bash
causal-bald evaluate \
    --experiment-dir experiments/active_learning/... \
    --output-dir experiments/results \
    pehe
```

## 数据集命令

### `ihdp`

配置 IHDP 数据集。

**选项：**
- `--root` (str, 必需): 数据集根目录路径

**示例：**
```bash
causal-bald ihdp --root assets/
```

### `cmnist`

配置 CMNIST 数据集。

**选项：**
- `--root` (str, 必需): 数据集根目录路径
- `--subsample` (float, 可选): 数据集子采样比例

**示例：**
```bash
causal-bald cmnist --root assets/ --subsample 0.5
```

### `synthetic`

配置合成数据集。

**选项：**
- `--num-examples` (int, 默认=10000): 训练样本数量
- `--beta` (float, 默认=2.0): x 对 t 的影响系数
- `--bimodal` (bool, 默认=False): 是否从双峰分布采样 x
- `--sigma` (float, 默认=1.0): y 的随机噪声标准差
- `--domain-limit` (float, 默认=2.5): x 的定义域范围 [-domain_limit, domain_limit]

**示例：**
```bash
causal-bald synthetic \
    --num-examples 10000 \
    --beta 2.0 \
    --sigma 1.0
```

## 模型命令

### `deep-kernel-gp`

配置 Deep Kernel GP 模型。

**选项：**
- `--kernel` (str, 默认="Matern32"): GP 核函数类型
  - 可选值: `RBF`, `Matern12`, `Matern32`, `Matern52`
- `--num-inducing-points` (int, 默认=100): Deep GP 诱导点数量
- `--dim-hidden` (int, 默认=200): 隐藏层神经元数量
- `--dim-output` (int, 默认=1): 输出维度
- `--depth` (int, 默认=3): 特征提取器深度
- `--negative-slope` (float, 默认=-1): LeakyReLU 负斜率（-1 表示使用 ELU）
- `--dropout-rate` (float, 默认=0.1): Dropout 率
- `--spectral-norm` (float, 默认=0.95): 谱归一化系数（0.0 表示不使用）
- `--learning-rate` (float, 默认=1e-3): 梯度下降学习率
- `--batch-size` (int, 默认=100): 每个训练步骤的样本数量
- `--epochs` (int, 默认=500): 训练轮数

**示例：**
```bash
causal-bald deep-kernel-gp \
    --kernel Matern32 \
    --num-inducing-points 100 \
    --dim-hidden 200 \
    --depth 3 \
    --learning-rate 0.001 \
    --batch-size 100 \
    --epochs 500
```

### `ensemble`

配置集成模型（Ensemble）。

**选项：**
- `--dim-hidden` (int, 默认=400): 神经元数量
- `--dim-output` (int, 默认=2): 输出维度
- `--depth` (int, 默认=3): 特征提取器深度
- `--negative-slope` (float, 默认=-1): LeakyReLU 负斜率
- `--dropout-rate` (float, 默认=0.15): Dropout 率
- `--spectral-norm` (float, 默认=0.95): 谱归一化系数
- `--learning-rate` (float, 默认=1e-3): 学习率
- `--batch-size` (int, 默认=200): 批次大小
- `--epochs` (int, 默认=500): 训练轮数
- `--ensemble-size` (int, 默认=5): 集成中模型数量

**示例：**
```bash
causal-bald ensemble \
    --dim-hidden 400 \
    --ensemble-size 5 \
    --epochs 500
```

## 评估命令

### `pehe`

计算 PEHE（Precision in Estimation of Heterogeneous Effect）指标。

**示例：**
```bash
causal-bald evaluate --experiment-dir experiments/... pehe
```

### `plot-convergence`

绘制收敛曲线。

**选项：**
- `--methods` / `-m` (str, 可重复): 要绘制的方法列表

**示例：**
```bash
causal-bald evaluate \
    --experiment-dir experiments/due/ihdp \
    plot-convergence \
    -m mu-rho -m random -m mu
```

### `plot-evolution`

绘制演化过程。

**选项：**
- `--trial` (int): 要绘制的试验编号
- `--num-steps` (int): 要绘制的步数

**示例：**
```bash
causal-bald evaluate \
    --experiment-dir experiments/... \
    plot-evolution \
    --trial 0 --num-steps 10
```

### `plot-distribution`

绘制分布图。

**选项：**
- `--acquisition-step` (int): 要绘制的获取步骤

**示例：**
```bash
causal-bald evaluate \
    --experiment-dir experiments/... \
    plot-distribution \
    --acquisition-step 5
```

### `plot-dataset`

绘制数据集可视化。

**选项：**
- `--output-dir` (str, 可选): 输出目录

**示例：**
```bash
causal-bald plot-dataset \
    --output-dir assets/ \
    synthetic
```

## 命令链式组合

CLI 支持链式命令组合，例如：

```bash
causal-bald \
    active-learning \
        --job-dir experiments/ \
        --num-trials 5 \
        --acquisition-function mu-rho \
    ihdp \
        --root assets/ \
    deep-kernel-gp \
        --kernel Matern32 \
        --epochs 500
```

命令的执行顺序：
1. `active-learning` 设置主动学习参数
2. `ihdp` 配置数据集
3. `deep-kernel-gp` 配置模型并启动训练

## 上下文对象

CLI 使用 Click 的上下文对象 (`context.obj`) 在不同命令间传递配置信息。上下文对象包含：

- `n_gpu`: 可用 GPU 数量
- `job_dir`: 工作目录
- `mode`: 运行模式（tune/train/active）
- `dataset_name`: 数据集名称
- `ds_train/ds_valid/ds_test`: 数据集配置
- `model_name`: 模型名称
- 模型和训练相关参数

## 注意事项

1. **GPU 资源**：如果系统没有 GPU，`gpu-per-trial` 会自动设置为 0
2. **Ray 初始化**：`tune`、`train` 和 `active-learning` 命令会自动初始化 Ray
3. **目录结构**：实验目录会根据配置参数自动生成结构化路径
4. **检查点**：如果检查点已存在，训练会跳过已完成的步骤

## 相关文档

- [工作流概览](workflows/overview.md)
- [主动学习工作流](workflows/active_learning.md)
- [训练工作流](workflows/training.md)
- [评估工作流](workflows/evaluation.md)

