# Causal-BALD 项目代码文档

## 项目简介

Causal-BALD 是一个用于从高维观测数据中估计个性化治疗效果的深度学习框架。该项目实现了论文 [Causal-BALD: Deep Bayesian Active Learning of Outcomes to Infer Treatment-Effects from Observational Data](https://arxiv.org/abs/2111.02275) 中提出的方法。

### 核心功能

- **深度贝叶斯主动学习**：通过信息论驱动的获取函数，高效选择训练数据
- **因果效应估计**：估计个性化平均治疗效应（CATE）
- **多种获取策略**：支持多种 BALD 变体（μρ-BALD, μ-BALD, ρ-BALD, π-BALD, τ-BALD 等）
- **多种模型架构**：支持 Deep Kernel GP、TARNet、Ensemble 等模型

## 架构概览

项目采用分层架构设计，主要分为两个层次：

### 1. 应用层 (`causal_bald/application/`)

应用层提供命令行接口和工作流管理：

- **`main.py`**：CLI 入口点，提供统一的命令行接口
- **`workflows/`**：包含各种工作流实现
  - `active_learning.py`：主动学习工作流
  - `training.py`：模型训练工作流
  - `evaluation.py`：模型评估工作流
  - `tuning.py`：超参数调优工作流
  - `utils.py`：工作流工具函数

### 2. 库层 (`causal_bald/library/`)

库层包含核心算法和模型实现：

- **`acquisitions.py`**：获取函数实现
- **`models/`**：模型实现
  - `core.py`：基础模型类
  - `deep_kernel.py`：Deep Kernel GP 模型
  - `tarnet.py`：TARNet 模型
  - `neural_network.py`：神经网络模型
- **`modules/`**：神经网络模块
  - `dense.py`：全连接层模块
  - `convolution.py`：卷积模块
  - `gaussian_process.py`：高斯过程模块
  - `variational.py`：变分推断模块
  - `spectral_norm.py`：谱归一化模块
  - `tarnet.py`：TARNet 模块
- **`datasets/`**：数据集加载器
  - `ihdp.py`：IHDP 数据集
  - `synthetic.py`：合成数据集
  - `hcmnist.py`：CMNIST 数据集
  - `active_learning.py`：主动学习数据集包装器
- **`plotting.py`**：可视化函数
- **`utils.py`**：工具函数

## 模块组织

```
causal_bald/
├── application/          # 应用层
│   ├── main.py          # CLI 入口点
│   └── workflows/       # 工作流实现
├── library/             # 库层
│   ├── acquisitions.py  # 获取函数
│   ├── models/          # 模型实现
│   ├── modules/         # 神经网络模块
│   ├── datasets/        # 数据集加载器
│   ├── plotting.py      # 可视化
│   └── utils.py         # 工具函数
└── __init__.py
```

## 快速开始

### 安装

```bash
git clone git@github.com:[anon]/causal-bald.git
cd causal-bald
conda env create -f environment.yml
conda activate causal-bald
pip install .
```

### 基本使用

#### 主动学习示例

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
    ihdp \
        --root assets/ \
    deep-kernel-gp
```

#### 评估示例

```bash
causal-bald \
    evaluate \
        --experiment-dir experiments/active_learning/... \
        --output-dir experiments/results \
    pehe
```

## 文档导航

### 应用层文档

- [CLI 接口文档](application/main.md) - 命令行接口详细说明
- [工作流概览](application/workflows/overview.md) - 工作流系统概述
- [主动学习工作流](application/workflows/active_learning.md) - 主动学习流程
- [训练工作流](application/workflows/training.md) - 模型训练流程
- [评估工作流](application/workflows/evaluation.md) - 模型评估流程
- [超参数调优](application/workflows/tuning.md) - 超参数优化

### 库层文档

- [库层概览](library/overview.md) - 库层整体介绍
- [获取函数](library/acquisitions.md) - 各种 BALD 获取函数
- [模型文档](library/models/overview.md) - 模型实现文档
- [数据集文档](library/datasets/overview.md) - 数据集加载器文档
- [可视化函数](library/plotting.md) - 绘图功能文档

### API 文档

- [API 参考](API.md) - 完整的 API 接口文档

## 核心概念

### 主动学习

主动学习通过智能选择最有价值的样本进行标注，从而提高模型学习效率。在因果推断场景中，需要特别关注处理组和对照组的重叠区域。

### 获取函数

获取函数用于评估样本的信息价值。Causal-BALD 实现了多种获取函数：

- **μρ-BALD**：结合均值不确定性和重叠区域信息
- **μ-BALD**：基于均值的不确定性
- **ρ-BALD**：关注重叠区域
- **π-BALD**：基于倾向性评分
- **τ-BALD**：直接关注治疗效应不确定性

### 模型架构

项目支持多种模型架构：

1. **Deep Kernel GP**：结合深度学习和高斯过程的模型
2. **TARNet**：Treatment-Agnostic Representation Network
3. **Ensemble**：集成多个 TARNet 模型

## 相关资源

- [论文链接](https://arxiv.org/abs/2111.02275)
- [GitHub 仓库](https://github.com/[anon]/causal-bald)

## 贡献指南

欢迎贡献代码和文档。请确保：

1. 代码符合项目风格
2. 添加适当的文档字符串
3. 更新相关文档

## 许可证

Apache-2.0 License

