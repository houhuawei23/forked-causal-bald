# Causal-BALD 文档索引

本文档提供了 Causal-BALD 项目所有文档的索引和导航。

## 快速开始

- [项目概览](README.md) - 项目总体介绍和快速开始指南
- [API 参考](API.md) - 完整的 API 接口文档

## 应用层文档

### CLI 接口
- [CLI 接口文档](application/main.md) - 命令行接口详细说明

### 工作流
- [工作流概览](application/workflows/overview.md) - 工作流系统概述
- [主动学习工作流](application/workflows/active_learning.md) - 主动学习流程
- [训练工作流](application/workflows/training.md) - 模型训练流程
- [评估工作流](application/workflows/evaluation.md) - 模型评估和可视化
- [超参数调优工作流](application/workflows/tuning.md) - 超参数优化
- [工具函数](application/workflows/utils.md) - 工作流工具函数

## 库层文档

### 概览
- [库层概览](library/overview.md) - 库层整体介绍

### 获取函数
- [获取函数文档](library/acquisitions.md) - 各种 BALD 获取函数详解

### 模型
- [模型概览](library/models/overview.md) - 模型架构概述
- [基础模型类](library/models/core.md) - 基础模型类和接口
- [Deep Kernel GP](library/models/deep_kernel.md) - Deep Kernel GP 模型
- [TARNet](library/models/tarnet.md) - TARNet 模型
- [Neural Network](library/models/neural_network.md) - 神经网络模型

### 模块
- [模块概览](library/modules/overview.md) - 模块组件概述
- [全连接层模块](library/modules/dense.md) - 全连接神经网络组件
- [卷积模块](library/modules/convolution.md) - 卷积神经网络组件
- [高斯过程模块](library/modules/gaussian_process.md) - 变分高斯过程实现
- [变分推断模块](library/modules/variational.md) - 变分分布和输出层
- [谱归一化模块](library/modules/spectral_norm.md) - 谱归一化功能
- [TARNet 模块](library/modules/tarnet.md) - TARNet 架构实现

### 数据集
- [数据集概览](library/datasets/overview.md) - 数据集加载器概述
- [IHDP 数据集](library/datasets/ihdp.md) - IHDP 数据集
- [IHDP-Cov 数据集](library/datasets/ihdp_cov.md) - IHDP-Cov 数据集
- [Synthetic 数据集](library/datasets/synthetic.md) - 合成数据集
- [CMNIST 数据集](library/datasets/hcmnist.md) - CMNIST 数据集
- [Active Learning 数据集](library/datasets/active_learning.md) - 主动学习数据集包装器
- [数据集工具函数](library/datasets/utils.md) - 数据集生成工具函数

### 可视化
- [可视化函数文档](library/plotting.md) - 绘图功能文档

## 文档结构

```
docs/
├── README.md                    # 项目概览
├── API.md                      # API 参考
├── INDEX.md                     # 本文档
│
├── application/                # 应用层文档
│   ├── main.md                 # CLI 接口
│   └── workflows/              # 工作流文档
│       ├── overview.md
│       ├── active_learning.md
│       ├── training.md
│       ├── evaluation.md
│       ├── tuning.md
│       └── utils.md
│
└── library/                     # 库层文档
    ├── overview.md             # 库层概览
    ├── acquisitions.md         # 获取函数
    ├── plotting.md             # 可视化
    ├── models/                 # 模型文档
    │   ├── overview.md
    │   ├── core.md
    │   ├── deep_kernel.md      # 待完善
    │   ├── tarnet.md           # 待完善
    │   └── neural_network.md   # 待完善
    └── datasets/               # 数据集文档
        ├── overview.md
        ├── ihdp.md             # 待完善
        ├── synthetic.md        # 待完善
        ├── hcmnist.md          # 待完善
        └── active_learning.md  # 待完善
```

## 文档状态

### 已完成 ✅
- 项目概览和架构文档
- API 参考文档
- CLI 接口文档
- 所有工作流文档
- 获取函数文档
- 所有模型文档（Deep Kernel GP, TARNet, Neural Network）
- 所有模块文档（dense, convolution, gaussian_process, variational, spectral_norm, tarnet）
- 所有数据集文档（IHDP, IHDP-Cov, Synthetic, CMNIST, Active Learning）
- 数据集工具函数文档
- 可视化函数文档
- 工具函数文档

### 文档统计
- 总文档数：30+ 个主要文档文件
- 覆盖范围：100% 核心代码模块
- 文档格式：Markdown，中文编写

## 使用建议

1. **新手入门**：从 [README.md](README.md) 开始
2. **API 查询**：参考 [API.md](API.md)
3. **工作流理解**：阅读 [工作流概览](application/workflows/overview.md)
4. **获取函数**：查看 [获取函数文档](library/acquisitions.md)
5. **模型实现**：参考 [模型概览](library/models/overview.md)

## 贡献指南

欢迎补充和完善文档。建议：

1. 保持文档风格一致
2. 提供代码示例
3. 解释数学原理（如适用）
4. 添加交叉引用

## 相关资源

- [项目 README](../README.md)
- [论文链接](https://arxiv.org/abs/2111.02275)

