# Causal-BALD API 参考文档

本文档提供 Causal-BALD 项目的完整 API 接口参考，用于指导代码生成和开发。

## 应用层 API

### CLI 接口 (`causal_bald.application.main`)

#### `cli()`
主命令行接口组。

**命令列表：**
- `tune` - 超参数调优
- `train` - 模型训练
- `active-learning` - 主动学习
- `evaluate` - 模型评估
- `ihdp` - IHDP 数据集配置
- `cmnist` - CMNIST 数据集配置
- `synthetic` - 合成数据集配置
- `deep-kernel-gp` - Deep Kernel GP 模型配置
- `ensemble` - Ensemble 模型配置
- `pehe` - PEHE 评估
- `plot-convergence` - 绘制收敛曲线
- `plot-evolution` - 绘制演化过程
- `plot-distribution` - 绘制分布
- `plot-dataset` - 绘制数据集

### 工作流 API (`causal_bald.application.workflows`)

#### `active_learning.active_learner(model_name, config, experiment_dir, trial)`
执行主动学习循环。

**参数：**
- `model_name` (str): 模型名称
- `config` (dict): 配置字典
- `experiment_dir` (Path): 实验目录
- `trial` (int): 试验编号

**返回：** None

#### `training.trainer(config, experiment_dir, trial, model_name)`
训练模型。

**参数：**
- `config` (dict): 配置字典
- `experiment_dir` (Path): 实验目录
- `trial` (int): 试验编号
- `model_name` (str): 模型名称

**返回：** int

#### `evaluation.pehe(experiment_dir, output_dir)`
计算 PEHE 指标。

**参数：**
- `experiment_dir` (Path): 实验目录
- `output_dir` (Path): 输出目录

**返回：** None

#### `evaluation.plot_convergence(experiment_dir, methods)`
绘制收敛曲线。

**参数：**
- `experiment_dir` (Path): 实验目录
- `methods` (list): 方法列表

**返回：** None

#### `tuning.deep_kernel_gp_tuner(config)`
Deep Kernel GP 超参数调优。

**参数：**
- `config` (dict): 配置字典

**返回：** None

#### `tuning.tarnet_tuner(config)`
TARNet 超参数调优。

**参数：**
- `config` (dict): 配置字典

**返回：** None

## 库层 API

### 获取函数 (`causal_bald.library.acquisitions`)

#### `random(mu_0, mu_1, t, pt, temperature)`
随机获取函数。

**参数：**
- `mu_0` (np.ndarray): 对照组均值预测
- `mu_1` (np.ndarray): 处理组均值预测
- `t` (np.ndarray): 处理指示变量
- `pt` (np.ndarray): 倾向性评分
- `temperature` (float): 温度参数

**返回：** np.ndarray - 获取分数

#### `mu(mu_0, mu_1, t, pt, temperature)`
μ-BALD 获取函数。

**参数：** 同上

**返回：** np.ndarray - 获取分数

#### `rho(mu_0, mu_1, t, pt, temperature)`
ρ-BALD 获取函数。

**参数：** 同上

**返回：** np.ndarray - 获取分数

#### `mu_rho(mu_0, mu_1, t, pt, temperature)`
μρ-BALD 获取函数。

**参数：** 同上

**返回：** np.ndarray - 获取分数

#### `pi(mu_0, mu_1, t, pt, temperature)`
π-BALD 获取函数。

**参数：** 同上

**返回：** np.ndarray - 获取分数

#### `mu_pi(mu_0, mu_1, t, pt, temperature)`
μπ-BALD 获取函数。

**参数：** 同上

**返回：** np.ndarray - 获取分数

#### `tau(mu_0, mu_1, t, pt, temperature)`
τ-BALD 获取函数。

**参数：** 同上

**返回：** np.ndarray - 获取分数

#### `sundin(mu_0, mu_1, t, pt, temperature)`
Sundin 获取函数。

**参数：** 同上

**返回：** np.ndarray - 获取分数

### 模型 API (`causal_bald.library.models`)

#### `core.BaseModel`
基础模型抽象类。

**方法：**
- `fit(train_dataset, tune_dataset)` - 训练模型
- `save()` - 保存模型
- `load()` - 加载模型

#### `core.PyTorchModel`
PyTorch 模型基类。

**属性：**
- `network` - 神经网络
- `optimizer` - 优化器
- `metrics` - 评估指标

**方法：**
- `train_step(engine, batch)` - 训练步骤
- `tune_step(engine, batch)` - 验证步骤
- `fit(train_dataset, tune_dataset)` - 训练模型
- `save()` - 保存模型
- `load()` - 加载模型

#### `deep_kernel.DeepKernelGP`
Deep Kernel GP 模型。

**参数：**
- `job_dir` (str): 工作目录
- `kernel` (str): 核函数类型
- `num_inducing_points` (int): 诱导点数量
- `inducing_point_dataset` (Dataset): 诱导点数据集
- `architecture` (str): 架构类型
- `dim_input` (int): 输入维度
- `dim_hidden` (int): 隐藏层维度
- `dim_output` (int): 输出维度
- `depth` (int): 网络深度
- `negative_slope` (float): LeakyReLU 负斜率
- `batch_norm` (bool): 是否使用批归一化
- `spectral_norm` (float): 谱归一化系数
- `dropout_rate` (float): Dropout 率
- `weight_decay` (float): 权重衰减
- `learning_rate` (float): 学习率
- `batch_size` (int): 批次大小
- `epochs` (int): 训练轮数
- `patience` (int): 早停耐心值
- `num_workers` (int): 数据加载工作进程数
- `seed` (int): 随机种子

**方法：**
- `predict_mus(ds, batch_size=None)` - 预测 μ₀ 和 μ₁

**返回：** tuple - (mu_0, mu_1)

#### `tarnet.TARNet`
TARNet 模型。

**参数：**
- `job_dir` (str): 工作目录
- `architecture` (str): 架构类型
- `dim_input` (int): 输入维度
- `dim_hidden` (int): 隐藏层维度
- `dim_output` (int): 输出维度
- `depth` (int): 网络深度
- `negative_slope` (float): LeakyReLU 负斜率
- `batch_norm` (bool): 是否使用批归一化
- `spectral_norm` (float): 谱归一化系数
- `dropout_rate` (float): Dropout 率
- `weight_decay` (float): 权重衰减
- `learning_rate` (float): 学习率
- `batch_size` (int): 批次大小
- `epochs` (int): 训练轮数
- `patience` (int): 早停耐心值
- `num_workers` (int): 数据加载工作进程数
- `seed` (int): 随机种子

**方法：**
- `predict_mus(ds, batch_size=None)` - 预测 μ₀ 和 μ₁

**返回：** tuple - (mu_0, mu_1)

#### `neural_network.NeuralNetwork`
神经网络模型。

**参数：** 类似 TARNet

**方法：**
- `predict_mean(ds, batch_size=None)` - 预测均值

**返回：** np.ndarray

### 数据集 API (`causal_bald.library.datasets`)

#### `ihdp.IHDP`
IHDP 数据集。

**参数：**
- `root` (str): 数据根目录
- `split` (str): 数据集划分（train/valid/test）
- `mode` (str): 模式（mu/pi）
- `seed` (int): 随机种子

**属性：**
- `x` (np.ndarray): 协变量
- `t` (np.ndarray): 处理指示变量
- `y` (np.ndarray): 结果变量
- `mu0` (np.ndarray): 对照组潜在结果
- `mu1` (np.ndarray): 处理组潜在结果
- `tau` (np.ndarray): 真实治疗效应
- `dim_input` (int): 输入维度

#### `synthetic.Synthetic`
合成数据集。

**参数：**
- `num_examples` (int): 样本数量
- `mode` (str): 模式（mu/pi）
- `beta` (float): β 参数
- `sigma_y` (float): 结果噪声标准差
- `bimodal` (bool): 是否使用双峰分布
- `seed` (int): 随机种子

**属性：** 类似 IHDP

#### `active_learning.ActiveLearningDataset`
主动学习数据集包装器。

**参数：**
- `dataset` (Dataset): 原始数据集
- `start_indices` (np.ndarray, optional): 初始索引

**属性：**
- `training_dataset` (Subset): 训练数据集
- `pool_dataset` (Subset): 池数据集
- `acquired_indices` (np.ndarray): 已获取索引

**方法：**
- `acquire(pool_indices)` - 从池中获取样本
- `is_empty()` - 检查池是否为空
- `get_random_pool_indices(size)` - 随机获取池索引

### 模块 API (`causal_bald.library.modules`)

#### `dense.NeuralNetwork`
全连接神经网络模块。

**参数：**
- `architecture` (str): 架构类型（resnet/linear）
- `dim_input` (int): 输入维度
- `dim_hidden` (int): 隐藏层维度
- `depth` (int): 网络深度
- `negative_slope` (float): LeakyReLU 负斜率
- `batch_norm` (bool): 是否使用批归一化
- `dropout_rate` (float): Dropout 率
- `spectral_norm` (float): 谱归一化系数
- `activate_output` (bool): 是否激活输出

#### `gaussian_process.VariationalGP`
变分高斯过程模块。

**参数：**
- `num_outputs` (int): 输出数量
- `initial_lengthscale` (float): 初始长度尺度
- `initial_inducing_points` (torch.Tensor): 初始诱导点
- `separate_inducing_points` (bool): 是否分离诱导点
- `kernel` (str): 核函数类型
- `ard` (bool): 是否使用 ARD
- `lengthscale_prior` (bool): 是否使用长度尺度先验

#### `variational.Categorical`
分类变分分布模块。

**参数：**
- `dim_input` (int): 输入维度
- `dim_output` (int): 输出维度

### 可视化 API (`causal_bald.library.plotting`)

#### `plotting.acquisition_clean(x_pool, t_pool, x_acquired, t_acquired, tau_true, tau_pred, domain, legend_title, file_path)`
绘制获取分布（清洁版）。

**参数：**
- `x_pool` (np.ndarray): 池协变量
- `t_pool` (np.ndarray): 池处理变量
- `x_acquired` (np.ndarray): 已获取协变量
- `t_acquired` (np.ndarray): 已获取处理变量
- `tau_true` (np.ndarray): 真实治疗效应
- `tau_pred` (np.ndarray): 预测治疗效应
- `domain` (np.ndarray): 域点
- `legend_title` (str): 图例标题
- `file_path` (Path): 保存路径

**返回：** None

#### `plotting.plot_convergence(x, y, y_err, x_label, y_label, marker_label, file_path)`
绘制收敛曲线。

**参数：**
- `x` (np.ndarray): x 轴数据
- `y` (np.ndarray): y 轴数据
- `y_err` (np.ndarray): y 轴误差
- `x_label` (str): x 轴标签
- `y_label` (str): y 轴标签
- `marker_label` (str): 标记标签
- `file_path` (Path): 保存路径

**返回：** None

## 工具函数 API (`causal_bald.application.workflows.utils`)

#### `utils.train_deep_kernel_gp(ds_train, ds_valid, job_dir, config, dim_input)`
训练 Deep Kernel GP 模型。

**参数：**
- `ds_train` (Dataset): 训练数据集
- `ds_valid` (Dataset): 验证数据集
- `job_dir` (Path): 工作目录
- `config` (dict): 配置字典
- `dim_input` (int): 输入维度

**返回：** None

#### `utils.train_ensemble(ds_train, ds_valid, job_dir, config, dim_input)`
训练集成模型。

**参数：** 同上

**返回：** None

#### `utils.predict_deep_kernel_gp(dataset, job_dir, config)`
使用 Deep Kernel GP 预测。

**参数：**
- `dataset` (Dataset): 数据集
- `job_dir` (Path): 工作目录
- `config` (dict): 配置字典

**返回：** tuple - (mu_0, mu_1)

#### `utils.predict_ensemble(dataset, job_dir, config)`
使用集成模型预测。

**参数：** 同上

**返回：** tuple - (mu_0, mu_1)

## 常量

### 获取函数字典
```python
acquisitions.FUNCTIONS = {
    "random": random,
    "tau": tau,
    "mu": mu,
    "rho": rho,
    "mu-rho": mu_rho,
    "pi": pi,
    "mu-pi": mu_pi,
    "sundin": sundin,
}
```

### 数据集字典
```python
datasets.DATASETS = {
    "ihdp": IHDP,
    "synthetic": Synthetic,
    "cmnist": HCMNIST,
    # ...
}
```

### 训练函数字典
```python
utils.TRAIN_FUNCTIONS = {
    "deep_kernel_gp": train_deep_kernel_gp,
    "ensemble": train_ensemble,
}
```

### 预测函数字典
```python
utils.PREDICT_FUNCTIONS = {
    "deep_kernel_gp": predict_deep_kernel_gp,
    "ensemble": predict_ensemble,
}
```

