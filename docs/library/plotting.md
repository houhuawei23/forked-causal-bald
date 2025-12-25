# 可视化函数文档

## 概述

`causal_bald.library.plotting` 模块提供了用于结果可视化和分析的功能，包括数据集可视化、获取分布可视化、收敛曲线等。

## 主要函数

### `dataset(ds, legend_title, file_path)`

绘制数据集可视化（适用于合成数据集）。

**参数：**
- `ds` (Dataset): 数据集对象
- `legend_title` (str, 可选): 图例标题
- `file_path` (Path): 保存路径

**功能：**
- 绘制协变量分布（处理组和对照组）
- 绘制结果散点图
- 显示重叠区域

### `mnist(ds, legend_title, file_path)`

绘制 CMNIST 数据集可视化。

**参数：** 同上

**功能：**
- 绘制协变量嵌入分布
- 显示 MNIST 图像标记
- 展示处理组和对照组的分布差异

### `acquisition_clean(x_pool, t_pool, x_acquired, t_acquired, tau_true, tau_pred, domain, legend_title, file_path)`

绘制获取分布（清洁版）。

**参数：**
- `x_pool` (np.ndarray): 池协变量
- `t_pool` (np.ndarray): 池处理变量
- `x_acquired` (np.ndarray): 已获取协变量
- `t_acquired` (np.ndarray): 已获取处理变量
- `tau_true` (np.ndarray): 真实治疗效应
- `tau_pred` (np.ndarray): 预测治疗效应（多个样本）
- `domain` (np.ndarray): 域点（用于绘制函数）
- `legend_title` (str, 可选): 图例标题
- `file_path` (Path): 保存路径

**功能：**
- 三子图布局：
  1. 池数据分布
  2. 已获取数据分布
  3. 真实和预测治疗效应

### `acquisition_hist(...)`

绘制获取分布（直方图版），参数同上。

### `functions(x, t, domain, tau_true, tau_mean, legend_title, legend_loc, file_path)`

绘制函数对比。

**参数：**
- `x` (np.ndarray): 协变量
- `t` (np.ndarray): 处理变量
- `domain` (np.ndarray): 域点
- `tau_true` (np.ndarray): 真实治疗效应
- `tau_mean` (np.ndarray): 预测治疗效应均值
- `legend_title` (str, 可选): 图例标题
- `legend_loc` (tuple, 可选): 图例位置
- `file_path` (Path): 保存路径

### `acquisition(x, t, tau_true, bald, legend_title, legend_loc, file_path)`

绘制获取分数分布。

**参数：**
- `x` (np.ndarray): 协变量
- `t` (np.ndarray): 处理变量
- `tau_true` (np.ndarray): 真实治疗效应
- `bald` (np.ndarray): BALD 分数
- `legend_title` (str, 可选): 图例标题
- `legend_loc` (tuple, 可选): 图例位置
- `file_path` (Path): 保存路径

### `errorbar(x, y, y_err, x_label, y_label, marker_label, x_pad, y_pad, legend_loc, file_path)`

绘制误差条图。

**参数：**
- `x` (np.ndarray): x 轴数据
- `y` (np.ndarray): y 轴数据
- `y_err` (np.ndarray): y 轴误差
- `x_label` (str): x 轴标签
- `y_label` (str): y 轴标签
- `marker_label` (str, 可选): 标记标签
- `x_pad` (int): x 轴刻度内边距
- `y_pad` (int): y 轴刻度内边距
- `legend_loc` (str): 图例位置
- `file_path` (Path): 保存路径

## 样式配置

模块使用 seaborn 和 matplotlib 进行可视化：

```python
sns.set(style="whitegrid", palette="colorblind")
```

## 使用示例

### 绘制数据集
```python
from causal_bald.library import plotting, datasets

ds = datasets.Synthetic(num_examples=1000, mode="mu", seed=0)
plotting.dataset(
    ds=ds,
    file_path=Path("synthetic_dataset.png"),
)
```

### 绘制获取分布
```python
plotting.acquisition_clean(
    x_pool=x_pool,
    t_pool=t_pool,
    x_acquired=x_acquired,
    t_acquired=t_acquired,
    tau_true=tau_true,
    tau_pred=tau_pred,
    domain=domain,
    legend_title=f"Acquired: {num_acquired}",
    file_path=Path("acquisition.png"),
)
```

## 相关文档

- [评估工作流](../../application/workflows/evaluation.md)
- [库层概览](overview.md)

