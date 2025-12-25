## 反事实治疗与重叠性假设的深入分析
### 一、反事实治疗（Counterfactual Treatment）的定义

#### 1.1 基本概念

在因果推断中，每个个体有两个潜在结果（Potential Outcomes）：
- $Y_i(1)$：个体 $i$ 接受治疗时的结果
- $Y_i(0)$：个体 $i$ 接受对照时的结果

但只能观测到一个事实结果（Factual Outcome）：
- 如果 $\mathrm{T}_i = 1$，观测到 $Y_i(1)$，$Y_i(0)$ 是反事实结果
- 如果 $\mathrm{T}_i = 0$，观测到 $Y_i(0)$，$Y_i(1)$ 是反事实结果

反事实治疗（Counterfactual Treatment）：
- 如果观测到 $\mathrm{T}_i = \mathrm{t}$，则 $\mathrm{t}' \neq \mathrm{t}$ 是反事实治疗
- 例如：$\mathrm{t} = 1$ 时，$\mathrm{t}' = 0$ 是反事实治疗

#### 1.2 数学表达

对于观测到的数据点 $(\mathbf{x}, \mathrm{t})$：
- 事实治疗：$\mathrm{t}$（已观测）
- 反事实治疗：$\mathrm{t}'$，满足 $\mathrm{t}' \neq \mathrm{t}$

在二分类情况下（$\mathrm{T} \in \{0, 1\}$）：
- 如果 $\mathrm{t} = 1$，则 $\mathrm{t}' = 0$
- 如果 $\mathrm{t} = 0$，则 $\mathrm{t}' = 1$

---

### 二、为什么 $1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 是反事实治疗的概率

#### 2.1 倾向得分的定义回顾

倾向得分定义为：

$$
\widehat{\pi}_{\mathrm{t}}(\mathbf{x}) = \widehat{P}(\mathrm{T} = \mathrm{t} \mid \mathbf{X} = \mathbf{x})
$$

即：在协变量 $\mathbf{x}$ 下，观察到治疗 $\mathrm{t}$ 的概率。

#### 2.2 概率的互补性

对于二分类治疗（$\mathrm{T} \in \{0, 1\}$），有：

$$
P(\mathrm{T} = 1 \mid \mathbf{X} = \mathbf{x}) + P(\mathrm{T} = 0 \mid \mathbf{X} = \mathbf{x}) = 1
$$

因此：

$$
P(\mathrm{T} = 0 \mid \mathbf{X} = \mathbf{x}) = 1 - P(\mathrm{T} = 1 \mid \mathbf{X} = \mathbf{x})
$$

#### 2.3 反事实治疗概率的推导

对于观测到的数据点 $(\mathbf{x}, \mathrm{t})$：

情况1：如果 $\mathrm{t} = 1$（观测到治疗组）
- $\widehat{\pi}_1(\mathbf{x}) = \widehat{P}(\mathrm{T} = 1 \mid \mathbf{X} = \mathbf{x})$
- 反事实治疗是 $\mathrm{t}' = 0$
- 反事实治疗的概率：$\widehat{P}(\mathrm{T} = 0 \mid \mathbf{X} = \mathbf{x}) = 1 - \widehat{\pi}_1(\mathbf{x})$

情况2：如果 $\mathrm{t} = 0$（观测到对照组）
- $\widehat{\pi}_0(\mathbf{x}) = \widehat{P}(\mathrm{T} = 0 \mid \mathbf{X} = \mathbf{x})$
- 反事实治疗是 $\mathrm{t}' = 1$
- 反事实治疗的概率：$\widehat{P}(\mathrm{T} = 1 \mid \mathbf{X} = \mathbf{x}) = 1 - \widehat{\pi}_0(\mathbf{x})$

#### 2.4 统一表达

对于任意观测到的 $(\mathbf{x}, \mathrm{t})$，反事实治疗 $\mathrm{t}' \neq \mathrm{t}$ 的概率为：

$$
\widehat{P}(\mathrm{T} = \mathrm{t}' \mid \mathbf{X} = \mathbf{x}) = 1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})
$$

证明：

$$
\begin{align}
\widehat{P}(\mathrm{T} = \mathrm{t}' \mid \mathbf{X} = \mathbf{x}) &= 1 - \widehat{P}(\mathrm{T} = \mathrm{t} \mid \mathbf{X} = \mathbf{x}) \\
&= 1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})
\end{align}
$$

因此，$1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 确实是在协变量 $\mathbf{x}$ 下观察到反事实治疗 $\mathrm{t}' \neq \mathrm{t}$ 的概率。

---

### 三、为什么要选择反事实观察概率较高的点

#### 3.1 重叠性假设的重要性

重叠性假设（Overlap Assumption）：

$$
0 < P(\mathrm{T} = \mathrm{t} \mid \mathbf{X} = \mathbf{x}) < 1, \quad \forall \mathbf{x}, \mathrm{t}
$$

含义：
- 对于任何协变量值 $\mathbf{x}$ 和治疗 $\mathrm{t}$，都有非零概率被分配到该治疗
- 确保每个协变量水平下，治疗组和对照组都有样本

#### 3.2 反事实观察概率与重叠性的关系

当 $1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 较大时：
- 在协变量 $\mathbf{x}$ 下，观察到反事实治疗的概率较高
- 意味着该协变量水平下，治疗组和对照组都可能出现
- 重叠性较好

当 $1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 较小时：
- 在协变量 $\mathbf{x}$ 下，观察到反事实治疗的概率较低
- 意味着该协变量水平下，主要只出现一种治疗
- 重叠性较差

#### 3.3 数学证明：反事实概率与重叠性的等价性

定理：对于观测到的 $(\mathbf{x}, \mathrm{t})$，反事实观察概率 $1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 较大当且仅当重叠性较好。

证明：

重叠性较好的定义：$\widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 不接近 0 或 1，即：

$$
\epsilon < \widehat{\pi}_{\mathrm{t}}(\mathbf{x}) < 1 - \epsilon, \quad \text{对于某个小的 } \epsilon > 0
$$

这等价于：

$$
\epsilon < 1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x}) < 1 - \epsilon
$$

即反事实观察概率 $1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 也远离 0 和 1。

因此：
- $1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 较大 $\Leftrightarrow$ $\widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 不接近 0 或 1 $\Leftrightarrow$ 重叠性较好

---

### 四、如何促进治疗组和对照组的重叠

#### 4.1 主动学习中的重叠性促进机制

通过选择 $1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 较大的点，我们：

1. 优先选择重叠性好的区域
   - 这些区域中，治疗组和对照组都可能出现
   - 有助于在训练集中同时包含两种治疗的数据

2. 避免选择重叠性差的区域
   - 这些区域中，主要只出现一种治疗
   - 无法有效估计治疗效果

#### 4.2 数学分析：训练集重叠性的提升

设训练集为 $\mathcal{D}_{\text{train}} = \{(\mathbf{x}_i, \mathrm{t}_i)\}_{i=1}^{n}$。

定义：训练集在协变量 $\mathbf{x}$ 处的重叠性为：

$$
\text{Overlap}(\mathbf{x}, \mathcal{D}_{\text{train}}) = \min\{P(\mathrm{T} = 1 \mid \mathbf{X} = \mathbf{x}, \mathcal{D}_{\text{train}}), P(\mathrm{T} = 0 \mid \mathbf{X} = \mathbf{x}, \mathcal{D}_{\text{train}})\}
$$

如果选择 $1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 较大的点 $(\mathbf{x}, \mathrm{t})$：

- 该点更可能同时出现在治疗组和对照组中
- 添加到训练集后，在 $\mathbf{x}$ 附近的重叠性提升

#### 4.3 具体例子

假设有两个候选点：

点1：$(\mathbf{x}_1, \mathrm{t}_1 = 1)$
- $\widehat{\pi}_1(\mathbf{x}_1) = 0.3$（接受治疗的概率较低）
- $1 - \widehat{\pi}_1(\mathbf{x}_1) = 0.7$（反事实观察概率较高）
- 解释：在 $\mathbf{x}_1$ 下，更可能出现在对照组（70%），但也可能出现在治疗组（30%）
- 重叠性：较好

点2：$(\mathbf{x}_2, \mathrm{t}_2 = 1)$
- $\widehat{\pi}_1(\mathbf{x}_2) = 0.95$（接受治疗的概率很高）
- $1 - \widehat{\pi}_1(\mathbf{x}_2) = 0.05$（反事实观察概率很低）
- 解释：在 $\mathbf{x}_2$ 下，几乎只出现在治疗组（95%），很少出现在对照组（5%）
- 重叠性：较差

选择策略：
- 优先选择点1（$1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x}) = 0.7$）
- 避免选择点2（$1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x}) = 0.05$）

结果：
- 训练集中在 $\mathbf{x}_1$ 附近同时包含治疗组和对照组的数据
- 训练集中在 $\mathbf{x}_2$ 附近主要只有治疗组的数据
- 整体重叠性提升

---

### 五、如何满足重叠性假设

#### 5.1 重叠性假设的数学表达

重叠性假设（Overlap Assumption）：

$$
0 < P(\mathrm{T} = \mathrm{t} \mid \mathbf{X} = \mathbf{x}) < 1, \quad \forall \mathbf{x}, \mathrm{t}
$$

等价地，对于任意 $\mathbf{x}$ 和 $\mathrm{t}$：

$$
0 < \widehat{\pi}_{\mathrm{t}}(\mathbf{x}) < 1
$$

#### 5.2 反事实倾向获取如何促进重叠性假设

通过选择 $1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 较大的点，我们间接确保：

$$
\widehat{\pi}_{\mathrm{t}}(\mathbf{x}) < 1 - \delta, \quad \text{对于某个 } \delta > 0
$$

这等价于：

$$
\delta < 1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x}) = \widehat{\pi}_{\mathrm{t}'}(\mathbf{x})
$$

因此，对于反事实治疗 $\mathrm{t}'$：

$$
\widehat{\pi}_{\mathrm{t}'}(\mathbf{x}) > \delta > 0
$$

同时，由于 $\widehat{\pi}_{\mathrm{t}}(\mathbf{x}) < 1 - \delta < 1$，我们有：

$$
0 < \delta < \widehat{\pi}_{\mathrm{t}}(\mathbf{x}) < 1 - \delta < 1
$$

因此，重叠性假设得到满足。

#### 5.3 渐进性证明

定理：如果持续选择 $1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x}) > \epsilon$ 的点（其中 $\epsilon > 0$ 是固定阈值），则训练集会渐进满足重叠性假设。

证明思路：

1. 初始状态：训练集可能在某些区域重叠性较差

2. 选择策略：每次选择 $1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x}) > \epsilon$ 的点

3. 重叠性提升：
   - 选择的点满足：$\widehat{\pi}_{\mathrm{t}}(\mathbf{x}) < 1 - \epsilon$
   - 同时满足：$\widehat{\pi}_{\mathrm{t}'}(\mathbf{x}) = 1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x}) > \epsilon$
   - 因此：$\epsilon < \widehat{\pi}_{\mathrm{t}}(\mathbf{x}) < 1 - \epsilon$ 和 $\epsilon < \widehat{\pi}_{\mathrm{t}'}(\mathbf{x}) < 1 - \epsilon$

4. 渐进结果：
   - 随着训练集增大，覆盖的协变量空间更广
   - 在覆盖的区域中，重叠性假设得到满足
   - 训练集渐进满足重叠性假设

#### 5.4 实际效果

通过反事实倾向获取：

1. 训练集分布更平衡
   - 治疗组和对照组在协变量空间中的分布更相似
   - 减少选择偏差

2. 治疗效果估计更准确
   - 在重叠性好的区域，可以更准确地估计 CATE
   - 减少对模型设定的敏感性

3. 减少外推
   - 避免在重叠性差的区域进行外推
   - 提高估计的稳健性

---

### 六、综合例子

#### 场景设置

研究新药对高血压患者的治疗效果。

**协变量空间：**
- $\mathbf{x}_1$：年龄=50，BMI=25，基础血压=150（中等风险）
- $\mathbf{x}_2$：年龄=70，BMI=30，基础血压=180（高风险）
- $\mathbf{x}_3$：年龄=30，BMI=22，基础血压=120（低风险）

**倾向得分估计：**
- $\widehat{\pi}_1(\mathbf{x}_1) = 0.5$（中等概率接受新药）
- $\widehat{\pi}_1(\mathbf{x}_2) = 0.9$（高概率接受新药）
- $\widehat{\pi}_1(\mathbf{x}_3) = 0.1$（低概率接受新药）

**反事实倾向获取分数：**
- $1 - \widehat{\pi}_1(\mathbf{x}_1) = 0.5$
- $1 - \widehat{\pi}_1(\mathbf{x}_2) = 0.1$
- $1 - \widehat{\pi}_1(\mathbf{x}_3) = 0.9$

**选择策略：**
- 优先选择 $\mathbf{x}_1$ 和 $\mathbf{x}_3$（反事实概率较高）
- 避免选择 $\mathbf{x}_2$（反事实概率较低）

**结果：**
- 训练集中在 $\mathbf{x}_1$ 和 $\mathbf{x}_3$ 附近同时包含治疗组和对照组的数据
- 重叠性假设在这些区域得到满足
- 可以更准确地估计这些区域的治疗效果

---

### 七、总结

1. 反事实治疗：对于观测到的 $(\mathbf{x}, \mathrm{t})$，$\mathrm{t}' \neq \mathrm{t}$ 是反事实治疗。

2. 反事实概率：$1 - \widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 是在协变量 $\mathbf{x}$ 下观察到反事实治疗的概率（由概率互补性得到）。

3. 选择高反事实概率的点：这些点所在区域重叠性更好，有助于提升训练集的重叠性。

4. 促进重叠：通过优先选择重叠性好的区域，训练集中治疗组和对照组的分布更平衡。

5. 满足重叠性假设：通过确保 $\widehat{\pi}_{\mathrm{t}}(\mathbf{x})$ 不接近 0 或 1，渐进满足重叠性假设。

反事实倾向获取函数通过选择反事实观察概率较高的点，系统性地促进重叠性，从而提升因果效应估计的准确性和稳健性。