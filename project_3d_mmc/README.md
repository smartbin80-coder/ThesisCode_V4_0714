# 3D-MMC Cantilever Topology Optimization with GNN Step Prediction

本项目用于三维悬臂梁的 Moving Morphable Component (MMC) 拓扑优化，并为后续图神经网络训练导出逐迭代图数据。当前技术路线不是用神经网络替代有限元或灵敏度分析，而是在传统 MMA 优化方向给定后，用 GNN 学习候选步长和组件级步长缩放，使 AI 只影响“走多远”，不改变“往哪里走”。

## 当前默认算例

默认设计域和离散网格：

```python
DL = 60.0
DW = 4.0
DH = 20.0
nelx = 60
nely = 4
nelz = 20
```

默认 MMC 和优化参数：

```python
num_components = 24
min_components_for_dataset = 24
max_components_for_dataset = 24
volfrac = 0.4
max_iter = 20
eta_candidates = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
save_density = False
save_process_plots = True
```

边界条件为悬臂梁工况：左端面 `x = 0` 全固定，右端面 `x = DL` 附近施加 `-Z` 方向集中力。默认载荷点位于右端面中心，即：

```python
load_point = [60.0, 2.0, 10.0]
```

批量生成数据时，`load_y` 和 `load_z` 会在设计域中随机扰动。

## 安装依赖

```bash
python -m pip install -r requirements.txt
```

主要依赖包括：

- `numpy`
- `scipy`
- `matplotlib`
- `torch`
- `torch-geometric`
- `scikit-image`

其中 `scikit-image` 用于从 MMC 全局隐式函数中提取 `phi = 0` 等值面。

## 快速运行

单条默认优化：

```bash
cd project_3d_mmc
python -B main.py
```

生成一条 50 步随机轨迹：

```bash
python -B generate_dataset.py --num-trajectories 1 --max-iter 50 --output-dir D:\codex\workspace_code_v2\dataset_domain60_random_1x50 --results-dir D:\codex\workspace_code_v2\results_domain60_random_1x50
```

生成 5 条 50 步轨迹：

```bash
python -B generate_dataset.py --num-trajectories 5 --max-iter 50 --output-dir D:\codex\workspace_code_v2\dataset_domain60_5x50 --results-dir D:\codex\workspace_code_v2\results_domain60_5x50
```

## 方法概述

每个 MMC 组件由 9 个设计变量描述：

```text
x0, y0, z0, L1, L2, L3, alpha, beta, gamma
```

组件默认采用超椭球 TDF，多个组件通过 KS 聚合近似并集，再经平滑 Heaviside 映射为有限元单元密度。有限元模块使用规则 Hex8 单元，目标函数为柔度：

```text
compliance = F^T U
```

约束为：

```text
volume_fraction <= volfrac
```

优化器默认使用 MMA。代码保留解析灵敏度链路：

```text
compliance -> density -> Heaviside(TDF) -> KS aggregation -> MMC parameters
```

## 几何可行域约束

当前代码对每个旋转后的 MMC 组件做严格设计域投影。投影逻辑基于旋转后 AABB 半宽：

```python
half_extent = abs(R) @ [L1, L2, L3]
```

并保证：

```text
center - half_extent >= [0, 0, 0]
center + half_extent <= [DL, DW, DH]
```

这避免了组件中心仍在域内、但旋转后实体越出 `60 x 4 x 20` 可行域的问题。

## 初始化策略

当前批量数据生成使用 24 个随机分散、严格域内可行的 MMC 组件。近期 1 条 50 步随机分散实验表明，完全随机撒点通常不能形成从固定端到载荷端的连续传力路径，最终等值面仍呈分散块状。因此，正式训练数据建议采用“悬臂梁专用随机初始化”：保留随机性，但需要保证固定端、载荷端和中间桥接区域有基本覆盖。

## 过程图与论文式等值面

每条轨迹会在开始、中间和结束三个节点保存过程图：

```text
initial_components_iter_0000.png
initial_isosurface_iter_0000.png
middle_components_iter_XXXX.png
middle_isosurface_iter_XXXX.png
final_components_iter_XXXX.png
final_isosurface_iter_XXXX.png
```

两类图含义不同：

- `*_components_*`：逐组件调试图，直接显示每个 MMC 组件表面。
- `*_isosurface_*`：论文式拓扑图，从 KS 聚合后的全局隐式边界 `phi = 0` 提取等值面。

默认等值面采样分辨率：

```python
isosurface_resolution = (240, 24, 80)
```

## GNN 数据导出

每次迭代会导出一个 `*_graph.npz` 图样本。节点对应 MMC 组件，边由几何接近、包围球重叠和接触评分建立。核心字段包括：

```text
node_features
node_features_raw
edge_index
edge_attr
global_features
params
delta_params
eta_candidates
eta_label
eta_label_index
eta_feasible_mask
eta_failure_flag
compliance_candidates
volume_candidates
response_targets
connected_to_load
spanning_ratio
largest_component_ratio
```

`eta_label` 来自真实 FEM 候选步长评估。程序会分别评估 6 个候选步长：

```python
[0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
```

若存在满足体积分数约束的候选，则选择柔度最低者作为标签；若都不可行，则记录失败标志，避免把失败样本错误映射到某个分类标签。

## 训练入口

按轨迹划分训练、验证、测试集：

```bash
python -B split_by_trajectory.py --index dataset/dataset_index.csv --output-dir dataset/splits
```

训练物理偏置 GAT：

```bash
python -B train_gnn_step.py --dataset-dir dataset --split-file dataset/splits/train_trajectories.txt --val-split-file dataset/splits/val_trajectories.txt --epochs 50 --batch-size 8
```

运行 MAML 风格元学习：

```bash
python -B meta_learning.py --dataset-dir dataset --epochs 20
```

在线单步测试：

```bash
python -B test_online_step.py --dataset-dir dataset --results-dir results/online_step
```

`test_online_step.py` 默认优先加载 `results_debug/gat_maml_model.pt`，若不存在则回退到 `results_debug/gat_step_model.pt`。

## 当前已验证内容

- 6 个 eta 候选步长链路已打通。
- `test_online_step.py` 可从项目目录或仓库根目录运行。
- GAT checkpoint 与 MAML checkpoint 路径回退逻辑已实现。
- 旋转后 MMC 组件严格限制在 `60 x 4 x 20` 设计域内。
- 隐式等值面绘图可输出论文式红色拓扑表面。
- 连接性诊断字段 `connected_to_load`, `spanning_ratio`, `largest_component_ratio` 已写入图样本。

## 当前限制

- 组件级标签目前仍主要来自图级 eta 标签广播，尚未实现真正逐组件差异化标签。
- 完全随机分散初始化不能可靠形成悬臂梁连续传力路径，正式训练前需要改为悬臂梁专用随机初始化。
- 当前 MAML 为 first-order 风格实现，不是完整二阶 MAML。
- GAT 尚未完全嵌入 MMC 主优化闭环做长期在线验证。
- 高分辨率 3D FEM 计算成本较高，建议逐步扩大轨迹数量和迭代步数。

## 建议下一步

1. 设计悬臂梁专用随机初始化，确保初始结构至少存在固定端到载荷端的可行连通路径。
2. 用 `20-50` 条短轨迹验证初始化稳定性、连通性和体积约束。
3. 生成 `100-500` 条正式训练轨迹，并按轨迹划分训练/验证/测试集。
4. 训练 GAT 与 MAML 模型，比较普通监督训练、元学习预训练和在线微调效果。
5. 将 GAT 步长预测接入主优化闭环，进行真实收敛速度、柔度和稳定性对比。
