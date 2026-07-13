# 3D-MMC 拓扑优化与 GNN 数据集生成

这是一个可运行、模块化的 3D Moving Morphable Component 项目，用于三维悬臂梁最小柔度拓扑优化，以及后续 GNN 数据集构建。

## 依赖安装

```bash
python -m pip install -r requirements.txt
```

依赖包括 `numpy`、`scipy`、`matplotlib`。当前版本不再使用无 SciPy 的替代优化路径。

## 方法概述

项目用多个三维 MMC 构件显式描述结构。每个构件包含 9 个设计变量：

```text
x0, y0, z0, L1, L2, L3, alpha, beta, gamma
```

构件 TDF 默认通过超椭球形式定义。也可以在 `config.py` 中设置 `component_shape = "box"`，此时使用高阶 p-norm 作为长方体/盒状构件的平滑近似，仍保留可导灵敏度。多构件 TDF 使用 KS 函数近似 `max` 聚合，再经平滑 Heaviside 映射为有限元单元密度。有限元模块使用规则 Hex8 单元，左端面固定，右端面中心施加向下集中力，目标函数为柔度 `F^T U`，约束为 `volume_fraction <= volfrac`。

优化默认使用 MMA。代码实现了柔度和体积分数对 MMC 参数的解析灵敏度，链式路径为：

```text
compliance -> element density -> Heaviside(TDF) -> KS aggregation -> MMC parameters
```

SLSQP 仍保留为可选优化器，并使用同一套解析梯度。

默认 `stop_on_convergence = False`，MMA 会执行完整 `max_iter`，便于生成固定长度的 GNN 迭代数据。若只关心优化收敛速度，可改为 `True`。

## 如何运行

```bash
cd project_3d_mmc
python main.py
```

输出：

- `dataset/iter_XXXX_graph.npz`: 每次迭代的 GNN 图数据。
- `results/*.npy`: 密度场和最终参数。
- `results/*.png`: 构件、密度切片和历史曲线。
- `results/summary.json`: 最终状态摘要。

程序优先写入项目内 `dataset/` 和 `results/`。如果当前沙箱或权限策略拒绝写入，入口程序会自动回退到可写临时目录，并在运行结束时打印实际输出路径。

## 修改规模

默认配置在 `config.py` 中：

```python
nelx = 20
nely = 6
nelz = 6
num_components = 8
max_iter = 20
```

构件形状可在同一文件中修改：

```python
component_shape = "superellipsoid"  # 默认
component_shape = "box"             # 高阶 p-norm 平滑盒状近似
```

更精细的实验可以改为：

```python
nelx = 40
nely = 12
nelz = 12
num_components = 16
max_iter = 100
```

三维有限元计算成本增长很快，建议先用小网格验证流程，再增加网格和构件数量。

## 如何导出 GNN 数据

默认 `save_graph=True`，每次优化迭代都会保存一个 `npz` 文件。图构建规则是：每个 MMC 构件为一个节点，构件距离接近或包围球近似重叠时建立双向边。

`node_features` 是训练用归一化特征，`node_features_raw` 保留原始物理量便于检查。

训练节点特征包含：

```text
x0/DL, y0/DW, z0/DH,
L1/max(DL,DW,DH), L2/max(DL,DW,DH), L3/max(DL,DW,DH),
sin/cos(alpha, beta, gamma),
component_volume/domain_volume,
active_flag,
compliance_grad_norm,
volume_grad_norm,
delta_params_norm,
delta_center_norm,
delta_size_norm,
delta_angle_norm
```

边特征：

```text
normalized_distance,
normalized_dx, normalized_dy, normalized_dz,
alignment_trace,
axis1_dot, axis2_dot, axis3_dot,
overlap_flag,
contact_score
```

每个图还保存 `global_features`：

```text
iteration/max_iter,
compliance,
volume_fraction,
volfrac,
load_y/DW,
load_z/DH,
num_components/max_components_for_dataset
```

## 批量生成 GNN 轨迹数据集

单次 `main.py` 只生成一条优化轨迹。用于训练 GNN 时，建议使用批量脚本生成多条随机轨迹：

```bash
python generate_dataset.py --num-trajectories 200 --max-iter 20 --min-components 6 --max-components 12 --output-dir dataset --results-dir results/batch
```

脚本会随机化：

- `seed`
- `num_components`
- 初始 MMC 构件中心、尺寸、角度
- 右端面载荷位置 `load_y, load_z`

图文件命名格式：

```text
traj_0007_iter_0012_graph.npz
```

同时生成：

```text
dataset/dataset_index.csv
```

其中记录每条 trajectory 的 seed、构件数量、载荷位置、最终柔度和最终体积分数。

## 如何使用本项目生成 GNN 步长预测数据集

本项目在 `optimizer.py` 中提供了 GNN 步长缩放接口：

```python
apply_gnn_step_scale(params_old, params_new, eta)
```

传统更新方向仍由 MMA 或 SLSQP 给出。每次迭代得到 `params_new` 后，程序计算：

```python
params_eta = params_old + eta * (params_new - params_old)
```

默认候选步长为：

```python
eta_candidates = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
```

程序会分别评估每个候选步长的 `compliance_eta` 和 `volume_fraction_eta`，再选择满足体积约束且柔度最低的 `eta` 作为 `eta_label`。若所有候选都违反体积约束，则 `eta_label = 0.0` 并设置失败标记。

每个 `dataset/iter_XXXX_graph.npz` 包含：

```text
node_features
node_features_raw
edge_index
edge_attr
global_features
compliance
volume_fraction
iteration
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
trajectory_id
seed
num_components
load_y
load_z
feature_schema_version
load_point
node_load_distance_norm
node_load_prior
edge_load_prior
eta_node_label
eta_node_label_index
trust_actual_delta
trust_predicted_delta
trust_bias
component_strain_energy_norm
```

后续训练 GNN 时可使用：

- 输入：当前 MMC 构件图，即 `node_features`, `edge_index`, `edge_attr`, `edge_load_prior`
- 全局输入：`global_features`
- 主输出：组件级连续步长 `eta_node_label`
- 图级辅助输出：连续最优步长 `eta_label`
- 辅助输出：`eta_label_index`, `eta_failure_flag`, `response_targets`

如果所有候选步长都违反体积约束：

- `eta_label = 0.0`
- `eta_label_index = -1`
- `eta_failure_flag = 1`

项目提供 PyTorch Geometric 数据加载器：

```python
from pyg_dataset import MMCStepDataset, create_dataloader

dataset = MMCStepDataset("dataset")
loader = create_dataloader("dataset", batch_size=8)
```

训练物理偏置 GAT：

```bash
python train_gnn_step.py --dataset-dir dataset --epochs 50 --batch-size 8
```

MAML 风格元学习预训练：

```bash
python meta_learning.py --dataset-dir dataset --epochs 20
```

按轨迹划分训练/验证/测试集，避免同一条轨迹的相邻迭代步泄漏到不同集合：

```bash
python split_by_trajectory.py --index dataset/dataset_index.csv --output-dir dataset/splits
```

加载指定 split：

```python
train_set = MMCStepDataset("dataset", split_file="dataset/splits/train_trajectories.txt")
val_set = MMCStepDataset("dataset", split_file="dataset/splits/val_trajectories.txt")
test_set = MMCStepDataset("dataset", split_file="dataset/splits/test_trajectories.txt")
```

## 当前限制

- MMA 实现面向本项目的单体积约束问题，不是通用多约束工业 MMA 包。
- 解析灵敏度覆盖柔度、体积分数、TDF、KS、Heaviside 和 MMC 参数，但尚未加入二阶信息。
- `component_shape = "box"` 使用高阶 p-norm 平滑近似，不是不可导的精确布尔长方体边界。
- 高分辨率 3D 问题计算量较大。
- 图边规则是几何近似，后续可加入载荷路径识别和更丰富的物理特征。

## 后续扩展

- 加入更多图标签，如下一步真实柔度、约束违反程度、灵敏度统计量。
- 用训练好的 GAT 预测组件级 `eta_node_label`，再通过组件级步长缩放调整传统优化器给出的更新步长。
- 在线阶段使用 `online_meta_controller.py` 中的 trust-bias 触发、稀疏真标签微调和安全回退阻尼。
