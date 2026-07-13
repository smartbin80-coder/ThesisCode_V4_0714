# 3D-MMC 项目代码导读与维护说明

本文档面向后续维护、复现实验和扩展 GNN 数据集生成流程，按代码模块说明每段功能的职责、输入输出和修改入口。

## 1. 项目整体流程

项目主线是：

```text
读取配置
  -> 生成三维 Hex8 网格
  -> 定义悬臂梁边界条件和载荷
  -> 初始化 MMC 构件
  -> 根据 MMC TDF 生成密度场
  -> 有限元求解位移和柔度
  -> 计算解析灵敏度
  -> MMA/SLSQP 更新 MMC 参数
  -> 每次迭代导出 GNN 图数据
```

核心入口有两个：

- `main.py`: 单条优化轨迹运行入口，适合调试和可视化。
- `generate_dataset.py`: 批量生成多条随机轨迹，适合生成 GNN 训练数据。

## 2. `config.py`: 全局配置

`config.py` 定义 `Config` 数据类和默认实例 `config`。

主要配置项：

- 设计域尺寸：`DL`, `DW`, `DH`
- 网格规模：`nelx`, `nely`, `nelz`
- 材料参数：`E0`, `Emin`, `nu`
- 体积分数约束：`volfrac`
- MMC 构件数和 TDF 参数：`num_components`, `p_norm`, `ks_rho`
- 构件形状：`component_shape`
  - `"superellipsoid"`: 默认超椭球近似
  - `"box"`: 高阶 p-norm 平滑盒状近似
- 优化参数：`max_iter`, `optimizer_type`, `stop_on_convergence`
- GNN 数据参数：`eta_candidates`, `save_graph`
- 批量数据参数：`min_components_for_dataset`, `max_components_for_dataset`
- 载荷位置：`load_y`, `load_z`
- 输出目录：`dataset_dir`, `results_dir`

常见修改：

- 想提高有限元精度：改 `nelx/nely/nelz`
- 想增加 MMC 复杂度：改 `num_components`
- 想让 MMA 提前停止：改 `stop_on_convergence=True`
- 想切换盒状构件：改 `component_shape="box"`

## 3. `main.py`: 单次运行入口

`main.py` 用于运行一条默认优化轨迹。

主要函数：

- `prepare_output_dirs(cfg)`
  - 检查 `dataset_dir` 和 `results_dir` 是否可写。
  - 如果当前沙箱或权限策略拒绝写入，会自动回退到可写临时目录。

- `main()`
  - 设置随机种子。
  - 调用 `generate_hex_mesh()` 生成有限元网格。
  - 调用 `define_cantilever_boundary_conditions()` 生成固定自由度和载荷向量。
  - 调用 `create_initial_components()` 生成默认 MMC 构件。
  - 初始化 `MMCOptimizer3D` 并运行优化。
  - 保存初始/最终构件图、密度切片、历史曲线。

维护建议：

- `main.py` 应保持简单，适合单次调试。
- 批量数据生成不要写进 `main.py`，应放在 `generate_dataset.py`。

## 4. `generate_dataset.py`: 批量轨迹生成

该文件用于生成 GNN 训练数据集，是解决“样本数量不足”的核心脚本。

主要函数：

- `writable_dir(path, fallback_name)`
  - 检查输出目录是否可写。
  - 不可写时回退到临时目录。

- `build_config_for_trajectory(args, trajectory_index)`
  - 为每条轨迹复制一份独立配置。
  - 随机化：
    - `seed`
    - `num_components`
    - `load_y`, `load_z`
    - `trajectory_id`

- `run_single_trajectory(cfg)`
  - 根据当前轨迹配置生成网格、边界、随机 MMC 构件。
  - 运行 `MMCOptimizer3D`。
  - 返回该轨迹的元数据。

- `write_index(index_path, rows)`
  - 写出 `dataset_index.csv`。
  - 后续按轨迹划分训练/验证/测试集依赖该文件。

典型命令：

```bash
python generate_dataset.py --num-trajectories 200 --max-iter 20 --min-components 6 --max-components 12
```

输出文件命名：

```text
traj_0000_iter_0000_graph.npz
traj_0000_iter_0001_graph.npz
...
```

## 5. `mmc3d_components.py`: MMC 几何构件

该文件定义 3D MMC 构件的几何参数和基本操作。

核心类：

### `MMCComponent3D`

每个构件有 9 个设计变量：

```text
x0, y0, z0,
L1, L2, L3,
alpha, beta, gamma
```

主要方法：

- `get_params()`
  - 返回 9 维设计变量。

- `set_params(params)`
  - 用 9 维变量更新构件。
  - 会限制 `L1/L2/L3` 不小于极小正数，避免除零。

- `rotation_matrix()`
  - 根据欧拉角构造三维旋转矩阵。

- `global_to_local(points)`
  - 将全局坐标点转换到构件局部坐标。

- `tdf(points, p=6)`
  - 计算构件的拓扑描述函数 TDF。
  - 正值表示点在构件内部，负值表示外部。

- `volume()`
  - 返回构件近似体积特征。

- `direction_vectors()`
  - 返回三个主方向向量，用于构造边特征中的方向对齐信息。

- `as_node_feature()`
  - 返回原始节点特征。

辅助函数：

- `create_initial_components(config)`
  - 为单次默认运行生成规则初始构件。

- `create_random_components(config, rng)`
  - 为批量数据生成随机初始构件。

- `params_to_components(params, template_components)`
  - 将扁平参数数组恢复成构件列表。

- `components_to_params(components)`
  - 将构件列表打平成优化变量。

- `get_bounds(components, config)`
  - 为优化器生成每个设计变量的上下界。

## 6. `tdf.py`: TDF 聚合与解析灵敏度

该文件负责从 MMC 构件生成全局 TDF、密度场，并计算对设计变量的解析导数。

主要函数：

- `compute_component_tdfs(components, points, p)`
  - 计算每个构件在所有节点上的 TDF。
  - 输出形状为 `[num_components, num_points]`。

- `ks_aggregate(phi_all, ks_rho)`
  - 使用 KS 函数近似 `max(phi_i)`。
  - 做了数值稳定处理，避免指数溢出。

- `compute_global_tdf(components, points, p, ks_rho)`
  - 计算多构件聚合后的全局 TDF。

- `compute_element_density_from_tdf(phi_nodes, elements, heaviside_params)`
  - 先用 Heaviside 将节点 TDF 转成节点密度。
  - 再对每个 Hex8 单元的 8 个节点取平均，得到单元密度。

- `compute_global_tdf_with_sensitivities(...)`
  - 计算全局 TDF 以及对所有 MMC 参数的解析导数。
  - 包括中心、尺寸、旋转角的链式导数。

- `compute_density_with_sensitivities(...)`
  - 计算单元密度和密度对设计变量的导数。

维护建议：

- 如果要换 TDF 形式，优先改这里。
- 改 TDF 后必须同步检查解析灵敏度。

## 7. `heaviside.py`: 平滑 Heaviside 映射

该文件将 TDF 映射为有限元密度。

主要函数：

- `heaviside(phi, eps, alpha)`
  - 将 TDF 转换成密度。
  - 输出范围为 `[alpha, 1]`。
  - `alpha` 是弱材料密度下限。

- `heaviside_derivative(phi, eps, alpha)`
  - 计算平滑 Heaviside 对 `phi` 的导数。
  - 用于解析灵敏度链式求导。

维护建议：

- `eps` 越大，边界越平滑，但结构边界更模糊。
- `alpha` 太小可能导致刚度矩阵病态。

## 8. `fem3d.py`: 三维有限元求解

该文件实现规则 Hex8 有限元分析。

主要函数：

- `generate_hex_mesh(DL, DW, DH, nelx, nely, nelz)`
  - 生成规则六面体网格。
  - 返回节点坐标和单元连接关系。

- `hex8_element_stiffness(E, nu, dx, dy, dz)`
  - 使用 2x2x2 高斯积分计算 Hex8 单元刚度矩阵。

- `assemble_global_stiffness(nodes, elements, densities, ke, E0, Emin)`
  - 根据单元密度组装全局稀疏刚度矩阵。
  - 使用 ersatz material:

```text
E_e = Emin + rho_e * (E0 - Emin)
```

- `solve_displacement(K, F, fixed_dofs)`
  - 删除固定自由度后求解线性方程。
  - 使用 SciPy 稀疏求解器 `spsolve`。

- `compute_compliance(F, U)`
  - 计算柔度 `F^T U`。

- `compute_compliance_density_gradient(U, elements, ke, E0, Emin)`
  - 计算柔度对单元密度的解析导数。

- `compute_volume_fraction(densities)`
  - 用单元平均密度作为体积分数。

维护建议：

- 如果后续加入更复杂载荷或边界，优先保证 `F` 和 `fixed_dofs` 正确。
- 如果网格变大，FEM 求解会成为主要耗时。

## 9. `loads_boundary.py`: 边界条件与载荷

该文件定义三维悬臂梁问题。

主要函数：

- `define_cantilever_boundary_conditions(nodes, DL, DW, DH, distributed=False, load_y=None, load_z=None)`

功能：

- 自动固定左端面 `x=0` 的所有节点自由度。
- 在右端面 `x=DL` 附近选择最接近 `(load_y, load_z)` 的节点施加向下力。
- 支持集中力和简单分布力。

批量数据生成中会随机化 `load_y/load_z`，让 GNN 看到不同载荷偏心情况。

## 10. `mma.py`: MMA 优化器

该文件实现当前项目使用的单体积约束 MMA。

核心类：

### `MMASolver`

主要方法：

- `_update_asymptotes(x)`
  - 更新 MMA 移动渐近线。

- `_subproblem_solution(...)`
  - 在固定对偶变量时求解可分近似子问题。

- `_constraint_approx(...)`
  - 评估 MMA 约束近似。

- `update(x, f, df, g, dg)`
  - 输入当前变量、目标值、目标梯度、约束值、约束梯度。
  - 输出下一步设计变量。

当前实现面向本项目：

- 单目标：柔度最小化。
- 单约束：体积分数不超过 `volfrac`。
- 不是通用多约束工业 MMA 包。

## 11. `optimizer.py`: 优化主控与 GNN 标签生成

这是项目的核心调度模块。

核心函数：

- `apply_gnn_step_scale(params_old, params_new, eta)`

用于 GNN 步长预测接口：

```text
params_scaled = params_old + eta * (params_new - params_old)
```

核心类：

### `MMCOptimizer3D`

主要职责：

- 评估当前 MMC 参数下的结构响应。
- 计算柔度、体积分数、密度场、位移场。
- 计算解析灵敏度。
- 调用 MMA 或 SLSQP 更新参数。
- 每次迭代保存 GNN 图数据和标签。

主要方法：

- `evaluate(params)`
  - 只计算响应，不计算梯度。
  - 用于候选 eta 响应评估。

- `evaluate_with_sensitivities(params)`
  - 计算响应和解析梯度。
  - 输出包括：
    - `compliance`
    - `volume_fraction`
    - `densities`
    - `U`
    - `compliance_grad`
    - `volume_grad`

- `_evaluate_eta_candidates(params_old, params_new)`
  - 对候选步长逐个评估：

```text
eta_candidates = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
```

  - 保存每个候选的柔度和体积分数。
  - 选择满足体积约束且柔度最低的 `eta_label`。
  - 如果全部候选违反约束：
    - `eta_label = 0.0`
    - `eta_label_index = -1`
    - `eta_failure_flag = 1`

- `callback(params)`
  - 每次迭代保存：
    - 历史柔度
    - 历史体积分数
    - 密度场
    - GNN 图数据
    - 多任务标签

- `_run_mma(x0)`
  - 使用 MMA 运行优化。
  - 默认 `stop_on_convergence=False`，会跑满 `max_iter`，便于生成固定长度轨迹。

- `_accept_mma_step(x, raw, current_data)`
  - 用真实 FEM 响应检查 MMA 候选步。
  - 防止步长过大导致体积约束失控。

- `_run_slsqp(x0)`
  - 可选 SLSQP 路径，也使用解析梯度。

## 12. `graph_export.py`: GNN 图构建与特征工程

该文件负责把 MMC 当前状态转换成 GNN 可用图。

主要函数：

- `build_node_features(...)`
  - 输出两套节点特征：
    - `node_features`: 归一化训练特征
    - `node_features_raw`: 原始物理特征

`node_features` 包含：

```text
x0/DL, y0/DW, z0/DH,
L1/max(DL,DW,DH), L2/max(DL,DW,DH), L3/max(DL,DW,DH),
sin(alpha), sin(beta), sin(gamma),
cos(alpha), cos(beta), cos(gamma),
component_volume/domain_volume,
active_flag,
compliance_grad_norm,
volume_grad_norm,
delta_params_norm,
delta_center_norm,
delta_size_norm,
delta_angle_norm
```

- `build_edge_features(comp_i, comp_j, config)`
  - 构造边特征：

```text
distance / max(DL, DW, DH),
dx/DL, dy/DW, dz/DH,
alignment_trace,
axis1_dot, axis2_dot, axis3_dot,
overlap_flag,
contact_score
```

- `build_global_features(...)`
  - 构造图级特征：

```text
iteration/max_iter,
compliance,
volume_fraction,
volfrac,
load_y/DW,
load_z/DH,
num_components/max_components_for_dataset
```

注意：

- `compliance` 当前仍是原始值，没有做 `log1p` 或标准化。
- 如果训练不稳定，可在后续版本中对 `compliance` 做压缩。

- `build_component_graph(...)`
  - 根据几何距离和重叠关系建立双向边。
  - 返回节点特征、原始节点特征、边索引、边特征、全局特征。

- `save_graph_npz(...)`
  - 将图数据和标签保存为 `.npz`。

## 13. `.npz` 图文件字段说明

每个图文件包含：

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
```

训练建议：

- 主任务：回归 `eta_label`
- 辅助分类：预测 `eta_label_index`
- 辅助回归：预测 `response_targets`
- 异常分类：预测 `eta_failure_flag`

## 14. `pyg_dataset.py`: PyTorch Geometric 数据加载

该文件将 `.npz` 图文件加载为 PyG `Data` 对象。

主要类：

### `MMCStepDataset`

功能：

- 扫描 `*_graph.npz` 文件。
- 可通过 `split_file` 按 trajectory 过滤。
- 将字段转换为 PyTorch Tensor。

关键输出：

- `data.x`: `node_features`
- `data.edge_index`: 图边索引
- `data.edge_attr`: 边特征
- `data.y`: 默认是连续 `eta_label`
- `data.global_features`: 图级特征
- `data.response_targets`: 候选 eta 的柔度/体积分数响应
- `data.eta_node_label`: 组件级 eta 标签，旧数据会由图级标签广播补齐
- `data.edge_load_prior`: 载荷距离物理先验注意力偏置，旧数据会补零
- `data.node_load_prior`: 组件到载荷点距离先验
- `data.trust_bias`: 在线微调触发用信任偏差
- `data.eta_label_index`: 离散 eta 类别
- `data.eta_feasible_mask`: 哪些候选 eta 满足体积约束
- `data.eta_failure_flag`: 是否全部候选失控

辅助函数：

- `create_dataloader(dataset_dir, batch_size=8, shuffle=True, split_file=None)`
  - 快速创建 PyG DataLoader。

维护建议：

- 后续如果加入序列模型，可在这里扩展滑动窗口读取。
- 训练/验证/测试必须按轨迹划分，不能随机按图划分。

## 15. `split_by_trajectory.py`: 按轨迹划分数据集

该文件用于防止时序数据泄漏。

主要函数：

- `read_trajectory_ids(index_path)`
  - 从 `dataset_index.csv` 读取所有 trajectory id。

- `write_ids(path, ids)`
  - 将划分后的 id 写成一行一个。

- `main()`
  - 按比例划分 train/val/test。

典型命令：

```bash
python split_by_trajectory.py --index dataset/dataset_index.csv --output-dir dataset/splits
```

输出：

```text
train_trajectories.txt
val_trajectories.txt
test_trajectories.txt
```

## 16. `visualization.py`: 可视化

该文件提供轻量级可视化工具。

主要函数：

- `plot_density_slice(...)`
  - 绘制密度切片散点图。

- `plot_3d_density_voxels(...)`
  - 绘制密度较高单元的 3D 散点图。

- `plot_components_3d(...)`
  - 绘制 MMC 构件中心和方向向量。

- `plot_history(history, save_path=None)`
  - 绘制柔度和体积分数迭代曲线。

实现细节：

- 使用 Matplotlib `Agg` 后端，适合无 GUI 环境。
- `_safe_savefig()` 会在保存图像失败时跳过，不中断主流程。

## 17. `utils.py`: 通用工具

主要函数：

- `ensure_dir(path)`
  - 创建目录。

- `save_json(path, data)`
  - 保存 JSON。

- `load_json(path)`
  - 读取 JSON。

- `normalize_features(x)`
  - 简单标准化工具，目前主图特征归一化主要在 `graph_export.py` 内完成。

- `set_random_seed(seed)`
  - 固定 Python 和 NumPy 随机种子。

- `timer(func)`
  - 简单计时装饰器。

## 18. 推荐维护顺序

如果后续继续扩展，建议按以下边界修改：

1. 改几何描述：优先改 `mmc3d_components.py` 和 `tdf.py`
2. 改材料/有限元：优先改 `fem3d.py`
3. 改边界和载荷：优先改 `loads_boundary.py`
4. 改优化器：优先改 `optimizer.py` 和 `mma.py`
5. 改 GNN 特征：优先改 `graph_export.py`
6. 改 GNN 数据加载：优先改 `pyg_dataset.py`
7. 改批量数据生成：优先改 `generate_dataset.py`
8. 改 GAT/元学习：优先改 `models.py`、`train_gnn_step.py`、`meta_learning.py`、`online_meta_controller.py`

## 19. 当前已知限制

- MMA 是单体积约束版本，不是通用多约束 MMA。
- `component_shape="box"` 是高阶 p-norm 平滑近似，不是精确不可导长方体边界。
- `global_features` 中的 `compliance` 未做归一化，训练时可考虑 `log1p` 或标准化。
- 边构建主要基于几何距离和重叠，暂未加入应变能传力路径边。
- 当前图样本是单步图，尚未实现连续多步序列样本。
- 第一版组件级标签由图级 `eta_label` 广播，在线微调阶段才使用稀疏 FEM 真标签产生差异化监督。

## 20. 快速命令参考

安装依赖：

```bash
python -m pip install -r requirements.txt
```

单次运行：

```bash
python main.py
```

批量生成数据：

```bash
python generate_dataset.py --num-trajectories 200 --max-iter 20 --min-components 6 --max-components 12
```

按轨迹划分：

```bash
python split_by_trajectory.py --index dataset/dataset_index.csv --output-dir dataset/splits
```

加载 PyG 数据：

```python
from pyg_dataset import MMCStepDataset

train_set = MMCStepDataset("dataset", split_file="dataset/splits/train_trajectories.txt")
sample = train_set[0]
print(sample.x.shape, sample.edge_index.shape, sample.y)
```

训练物理偏置 GAT：

```bash
python train_gnn_step.py --dataset-dir dataset --epochs 50 --batch-size 8
```

注意力物理先验缩放使用非负约束：实际缩放为 `actual_alpha = 0.1 + 0.9 * sigmoid(raw_alpha)`，始终位于 `[0.1, 1.0]`。`raw_alpha` 默认前 20 个 epoch 冻结，使 `actual_alpha` 贴近 `0.1`；之后释放训练，并通过 `alpha_weight_decay = 1e-4` 单独正则。MAML support 内循环使用 SGD 且不对 `raw_alpha` 施加 weight decay。

MAML 风格元学习：

```bash
python meta_learning.py --dataset-dir dataset --epochs 20
```
