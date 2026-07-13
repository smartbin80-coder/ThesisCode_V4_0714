# Project Handoff: 3D MMC Topology Optimization + GNN Step Dataset

本文档用于把当前项目状态交接给下一个 AI 或开发者。项目目标是：基于 3D Moving Morphable Component (MMC) 拓扑优化过程，生成图神经网络数据集，用 GNN 预测优化迭代中的步长 `eta`，并为后续在线元学习控制收敛可靠性预留数据基础。

## 1. 当前项目位置

主项目目录：

```text
D:\codex\workspace_code_v2\project_3d_mmc
```

已上传过 GitHub 的仓库：

```text
https://github.com/smartbin80-coder/ThesisCode_V3_0709
```

注意：GitHub 上的版本不一定包含本地最后几次改动。若需要最新代码，应以本地 `project_3d_mmc` 为准。

## 2. 当前核心目标

代码当前实现的是：

1. 3D MMC 构件参数化。
2. 基于有限元的柔度与体积分数评估。
3. 基于 SLSQP/MMA 风格优化器更新 MMC 参数。
4. 在每次优化迭代中，把当前 MMC 构件状态图化为 GNN 样本。
5. 为候选步长 `eta_candidates = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]` 计算柔度和体积分数响应。
6. 选择满足体积约束且柔度最低的候选步长作为 `eta_label`。
7. 支持批量生成多条随机轨迹，并按轨迹划分训练集、验证集、测试集，避免时序数据泄漏。

## 3. 主要文件职责

### `config.py`

全局配置文件，包含：

- 设计域尺寸：`DL, DW, DH`
- 网格数量：`nelx, nely, nelz`
- 构件数量：`num_components`
- 体积分数约束：`volfrac`
- 优化器类型：`optimizer_type`
- 最大迭代步数：`max_iter`
- GNN 步长候选：`eta_candidates`
- 数据集随机生成范围：`min_components_for_dataset`, `max_components_for_dataset`
- 载荷位置：`load_y`, `load_z`

### `mmc3d_components.py`

定义 3D MMC 构件及参数转换：

- MMC 构件中心：`x, y, z`
- 尺寸参数：`L1, L2, L3`
- 姿态角：`theta_x, theta_y, theta_z`
- 参数向量与组件对象互转
- 组件上下界定义

### `tdf.py`

负责计算 MMC 的拓扑描述函数 TDF 以及对参数的灵敏度。

### `fem3d.py`

3D 有限元分析模块，负责：

- 组装刚度矩阵
- 求解位移
- 计算柔度
- 计算单元应变能相关量

### `loads_boundary.py`

定义边界条件和载荷位置。当前用于梁结构测试，左端固定，右端施加载荷。

### `mma.py`

MMA 优化器实现。当前不是完全通用多约束 MMA，但已作为当前项目优化流程的一部分使用。

### `optimizer.py`

优化主流程。最关键逻辑包括：

- `apply_gnn_step_scale(params_old, params_new, eta)`

  预留给 GNN 预测步长的接口：

  ```python
  params_scaled = params_old + eta * (params_new - params_old)
  ```

- `_evaluate_eta_candidates`

  对每个候选 `eta` 计算：

  ```text
  compliance_eta
  volume_fraction_eta
  ```

  然后选择满足体积约束且柔度最低的 `eta` 作为标签。

- `callback`

  每次迭代保存：

  - 当前密度场
  - 当前图样本 `.npz`
  - 当前参数
  - 当前候选步长响应
  - 当前标签

### `graph_export.py`

把 MMC 构件状态导出为 GNN 图数据，是 GNN 数据质量的核心文件。

当前图化方式：

- 一个 MMC 构件对应一个图节点。
- 两个构件足够接近或包围球近似重叠时建立双向边。
- 节点特征保存归一化几何、角度、灵敏度、上一迭代参数变化。
- 边特征保存距离、方向、姿态对齐、重叠标记和重叠评分。
- 图级特征保存迭代步、柔度、体积分数、目标体积分数、载荷位置、构件数量。

重要：尺寸特征归一化 bug 已修复。当前尺寸特征使用：

```python
max_dim = max(config.DL, config.DW, config.DH)
sizes = params[:, 3:6] / max_dim
```

而不是分别除以 `DL, DW, DH`。

### `generate_dataset.py`

批量生成 GNN 数据集的入口脚本。支持：

- 多轨迹生成
- 不同随机种子
- 随机构件数量
- 载荷位置扰动
- 输出 `dataset_index.csv`
- 输出 `graph_index.csv`

其中 `graph_index.csv` 是 Excel 友好的逐图快照索引，每一行对应一个 `.npz` 图样本。

### `split_by_trajectory.py`

按轨迹划分训练集、验证集、测试集，避免把同一条优化轨迹的不同时刻同时放进训练和测试。

### `pyg_dataset.py`

PyTorch Geometric 数据加载器，把 `.npz` 图文件加载为：

```python
torch_geometric.data.Data
```

主要字段：

- `x`: 节点特征
- `edge_index`: 边连接
- `edge_attr`: 边特征
- `y`: 默认是 `eta_label`
- `global_features`: 图级特征
- `response_targets`: 多任务标签，包含候选步长对应的柔度和体积分数

### `README.md`

用户使用说明，包括如何生成 GNN 步长预测数据集。

### `CODE_GUIDE.md`

整体代码说明文档，更偏代码维护与模块解释。

## 4. 当前数据目录说明

### `dataset_quality_5_fixed`

当前保留的主要测试数据集目录：

```text
D:\codex\workspace_code_v2\project_3d_mmc\dataset_quality_5_fixed
```

里面是 5 条轨迹，每条轨迹 21 个图样本：

```text
traj_0000_iter_0000_graph.npz
traj_0000_iter_0001_graph.npz
...
traj_0004_iter_0020_graph.npz
```

用途：GNN 训练或数据质量检查。

每个 `.npz` 包含：

```text
node_features
node_features_raw
edge_index
edge_attr
global_features
compliance
volume_fraction
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

### `results_quality_5_fixed`

对应的优化结果摘要目录：

```text
D:\codex\workspace_code_v2\project_3d_mmc\results_quality_5_fixed
```

每条轨迹一个子目录：

```text
traj_0000/
  final_params.npy
  summary.json
```

用途不是直接训练 GNN，而是：

- 轨迹级审计
- 判断最终是否收敛
- 检查最终柔度和体积分数
- 后续在线元学习可靠性分析

## 5. GNN 图化细节

### 节点

一个 MMC 组件是一个节点。

`node_features_raw` 原始特征约为：

```text
x, y, z,
L1, L2, L3,
theta_x, theta_y, theta_z,
component_volume,
active
```

`node_features` 当前为 20 维：

```text
center_x_norm, center_y_norm, center_z_norm,
L1_norm, L2_norm, L3_norm,
sin(theta_x), sin(theta_y), sin(theta_z),
cos(theta_x), cos(theta_y), cos(theta_z),
component_volume_norm,
active,
compliance_grad_norm_scaled,
volume_grad_norm_scaled,
delta_params_norm_scaled,
delta_center_norm,
delta_size_norm,
delta_angle_norm
```

归一化方式：

- 坐标：`x/DL, y/DW, z/DH`
- 尺寸：`L1/max_dim, L2/max_dim, L3/max_dim`
- 角度：`sin(theta), cos(theta)`
- 体积：`component_volume / (DL * DW * DH)`
- 灵敏度范数：`log1p(abs(norm))` 后按当前图最大值缩放

### 边

若两个构件满足任一条件，则连边：

```text
中心距离 < 0.45 * max(DL, DW, DH)
```

或：

```text
构件包围球近似重叠
```

边是双向保存的。

`edge_attr` 当前为 10 维：

```text
normalized_distance,
dx_norm, dy_norm, dz_norm,
alignment_trace,
axis_dot_1, axis_dot_2, axis_dot_3,
overlap_flag,
overlap_score
```

### 图级特征

`global_features` 包含：

```text
iteration_norm,
compliance,
volume_fraction,
target_volfrac,
load_y_norm,
load_z_norm,
num_components_norm
```

## 6. 标签定义

当前候选步长：

```python
eta_candidates = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
```

对每个候选步长：

```python
params_eta = params_old + eta * (params_new - params_old)
```

计算：

```text
compliance_eta
volume_fraction_eta
```

标签选择规则：

1. 找出所有满足体积约束的候选步长。
2. 在可行候选中选择柔度最低的 `eta`。
3. 如果全部候选违反体积约束，则 `eta_label = 0.0`，并设置 `eta_failure_flag = 1`。

多任务标签：

```text
response_targets[:, 0] = compliance_candidates
response_targets[:, 1] = volume_candidates
```

这支持后续训练模型同时预测最优步长和候选步长响应曲线。

## 7. 已完成的重要修改

1. 增加 GNN 步长预测接口：

   ```python
   apply_gnn_step_scale(params_old, params_new, eta)
   ```

2. 每次迭代保存候选步长响应。

3. 每个图 `.npz` 保存：

   ```text
   eta_candidates
   eta_label
   compliance_candidates
   volume_candidates
   response_targets
   ```

4. 增加批量轨迹生成脚本 `generate_dataset.py`。

5. 增加按轨迹划分脚本 `split_by_trajectory.py`。

6. 增加 PyG 数据加载器 `pyg_dataset.py`。

7. 修复尺寸特征归一化不一致 bug。

8. 将 MMA 保守接受步长与 `config.eta_candidates` 对齐，并额外加入 `0.0` 作为保底。

9. 增加 `graph_index.csv`，方便在 Excel 中检查每次迭代快照。

10. 增加 `CODE_GUIDE.md`。

## 8. 当前已知限制

1. 当前 MMA 不是完整通用多约束工业级 MMA。用户明确说过：“通用多约束 MMA 可以不改”。

2. 当前 GNN 训练代码还没有完整实现，项目目前主要完成了数据生成和 PyG 加载接口。

3. 当前在线元学习控制收敛可靠性还没有真正实现，只是通过以下数据为其打基础：

   - 每步图状态
   - 候选步长响应
   - 失败标记 `eta_failure_flag`
   - 轨迹级 `summary.json`

4. 当前边构造主要基于几何距离和包围球重叠，还没有加入应变能传力路径边。

5. 当前数据规模只有一个小测试集 `dataset_quality_5_fixed`，正式训练需要更多轨迹，建议至少数千图样本。

## 9. 推荐下一步

### 第一步：生成更大数据集

建议从 100 到 500 条轨迹开始，而不是直接上千条。

示例：

```powershell
cd D:\codex\workspace_code_v2\project_3d_mmc
python generate_dataset.py --num-trajectories 100 --max-iter 20 --min-components 6 --max-components 12 --output-dir dataset_train_100 --results-dir results_train_100
```

生成后检查：

```text
dataset_train_100\graph_index.csv
results_train_100\traj_xxxx\summary.json
```

### 第二步：按轨迹划分

```powershell
python split_by_trajectory.py --dataset-dir dataset_train_100 --output-dir splits_train_100
```

训练、验证、测试必须按轨迹划分，不要随机打散所有 `.npz` 图。

### 第三步：实现 GNN 训练脚本

建议新增：

```text
train_gnn_step.py
models.py
```

模型输入：

- `x`
- `edge_index`
- `edge_attr`
- `global_features`

模型输出：

- 主任务：`eta_label`
- 辅助任务：`response_targets`
- 可选分类任务：`eta_failure_flag`

### 第四步：在线元学习接口

建议新增：

```text
online_meta_controller.py
```

初始策略：

1. GNN 预测 `eta_pred`。
2. 用安全投影限制范围 `[0.25, 1.5]`，模型输出公式为 `0.25 + 1.25 * sigmoid(raw_eta)`。
3. 在线检查预测步长对应的体积约束和柔度变化。
4. 若预测不可靠，回退到传统 `eta_candidates` 搜索或 MMA 保守接受策略。
5. 将失败样本加入在线缓冲区，用于元学习微调。

## 10. 给下一个 AI 的注意事项

1. 不要删除 `dataset_quality_5_fixed`，这是当前修复归一化后的有效测试数据。

2. 不要把 `results_quality_5_fixed` 当作训练输入，它是轨迹审计和可靠性分析辅助目录。

3. 如果要上传 GitHub，先确认本地代码是否已经同步到上传目录：

   ```text
   D:\codex\workspace_code_v2\upload_ThesisCode_V3_0709
   ```

4. 如果要重新导出桌面 TXT，需要写到：

   ```text
   C:\Users\lenovo\Desktop\project_3d_mmc_full_code_latest.txt
   ```

   这通常需要额外文件系统权限。

5. 修改图特征时，务必同步更新：

   - `graph_export.py`
   - `pyg_dataset.py`
   - `README.md`
   - `CODE_GUIDE.md`
   - 本文档

6. 如果未来加入应变能边或时序窗口样本，要注意 `.npz` 格式兼容性，不要破坏已有字段读取。

## 11. 快速判断当前代码是否满足用户目标

当前已经满足：

- 批量轨迹数据生成
- 归一化节点特征
- 候选步长多任务标签
- 按轨迹划分
- PyG 数据加载
- 为 GNN 步长预测预留接口

当前尚未完成：

- 大规模正式 GAT 训练与调参
- GAT 嵌入优化器的工程级闭环验证
- 在线元学习可靠性控制的长期轨迹验证
- 基于应变能或传力路径增强图边
- 大规模正式训练集

