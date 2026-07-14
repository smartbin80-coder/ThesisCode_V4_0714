# Project Handoff: 3D MMC + Physics-Biased GAT + Meta-Learning

本文档交接当前 `project_3d_mmc` 项目状态给下一个 AI 或开发者。请以下文为准；旧交接信息中关于“GNN 训练尚未实现”的表述已经过期。

## 1. 项目目标与核心思路

课题方向：基于图注意力网络的 MMC 拓扑优化组件级自适应步长预测方法研究。

核心思路：不替代 FEM 和灵敏度求解，保留物理导数方向，只让 GAT 预测每个 MMC 组件的步长缩放系数。这样 AI 只控制“走多远”，不控制“往哪走”。若预测不可靠，通过 FEM 验证、安全回退和阻尼机制保证收敛可靠性。

当前实现重点：

1. MMC 组件图化与 GNN 数据导出。
2. 候选步长响应标签生成。
3. 物理偏置 GAT 模型。
4. 离线监督训练入口。
5. MAML 风格元学习入口。
6. 在线 trust-bias 触发、稀疏真标签和安全回退控制接口。

## 2. 当前仓库状态

主项目目录：

```text
D:\codex\workspace_code_v2\project_3d_mmc
```

Git 仓库根目录：

```text
D:\codex\workspace_code_v2
```

已有初始提交：

```text
ea307c8 feat: 初始化项目，完成 GAT 物理偏置、MAML 及在线微调框架
```

本交接版本包含初始提交之后的 alpha 非负约束修复、训练策略修正和文档同步。接手时请先运行：

```powershell
git status --short
git log --oneline -3
```

若工作区干净，则最新提交已经包含本交接文档描述的 alpha 修复；若仍有未提交改动，应先审查并提交。当前仓库未配置远程 `origin`，如需上传到 GitHub，必须先添加 remote。

## 3. 数据与结果目录

不要把数据和结果目录纳入 Git。根目录 `.gitignore` 已忽略数据、结果、备份和二进制数组文件。

重要目录：

```text
project_3d_mmc/dataset_quality_5_fixed
project_3d_mmc/results_quality_5_fixed
D:\codex\workspace_code_v2\dataset_schema_v2_smoke
D:\codex\workspace_code_v2\results_schema_v2_smoke
```

说明：

- `dataset_quality_5_fixed` 是旧 schema 测试数据集，不能删除。它用于验证 loader 对旧 `.npz` 的兼容性。
- `results_quality_5_fixed` 是轨迹审计和可靠性分析辅助目录，不要当作 GNN 训练输入。
- `dataset_schema_v2_smoke` 是最近生成的 schema v2 smoke 数据，用于验证新增字段形状。
- 正式训练建议重新生成 100-500 条轨迹起步。

## 4. 当前核心配置

`config.py` 中关键配置：

```python
eta_candidates = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
feature_schema_version = 2
load_prior_lambda = 2.0
alpha_weight_decay = 1e-4
alpha_freeze_epochs = 20
trust_bias_threshold = 0.10
node_step_label_mode = "global_broadcast"
```

2026-07-14 默认配置已经切换到 60×4×20 设计域和固定 24 组件：

```python
DL = 60.0
DW = 4.0
DH = 20.0
nelx = 60
nely = 4
nelz = 20
max_iter = 20
num_components = 24
min_components_for_dataset = 24
max_components_for_dataset = 24
save_density = False
save_process_plots = True
eta_candidates = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
```

候选步长标签规则：

1. 对每个候选 `eta` 计算真实 FEM 响应 `compliance_eta` 和 `volume_fraction_eta`。
2. 在满足体积约束的候选中选柔度最低者作为 `eta_label`。
3. 若全部候选违反体积约束，则 `eta_label = 0.0`，`eta_failure_flag = 1`。

## 5. 已实现模块

### 数据生成与图导出

- `generate_dataset.py`：批量生成随机轨迹，写出 `dataset_index.csv` 和 `graph_index.csv`。
- `graph_export.py`：构建 MMC 组件图、节点特征、边特征、载荷距离先验和 schema v2 辅助字段。
- `optimizer.py`：在每步优化回调中保存图样本、候选步长响应、组件级标签和 trust-bias 字段。

### PyG 数据加载

- `pyg_dataset.py`：加载 `.npz` 为 `torch_geometric.data.Data`。
- 对旧 schema 数据兼容：缺失字段用 `torch.zeros(...)` 或合理默认张量补齐，绝不写 `None`，避免 PyG batch 失败。

### GAT 模型

- `models.py`：新增 `PhysicsBiasedGATLayer` 和 `MMCStepGAT`。
- `MMCStepGAT` 输出：
  - `node_eta_pred`
  - `graph_eta_pred`
  - `node_eta_logits`
  - `graph_eta_logits`
  - `response_pred`

### 训练与元学习

- `train_gnn_step.py`：离线监督训练入口。
- `meta_learning.py`：first-order MAML 风格元学习入口，按 trajectory 构建 task，support/query 在同一轨迹内随机采样。
- `online_meta_controller.py`：在线推理、trust-bias 检查、稀疏真标签、加权微调和安全回退阻尼接口。

## 6. GNN/GAT 数据 schema

旧基础字段仍保留：

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
DL
DW
DH
nelx
nely
nelz
E0
Emin
nu
volfrac
max_iter
```

schema v2 新增字段：

```text
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

字段含义：

- `load_point = [DL, load_y, load_z]`。
- `node_load_distance_norm` 是组件中心到载荷点的归一化距离。
- `node_load_prior = exp(-load_prior_lambda * node_load_distance_norm)`。
- `edge_load_prior` 是每条有向边的物理注意力偏置，shape 为 `(num_edges, 1)`。
- `eta_node_label` 当前第一版由图级 `eta_label` 广播得到。
- `component_strain_energy_norm` 用单元应变能和密度灵敏度软聚合到组件级。
- `trust_bias` 用于在线微调触发。
- `DL/DW/DH`, `nelx/nely/nelz`, `E0/Emin/nu`, `volfrac`, `max_iter` 是在线单步 FEM evaluator 重建元数据；旧数据缺失时 `test_online_step.py` 回退到当前 `config.py`。

重要：旧数据没有这些字段，但 loader 会补齐默认张量。

## 7. 物理偏置 GAT 的关键安全约束

### eta 输出范围

节点级步长输出被固定映射到 `[0.25, 1.5]`：

```python
node_eta_pred = 0.25 + 1.25 * torch.sigmoid(raw_eta)
```

这避免 eta 小于 0.25 导致组件近似卡死，也避免过大步长破坏稳定性。

### 物理先验 attention bias

当前注意力逻辑：

```python
attention_logit = learned_logit + actual_alpha * edge_load_prior_bias
```

关键修复：`actual_alpha` 不是裸 `nn.Parameter`，而是非负约束值：

```python
actual_alpha = 0.1 + 0.9 * torch.sigmoid(raw_alpha)
```

因此即使 `raw_alpha` 被优化到负数，实际物理先验权重仍在 `[0.1, 1.0]`，不会变成负数，也不会把靠近载荷的组件反向扣分。

训练策略：

- `raw_alpha` 默认前 20 个 epoch 冻结，使 `actual_alpha` 贴近 0.1。
- 20 epoch 后释放训练。
- 外层 optimizer 对 alpha 参数组使用 `alpha_weight_decay = 1e-4`。
- MAML support 内循环使用 SGD，且不对 `raw_alpha` 施加 weight decay。

## 8. 训练接口

离线监督训练：

```powershell
cd D:\codex\workspace_code_v2\project_3d_mmc
python train_gnn_step.py --dataset-dir dataset_train_100 --epochs 50 --batch-size 8
```

可选参数：

```text
--alpha-freeze-epochs
--alpha-weight-decay
--hidden-channels
--heads
--num-layers
--response-candidates
--checkpoint
```

训练损失包含：

- 节点 eta 回归：`node_eta_pred` 对 `eta_node_label`。
- 图级 eta 辅助回归：`graph_eta_pred` 对 `eta_label`。
- 节点/图级候选步长分类。
- 候选响应曲线预测：`response_pred` 对 `response_targets`。

注意：响应目标中 compliance 已在训练损失内用 `log1p` 缩放，避免数值爆炸。

重要：`eta_label_index = -1` 表示全部候选步长失败，不能被 `clamp(min=0)` 当成 0 类训练。当前 `compute_step_losses` 会用 `eta_failure_flag` 屏蔽失败样本的节点/图级分类损失，只保留回归和响应辅助损失。

## 9. MAML 元学习接口

命令示例：

```powershell
python meta_learning.py --dataset-dir dataset_train_100 --epochs 20 --tasks-per-epoch 8
```

实现特点：

- 一个 trajectory 作为一个 task。
- support/query 在同一 trajectory 内随机采 step，不按时间前后硬切。
- support 内循环用 SGD。
- 外循环 loss 显式包含 support loss、query loss 和 L2 regularization。
- support 内循环不对 `raw_alpha` 使用 weight decay。

当前是 first-order MAML 风格实现，主要用于打通元学习训练接口，后续可进一步改成严格二阶 MAML 或 Reptile 风格。

## 10. 在线微调与安全回退接口

`online_meta_controller.py` 提供：

```python
predict_component_eta(...)
apply_component_step_scale(...)
check_trust_bias(...)
fallback_to_safe_eta_search(...)
select_key_components(...)
build_sparse_true_component_labels(...)
weighted_online_mse(...)
adapt_online_on_sparse_labels(...)
SafetyDampingController
```

在线策略：

1. GAT 预测组件级 eta。
2. eta 通过 `[0.25, 1.5]` 投影。
3. FEM 验证预测步长后的柔度和体积约束。
4. 若 `trust_bias > 0.10`，触发在线微调。
5. 微调时选 3-5 个关键组件：优先 `component_strain_energy_norm`，其次 `node_load_prior`，最后节点度。
6. 对关键组件尝试局部候选 `(0.5, 1.0, 1.5)` 并 FEM 验证，生成稀疏真标签。
7. 未验证组件用当前模型预测作为软伪标签。
8. 微调损失中真实节点权重 5.0，伪标签节点权重 0.1。
9. 若预测导致柔度上升或体积约束恶化，回退到 `eta_candidates + 0.0` 的保守搜索。
10. 触发回退后，`SafetyDampingController` 后续 3 步使用 0.8 阻尼，防止反复震荡。

## 11. 已验证内容

已经完成的 smoke/接口验证：

1. `py_compile` 通过关键模块。
2. 旧 `dataset_quality_5_fixed` 可用 PyG DataLoader batch。
3. 新 schema smoke 数据字段正确：
   - `eta_candidates.shape == (6,)`
   - `response_targets.shape == (6, 2)`
   - `eta_node_label.shape == (num_components,)`
   - `edge_load_prior.shape == (num_edges, 1)`
4. GAT forward 可处理 PyG batch。
5. `node_eta_pred` 输出满足 `[0.25, 1.5]`。
6. `actual_alpha` 定向验证：
   - raw alpha = -100 时，`actual_alpha` 仍约为 0.1。
   - raw alpha = 100 时，`actual_alpha` 约为 1.0。
7. MAML support 内循环 SGD 参数组 weight decay 为 `[0, 0.0]`。
8. 失败样本分类掩码定向验证通过：当 `eta_failure_flag = 1` 且 label index 为 `-1` 时，`node_class = 0`、`graph_class = 0`。
9. 普通训练 smoke 1 epoch 跑通。
10. MAML smoke 1 epoch 跑通。
11. 在线控制器 smoke test 跑通，安全回退后阻尼序列为 `0.8, 0.8, 0.8, 1.0`。

### 2026-07-14 笔记本 5 轨迹全链路验证（6 候选）

按桌面作业指导书完成了轻量链路调试。由于当前 Python 进程在 `project_3d_mmc` 子目录下创建新目录时会遇到权限限制，`generate_dataset.py` 的 `writable_dir` 自动把调试数据和结果落到仓库根目录：

```text
D:\codex\workspace_code_v2\dataset_debug
D:\codex\workspace_code_v2\results_debug
```

已执行并通过：

```powershell
python -B generate_dataset.py --num-trajectories 5 --max-iter 10 --min-components 4 --max-components 8 --output-dir dataset_debug --results-dir results_debug/batch
python -B split_by_trajectory.py --index D:\codex\workspace_code_v2\dataset_debug\dataset_index.csv --output-dir D:\codex\workspace_code_v2\dataset_debug\splits --train-ratio 0.6 --val-ratio 0.2
python -B train_gnn_step.py --dataset-dir D:\codex\workspace_code_v2\dataset_debug --split-file D:\codex\workspace_code_v2\dataset_debug\splits\train_trajectories.txt --val-split-file D:\codex\workspace_code_v2\dataset_debug\splits\val_trajectories.txt --epochs 5 --batch-size 4 --hidden-channels 16 --heads 2
python -B meta_learning.py --dataset-dir D:\codex\workspace_code_v2\dataset_debug --epochs 3 --tasks-per-epoch 2 --support-size 2 --query-size 2 --inner-steps 1 --meta-lr 1e-4 --hidden-channels 16 --heads 2
python -B test_online_step.py
python -B project_3d_mmc\test_online_step.py
```

验收结果：

- 数据生成成功：5 条轨迹，每条包含 `iter_0000` 到 `iter_0010`，共 55 个 `*_graph.npz`。
- 首个样本包含 `edge_load_prior`，shape 为 `(20, 1)`；`eta_candidates.shape == (6,)`，`response_targets.shape == (6, 2)`。
- 轨迹切分成功：train=3，val=1，test=1。
- GAT 训练 5 epoch 跑通，无 NaN；最后一轮 `train_loss=1.986164e+00`，`val_loss=2.829171e+00`，并保存 `D:\codex\workspace_code_v2\results_debug\gat_step_model.pt`。
- MAML 训练 3 epoch 跑通并保存 `D:\codex\workspace_code_v2\results_debug\gat_maml_model.pt`；最后一轮 `meta_loss=1.074905e+01`。
- 新增 `test_online_step.py` 单步在线模拟脚本，已输出组件级 eta 和 FEM 试探结果：
  - 从 `project_3d_mmc` 子目录和仓库根目录运行均通过。
  - 默认加载 MAML checkpoint，预测的组件步长：`[0.8238, 0.7945, 0.8417, 0.8389, 0.8173]`
  - 显式加载 GAT checkpoint 通过，预测的组件步长：`[0.6482, 0.6391, 0.6226, 0.6727, 0.6461]`
  - 试探后的柔度值：`7.942713e+04`
  - 试探后的体积分数：`1.300854e-01`
  - checkpoint 回退逻辑已定向验证：当 MAML checkpoint 缺失时会回退到 `gat_step_model.pt`。

结论：全链路（数据生成 -> 加载 -> 训练 -> MAML -> 在线单步模拟）已打通，可进行 5 轨迹验证级别的后续实验。

### 2026-07-14 代码审阅后修复

本轮自审后修复了 3 个问题：

- `_accept_mma_step` 的兜底逻辑不再被 `eta=0.0` 吞掉；约束违反未改善时，会在正候选步长中选择 `(violation, compliance)` 最小者，避免继续原地停滞。
- `generate_dataset.writable_dir()` 的回退路径保留相对层级，例如 `results_debug/batch` 不再退化成仓库根目录 `batch`。
- 图样本新增 FEM evaluator 重建元数据，`test_online_step.py` 优先读取样本元数据，并支持 `--nelx --nely --nelz --Emin` 显式覆盖；旧数据缺字段时回退到当前 `config.py`。

已验证：

```powershell
python -B -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text(encoding='utf-8')) for p in ['generate_dataset.py','optimizer.py','pyg_dataset.py','test_online_step.py']]; print('syntax ok')"
python -B project_3d_mmc\test_online_step.py
python -B generate_dataset.py --num-trajectories 1 --max-iter 1 --min-components 4 --max-components 4 --output-dir D:\codex\workspace_code_v2\dataset_metadata_smoke --results-dir D:\codex\workspace_code_v2\results_metadata_smoke
python -B test_online_step.py --dataset-dir D:\codex\workspace_code_v2\dataset_metadata_smoke --results-dir D:\codex\workspace_code_v2\results_metadata_smoke\online_step
```

metadata smoke 首个样本已确认包含 `DL/DW/DH`, `nelx/nely/nelz`, `E0/Emin/nu`, `volfrac`, `max_iter`。

### 2026-07-14 MMC 过程拓扑图功能

新增每条轨迹自动输出三张红色 3D 拓扑过程图：

```text
results_dir/process_plots/initial_iter_0000.png
results_dir/process_plots/middle_iter_XXXX.png
results_dir/process_plots/final_iter_XXXX.png
```

实现位置：

- `visualization.py`: `plot_topology_process(...)`，红色体素实体、白底、固定视角。
- `optimizer.py`: `callback()` 后调用 `_save_process_plot_if_needed(...)`，在 `iteration == 0`, `max(1, max_iter // 2)`, `max_iter` 保存三张图。
- `generate_dataset.py`: 新增 `--no-process-plots` 和 `--process-plot-threshold`。

已验证：

```powershell
python -B generate_dataset.py --num-trajectories 1 --max-iter 2 --min-components 4 --max-components 4 --output-dir D:\codex\workspace_code_v2\dataset_plot_smoke --results-dir D:\codex\workspace_code_v2\results_plot_smoke
```

输出确认：

```text
D:\codex\workspace_code_v2\results_plot_smoke\traj_0000\process_plots\initial_iter_0000.png
D:\codex\workspace_code_v2\results_plot_smoke\traj_0000\process_plots\middle_iter_0001.png
D:\codex\workspace_code_v2\results_plot_smoke\traj_0000\process_plots\final_iter_0002.png
```

当前低网格 smoke 图会偏块状；扩大到 `20x8x8` 或更高网格后会更接近参考图的连续结构外观。

### 2026-07-14 默认设计域扩大到 60×4×20

默认设计域和组件数已从小型调试算例改为：

```python
DL = 60.0
DW = 4.0
DH = 20.0
nelx = 60
nely = 4
nelz = 20
num_components = 24
min_components_for_dataset = 24
max_components_for_dataset = 24
max_iter = 20
```

`create_initial_components(config)` 的单次默认初始构件改为 `6 × 2 × 2 = 24` 规则阵列：

- x 从 `0.15*DL` 到 `0.90*DL` 共 6 组。
- y 为 `0.30*DW` 和 `0.70*DW`。
- z 为 `0.30*DH` 和 `0.70*DH`。
- 初始半轴为 `L1=DL/14`, `L2=DW/5`, `L3=DH/8`。

批量数据生成默认固定 24 组件：`min_components_for_dataset=max_components_for_dataset=24`。若显式传入不同 `--min-components/--max-components`，仍可恢复随机组件数量。

已执行最小 smoke：

```powershell
python -B generate_dataset.py --num-trajectories 1 --max-iter 2 --output-dir D:\codex\workspace_code_v2\dataset_domain60_smoke --results-dir D:\codex\workspace_code_v2\results_domain60_smoke
```

验收结果：

- `num_components=24`。
- `DL=60.0`, `DW=4.0`, `DH=20.0`。
- `nelx=60`, `nely=4`, `nelz=20`，单元数 `4800`。
- 首个样本 `load_point=[60.0, 2.0427714759501256, 11.038418470063295]`。
- `eta_candidates.shape == (6,)`，`response_targets.shape == (6, 2)`。
- 三张过程图生成成功，尺寸均为 `1980×1144`：
  - `initial_iter_0000.png`
  - `middle_iter_0001.png`
  - `final_iter_0002.png`

注意：该 smoke 约耗时 60-90 秒。后续扩大到 `5` 条轨迹、`10` 步前，建议先确认机器资源可接受；`20` 条轨迹、`20` 步会显著更慢。

## 12. 当前未完成和风险

当前尚未完成：

- 大规模正式训练集。
- 真实离线 GAT 训练收敛性评估。
- MAML 在多载荷工况上的泛化实验。
- GAT 嵌入 MMC 优化主循环的完整在线闭环。
- 在线微调长期稳定性实验。
- 真正组件级标签的离线生成，目前离线阶段仍是图级 `eta_label` 广播。
- 基于应变能或传力路径的边增强，目前仅有组件级应变能特征和载荷距离先验。

主要风险：

- 当前 `component_strain_energy_norm` 是基于灵敏度软权重的近似聚合，不是严格物理能量分解。
- MAML 当前是 first-order 风格，不是完整二阶 MAML。
- 在线稀疏真标签接口已实现，但尚未在真实优化长轨迹中闭环验证。
- `global_features` 中原始 compliance 很大，模型内部已对图级 compliance 做 `log1p`，后续若新增模型需注意同样缩放。

## 13. 推荐下一步

建议按以下顺序继续：

1. 确认 alpha 非负约束和交接文档已经提交：

   ```powershell
   git status --short
   git log --oneline -3
   ```

2. 生成 100 条轨迹起步的数据集：

   ```powershell
   cd D:\codex\workspace_code_v2\project_3d_mmc
   python generate_dataset.py --num-trajectories 100 --max-iter 20 --min-components 6 --max-components 12 --output-dir dataset_train_100 --results-dir results_train_100
   ```

3. 按轨迹划分训练、验证、测试集：

   ```powershell
   python split_by_trajectory.py --dataset-dir dataset_train_100 --output-dir splits_train_100
   ```

4. 跑离线 GAT 训练，检查 validation loss、eta 误差、候选分类准确率和响应预测误差。

5. 跑 MAML 训练，对比普通监督预训练在新载荷工况下的少步适应能力。

6. 将 `online_meta_controller.py` 接入 MMC 主循环，做短轨迹闭环验证，再逐步扩大轨迹规模。

## 14. 给下一个 AI 的硬性注意事项

1. 不要删除 `dataset_quality_5_fixed`。
2. 不要把 `results_quality_5_fixed` 当训练输入。
3. 不要用 `git add .`，只添加代码和文档；数据、结果、`.npz`、`.npy` 都应保持忽略。
4. 修改图字段时必须同步更新：
   - `graph_export.py`
   - `pyg_dataset.py`
   - `README.md`
   - `CODE_GUIDE.md`
   - `HANDOFF.md`
5. 物理先验注意力必须保持正向，不要再改回裸 `alpha * edge_load_prior_bias`。
6. MAML support 内循环不要对 `raw_alpha` 加 weight decay。
7. `eta_label_index = -1` 是失败标记，不要在分类损失里强行 clamp 成 0 类。
8. GAT 只允许预测步长缩放，不允许替代 FEM 或灵敏度方向。
9. 所有在线预测步长必须经 FEM 验证，不可靠时必须回退到保守候选搜索。

## 2026-07-14 Strict MMC Domain Feasibility Fix

Issue confirmed from `dataset_domain60_5x50`: component centers stayed inside `60 x 4 x 20`, but rotated component AABBs exceeded the design domain, especially in the narrow `Y` direction. This was a real geometry issue, not a plotting artifact.

Implemented fix:
- `mmc3d_components.py` now provides rotated-AABB helpers: `component_half_extent`, `component_domain_margins`, `components_are_in_domain`, `project_component_to_domain`, and `project_params_to_domain`.
- `create_random_components(config, rng)` now samples strictly feasible randomized components by sampling size/orientation first, then sampling centers from the feasible AABB range.
- `create_initial_components(config)` now also passes through strict domain projection.
- `optimizer.py` projects parameters before FEM evaluation, eta-candidate evaluation, MMA candidate acceptance, callback export, and final saving.
- Existing `dataset_domain60_5x50` should be treated as diagnostic/invalid for final training because it contains out-of-domain component geometry.

Validation completed:
```powershell
python -B generate_dataset.py --num-trajectories 1 --max-iter 5 --output-dir D:\codex\workspace_code_v2\dataset_domain60_feasible_smoke_1x5 --results-dir D:\codex\workspace_code_v2\results_domain60_feasible_smoke_1x5
python -B generate_dataset.py --num-trajectories 1 --max-iter 50 --output-dir D:\codex\workspace_code_v2\dataset_domain60_feasible_1x50 --results-dir D:\codex\workspace_code_v2\results_domain60_feasible_1x50
```

Final `1 x 50` verification:
- Graph samples: `51` (`traj_0000_iter_0000_graph.npz` through `traj_0000_iter_0050_graph.npz`).
- Metadata: `DL=60.0`, `DW=4.0`, `DH=20.0`, `num_components=24`, `eta_candidates.shape=(6,)`.
- Process plots generated under `results_domain60_feasible_1x50\traj_0000\process_plots`: `initial_iter_0000.png`, `middle_iter_0025.png`, `final_iter_0050.png`.
- Rotated-AABB domain check over all 51 graph files: `bad_count=0`, `min_margin=0.0`.

Note: strict projection keeps geometry valid but can make the optimizer stall when several candidate steps project to the same boundary-feasible state. The verified `1 x 50` run plateaued after roughly iteration 31. Future work should consider smoother geometry constraints or narrower move limits for scale/orientation variables.

## 2026-07-14 Connected MMC Isosurface Validation

Implemented paper-style connected MMC topology visualization and connectivity guarding:
- Installed and recorded `scikit-image` for `skimage.measure.marching_cubes`.
- `visualization.py` now renders `phi=0` isosurfaces from the KS-aggregated global MMC TDF.
- Process plots now save both component-debug images and isosurface topology images.
- `mmc3d_components.py` now supports strictly feasible 24-component initialization for the cantilever setting. The latest experimental default for batched data generation is randomized scattered components; the 1 x 50 random-scattered test showed poor load-path connectivity, so the next production initializer should be cantilever-biased rather than copied from another beam benchmark or fully random.
- `optimizer.py` computes density connectivity with 6-neighbor connected components and includes `connected_to_load`, `spanning_ratio`, and `largest_component_ratio` in graph labels.
- MMA candidate acceptance now prefers volume-feasible candidates that do not worsen connectivity.

Validation commands completed:
```powershell
python -B generate_dataset.py --num-trajectories 1 --max-iter 5 --output-dir D:\codex\workspace_code_v2\dataset_domain60_connected_smoke_1x5 --results-dir D:\codex\workspace_code_v2\results_domain60_connected_smoke_1x5
python -B generate_dataset.py --num-trajectories 1 --max-iter 50 --output-dir D:\codex\workspace_code_v2\dataset_domain60_connected_1x50 --results-dir D:\codex\workspace_code_v2\results_domain60_connected_1x50
```

Final `1 x 50` result:
- Graph samples: `51`.
- Metadata: `DL=60.0`, `DW=4.0`, `DH=20.0`, `num_components=24`, `eta_candidates.shape=(6,)`.
- Connectivity over all graph files: `bad_connected=[]`, `min_spanning=1.0`.
- Compliance improved from `114.65327441733338` to `103.88421390338605`.
- Volume fraction changed from `0.39235007472465117` to `0.3818868275336585`.
- Final paper-style image: `D:\codex\workspace_code_v2\results_domain60_connected_1x50\traj_0000\process_plots\final_isosurface_iter_0050.png`.

## 2026-07-14 Random Initialization Connectivity Repair

The user requirement changed to preserving randomized initial MMC layouts while still converging to a continuous cantilever load path. A fixed beam skeleton is not acceptable for this goal.

Implemented changes:
- `create_random_components(config, rng)` now uses x-covered random initialization: the domain is split into `random_cover_segments=6`, with 24 components distributed as 4 randomized components per segment.
- `config.py` adds connection-control parameters:
  - `connectivity_density_threshold = 0.4`
  - `connection_penalty_initial = 1000.0`
  - `connection_penalty_final = 5.0`
  - `connection_penalty_decay_fraction = 0.6`
  - `volume_fill_weight = 200000.0`
- `optimizer.py` now augments the MMA objective with:
  - `connection_gap_penalty`, based on the minimum positive-gap path from the fixed face to the load point.
  - `volume_fill_penalty`, active when volume fraction is below `volfrac`.
- `_accept_mma_step()` now evaluates an additional connection-repair candidate when the current density field is not connected. This candidate slightly moves and thickens components along the minimum-gap path, then still goes through projection, FEM evaluation, volume checking, and connectivity ranking.
- Connectivity diagnostics now use `connectivity_density_threshold`, while process plots and isosurfaces remain visualized from the actual density/TDF fields.

Validation command:
```powershell
python -B generate_dataset.py --num-trajectories 1 --max-iter 50 --output-dir D:\codex\workspace_code_v2\dataset_connected_random_verified_1x50 --results-dir D:\codex\workspace_code_v2\results_connected_random_verified_1x50
```

Final verified result:
- Graph samples: `51`.
- Initial state: `connected_to_load=0`, `spanning_ratio=0.0`, `gap=4.3122`, `volume_fraction=0.2382`, `compliance=2889.33`.
- Final state: `connected_to_load=1`, `spanning_ratio=1.0`, `largest_component_ratio=1.0`, `gap=0.1645`, `volume_fraction=0.3139`, `compliance=1028.27`.
- Final isosurface image: `D:\codex\workspace_code_v2\results_connected_random_verified_1x50\traj_0000\process_plots\final_isosurface_iter_0050.png`.

Important interpretation:
- This does not fake cylindrical members or draw artificial bridges.
- The final image is still extracted from the true global implicit boundary `phi=0`.
- The repair candidate is an optimization candidate, not a visualization patch.
