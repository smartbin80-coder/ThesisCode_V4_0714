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
8. 普通训练 smoke 1 epoch 跑通。
9. MAML smoke 1 epoch 跑通。
10. 在线控制器 smoke test 跑通，安全回退后阻尼序列为 `0.8, 0.8, 0.8, 1.0`。

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
7. GAT 只允许预测步长缩放，不允许替代 FEM 或灵敏度方向。
8. 所有在线预测步长必须经 FEM 验证，不可靠时必须回退到保守候选搜索。
