# Dataset 目录

本目录保存每次迭代导出的 GNN 图数据，文件名格式为 `iter_XXXX_graph.npz`。

核心字段：

- `node_features`: MMC 构件节点特征。
- `node_features_raw`: 未归一化的原始构件节点特征。
- `edge_index`: 双向边索引，形状兼容 PyTorch Geometric。
- `edge_attr`: 边特征。
- `global_features`: 图级全局特征，包括迭代进度、载荷位置、体积分数等。
- `compliance`, `volume_fraction`, `iteration`, `params`, `delta_params`: 当前优化状态标签。
- `eta_candidates`, `eta_label`, `compliance_candidates`, `volume_candidates`: GNN 步长预测训练标签。
- `eta_label_index`, `eta_feasible_mask`, `eta_failure_flag`, `response_targets`: 多任务训练标签。
- `trajectory_id`, `seed`, `num_components`, `load_y`, `load_z`: 轨迹元数据。

训练 GNN 步长预测模型时，可将 `node_features`, `edge_index`, `edge_attr` 作为输入，将 `eta_label` 作为监督标签。

项目提供 `pyg_dataset.py`，可直接将本目录的 `npz` 文件加载为 PyTorch Geometric `Data` 对象。

批量生成脚本会创建 `dataset_index.csv`。训练/验证/测试划分必须按 `trajectory_id` 划分，避免同一条优化轨迹的前后迭代样本泄漏。
