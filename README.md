# ThesisCode_V4_0714

本仓库包含三维悬臂梁 Moving Morphable Component (MMC) 拓扑优化代码，以及基于图神经网络的候选步长预测研究路线。

主项目目录：

```text
project_3d_mmc/
```

请从以下文件开始阅读：

- [项目 README](project_3d_mmc/README.md)
- [开题报告](project_3d_mmc/OPENING_REPORT.md)
- [代码说明](project_3d_mmc/CODE_GUIDE.md)
- [交接文档](project_3d_mmc/HANDOFF.md)

当前技术路线保留 FEM、解析灵敏度和 MMA 优化方向，只让 GNN 预测候选步长或组件级步长缩放系数，使学习模型只控制“走多远”，不替代物理求解。
