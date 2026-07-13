"""3D Moving Morphable Component 几何定义。"""

import numpy as np


class MMCComponent3D:
    """三维可移动、可旋转、可缩放 MMC 构件。"""

    def __init__(self, x0, y0, z0, L1, L2, L3, alpha=0.0, beta=0.0, gamma=0.0, active=True):
        self.x0, self.y0, self.z0 = float(x0), float(y0), float(z0)
        self.L1, self.L2, self.L3 = max(float(L1), 1e-6), max(float(L2), 1e-6), max(float(L3), 1e-6)
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self.active = bool(active)

    def get_params(self):
        """返回 9 个设计变量。"""
        return np.array([self.x0, self.y0, self.z0, self.L1, self.L2, self.L3, self.alpha, self.beta, self.gamma], dtype=float)

    def set_params(self, params):
        """从 9 个设计变量更新构件。"""
        p = np.asarray(params, dtype=float)
        self.x0, self.y0, self.z0 = p[:3]
        self.L1, self.L2, self.L3 = np.maximum(p[3:6], 1e-6)
        self.alpha, self.beta, self.gamma = p[6:9]

    def rotation_matrix(self):
        """由欧拉角 alpha、beta、gamma 构造旋转矩阵。"""
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        cb, sb = np.cos(self.beta), np.sin(self.beta)
        cg, sg = np.cos(self.gamma), np.sin(self.gamma)
        rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        return rz @ ry @ rx

    def global_to_local(self, points):
        """将全局点坐标变换到构件局部坐标系。"""
        pts = np.asarray(points, dtype=float) - self.center()
        return pts @ self.rotation_matrix()

    def tdf(self, points, p=6):
        """计算超椭球式 TDF，正值表示点在构件内部。"""
        q = self.global_to_local(points)
        lengths = np.maximum(np.array([self.L1, self.L2, self.L3]), 1e-6)
        val = np.sum(np.abs(q / lengths) ** p, axis=1) ** (1.0 / p)
        return 1.0 - val if self.active else np.full(points.shape[0], -1e3)

    def volume(self):
        """返回构件包围盒尺度下的近似体积特征。"""
        return 8.0 * self.L1 * self.L2 * self.L3

    def center(self):
        """返回构件中心坐标。"""
        return np.array([self.x0, self.y0, self.z0], dtype=float)

    def direction_vectors(self):
        """返回三个局部主方向在全局坐标中的方向向量。"""
        return self.rotation_matrix().T

    def as_node_feature(self):
        """转换为 GNN 节点特征。"""
        return np.r_[self.get_params(), self.volume(), float(self.active)]


def create_initial_components(config):
    """生成悬臂梁初始 MMC 构件阵列。"""
    comps = []
    xs = np.linspace(0.35 * config.DL, 0.85 * config.DL, config.num_components)
    for i, x in enumerate(xs):
        y = config.DW * (0.35 + 0.3 * (i % 2))
        z = config.DH * (0.35 + 0.3 * ((i // 2) % 2))
        comps.append(MMCComponent3D(x, y, z, config.DL / 5, config.DW / 6, config.DH / 6))
    return comps


def create_random_components(config, rng):
    """Generate randomized MMC components for dataset trajectory diversity."""
    comps = []
    min_l = 0.7 * min(config.DL / config.nelx, config.DW / config.nely, config.DH / config.nelz)
    for _ in range(config.num_components):
        x = rng.uniform(0.15 * config.DL, 0.95 * config.DL)
        y = rng.uniform(0.15 * config.DW, 0.85 * config.DW)
        z = rng.uniform(0.15 * config.DH, 0.85 * config.DH)
        L1 = rng.uniform(max(min_l, 0.08 * config.DL), 0.28 * config.DL)
        L2 = rng.uniform(max(min_l, 0.06 * config.DW), 0.22 * config.DW)
        L3 = rng.uniform(max(min_l, 0.06 * config.DH), 0.22 * config.DH)
        alpha, beta, gamma = rng.uniform(-0.35 * np.pi, 0.35 * np.pi, size=3)
        comps.append(MMCComponent3D(x, y, z, L1, L2, L3, alpha, beta, gamma))
    return comps


def params_to_components(params, template_components):
    """将扁平参数数组转换为构件列表。"""
    params = np.asarray(params, dtype=float).reshape((-1, 9))
    comps = []
    for p, old in zip(params, template_components):
        c = MMCComponent3D(*old.get_params(), active=old.active)
        c.set_params(p)
        comps.append(c)
    return comps


def components_to_params(components):
    """将构件列表转换为扁平参数数组。"""
    return np.concatenate([c.get_params() for c in components])


def get_bounds(components, config):
    """为每个设计变量设置上下界。"""
    min_l = 0.5 * min(config.DL / config.nelx, config.DW / config.nely, config.DH / config.nelz)
    max_l = max(config.DL, config.DW, config.DH)
    bounds = []
    for _ in components:
        bounds += [(0, config.DL), (0, config.DW), (0, config.DH), (min_l, max_l), (min_l, max_l), (min_l, max_l),
                   (-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    return bounds
