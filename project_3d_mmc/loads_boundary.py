"""悬臂梁边界条件和载荷。"""

import numpy as np


def define_cantilever_boundary_conditions(nodes, DL, DW, DH, distributed=False, load_y=None, load_z=None):
    """左端 x=0 全固定，右端面指定位置附近施加向下力。"""
    tol = 1e-9
    fixed_nodes = np.where(np.abs(nodes[:, 0]) < tol)[0]
    fixed_dofs = np.sort(np.r_[fixed_nodes * 3, fixed_nodes * 3 + 1, fixed_nodes * 3 + 2])
    F = np.zeros(nodes.shape[0] * 3)
    right = np.where(np.abs(nodes[:, 0] - DL) < tol)[0]
    if load_y is None:
        load_y = DW / 2
    if load_z is None:
        load_z = DH / 2
    center = np.array([DL, load_y, load_z])
    if distributed:
        dist = np.linalg.norm(nodes[right] - center, axis=1)
        load_nodes = right[dist <= np.percentile(dist, 25)]
        F[load_nodes * 3 + 2] = -1.0 / len(load_nodes)
    else:
        load_node = right[np.argmin(np.linalg.norm(nodes[right] - center, axis=1))]
        F[load_node * 3 + 2] = -1.0
    return fixed_dofs, F
