"""小规模 3D Hex8 有限元模块。"""

import numpy as np

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def generate_hex_mesh(DL, DW, DH, nelx, nely, nelz):
    """生成规则六面体网格。"""
    xs = np.linspace(0, DL, nelx + 1)
    ys = np.linspace(0, DW, nely + 1)
    zs = np.linspace(0, DH, nelz + 1)
    nodes = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)

    def nid(i, j, k):
        return i * (nely + 1) * (nelz + 1) + j * (nelz + 1) + k

    elements = []
    for i in range(nelx):
        for j in range(nely):
            for k in range(nelz):
                elements.append([nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                                 nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)])
    return nodes, np.asarray(elements, dtype=int)


def _elastic_matrix(E, nu):
    c = E / ((1 + nu) * (1 - 2 * nu))
    return c * np.array([[1 - nu, nu, nu, 0, 0, 0],
                         [nu, 1 - nu, nu, 0, 0, 0],
                         [nu, nu, 1 - nu, 0, 0, 0],
                         [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                         [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                         [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]])


def hex8_element_stiffness(E, nu, dx, dy, dz):
    """使用 2x2x2 高斯积分计算 Hex8 单元刚度矩阵。"""
    gp = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
    xi_nodes = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    D = _elastic_matrix(E, nu)
    ke = np.zeros((24, 24))
    jac = np.diag([dx / 2, dy / 2, dz / 2])
    inv_jac = np.diag([2 / dx, 2 / dy, 2 / dz])
    det_j = np.linalg.det(jac)
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                dN_nat = np.zeros((8, 3))
                for a, (xa, ya, za) in enumerate(xi_nodes):
                    dN_nat[a] = [xa * (1 + ya * eta) * (1 + za * zeta),
                                 ya * (1 + xa * xi) * (1 + za * zeta),
                                 za * (1 + xa * xi) * (1 + ya * eta)]
                dN = 0.125 * dN_nat @ inv_jac
                B = np.zeros((6, 24))
                for a in range(8):
                    ix = 3 * a
                    B[:, ix:ix + 3] = [[dN[a, 0], 0, 0], [0, dN[a, 1], 0], [0, 0, dN[a, 2]],
                                       [dN[a, 1], dN[a, 0], 0], [0, dN[a, 2], dN[a, 1]], [dN[a, 2], 0, dN[a, 0]]]
                ke += B.T @ D @ B * det_j
    return ke


def assemble_global_stiffness(nodes, elements, densities, ke, E0, Emin):
    """组装全局稀疏刚度矩阵。"""
    ndof = nodes.shape[0] * 3
    rows, cols, vals = [], [], []
    for e, conn in enumerate(elements):
        dofs = np.repeat(conn * 3, 3) + np.tile([0, 1, 2], 8)
        scale = Emin + densities[e] * (E0 - Emin)
        kk = ke * scale
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.ravel())
        cols.extend(cc.ravel())
        vals.extend(kk.ravel())
    K = coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K = K + coo_matrix((np.full(ndof, 1e-9), (np.arange(ndof), np.arange(ndof))), shape=(ndof, ndof)).tocsr()
    return K


def solve_displacement(K, F, fixed_dofs):
    """施加固定位移边界并求解位移。"""
    ndof = F.size
    fixed = np.asarray(fixed_dofs, dtype=int)
    free = np.setdiff1d(np.arange(ndof), fixed)
    U = np.zeros(ndof)
    U[free] = spsolve(K[free][:, free], F[free])
    return U


def compute_compliance(F, U):
    """计算柔度 F^T U。"""
    return float(F @ U)


def compute_element_strain_energy(U, elements, ke, densities):
    """计算单元应变能，用于后续灵敏度或可视化。"""
    out = np.zeros(len(elements))
    for e, conn in enumerate(elements):
        dofs = np.repeat(conn * 3, 3) + np.tile([0, 1, 2], 8)
        ue = U[dofs]
        out[e] = densities[e] * (ue @ ke @ ue)
    return out


def compute_compliance_density_gradient(U, elements, ke, E0, Emin):
    """解析柔度灵敏度 dC/drho_e = -(E0-Emin) * u_e^T k0 u_e。"""
    grad = np.zeros(len(elements), dtype=float)
    for e, conn in enumerate(elements):
        dofs = np.repeat(conn * 3, 3) + np.tile([0, 1, 2], 8)
        ue = U[dofs]
        grad[e] = -(E0 - Emin) * float(ue @ ke @ ue)
    return grad


def compute_volume_fraction(densities):
    """计算平均单元密度作为体积分数。"""
    return float(np.mean(densities))
