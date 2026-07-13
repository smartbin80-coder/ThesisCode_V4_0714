"""TDF 聚合与密度计算。"""

import numpy as np

from heaviside import heaviside, heaviside_derivative


def compute_component_tdfs(components, points, p):
    """返回所有构件在所有点上的 TDF，形状为 [num_components, num_points]。"""
    return np.vstack([c.tdf(points, p=p) for c in components])


def ks_aggregate(phi_all, ks_rho):
    """用数值稳定的 KS 函数近似 max。"""
    m = np.max(ks_rho * phi_all, axis=0, keepdims=True)
    return (np.log(np.sum(np.exp(ks_rho * phi_all - m), axis=0)) + m.ravel()) / ks_rho


def compute_global_tdf(components, points, p, ks_rho):
    """计算多构件聚合后的全局 TDF。"""
    return ks_aggregate(compute_component_tdfs(components, points, p), ks_rho)


def compute_element_density_from_tdf(phi_nodes, elements, heaviside_params):
    """根据节点 TDF 平均得到单元密度。"""
    rho_nodes = heaviside(phi_nodes, heaviside_params["eps"], heaviside_params["alpha"])
    return np.mean(rho_nodes[elements], axis=1)


def _rotation_derivatives(alpha, beta, gamma):
    """返回 Rz Ry Rx 及其对三个欧拉角的解析导数。"""
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)
    rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
    drx = np.array([[0, 0, 0], [0, -sa, -ca], [0, ca, -sa]])
    dry = np.array([[-sb, 0, cb], [0, 0, 0], [-cb, 0, -sb]])
    drz = np.array([[-sg, -cg, 0], [cg, -sg, 0], [0, 0, 0]])
    return rz @ ry @ rx, rz @ ry @ drx, rz @ dry @ rx, drz @ ry @ rx


def compute_global_tdf_with_sensitivities(components, points, p, ks_rho):
    """计算全局 TDF 及其对全部 MMC 参数的解析导数。"""
    n_comp = len(components)
    n_points = points.shape[0]
    phi_all = np.zeros((n_comp, n_points), dtype=float)
    dphi_all = np.zeros((n_comp, n_points, n_comp * 9), dtype=float)

    for i, comp in enumerate(components):
        center = comp.center()
        lengths = np.maximum(np.array([comp.L1, comp.L2, comp.L3], dtype=float), 1e-9)
        R, dRa, dRb, dRg = _rotation_derivatives(comp.alpha, comp.beta, comp.gamma)
        a = points - center
        q = a @ R
        abs_q = np.abs(q)
        scaled = abs_q / lengths
        s = np.sum(scaled**p, axis=1)
        s_safe = np.maximum(s, 1e-30)
        r = s_safe ** (1.0 / p)
        phi_all[i] = 1.0 - r if comp.active else -1e3
        if not comp.active:
            continue

        dr_dq = (s_safe ** (1.0 / p - 1.0))[:, None] * (scaled ** (p - 1.0)) * np.sign(q) / lengths
        dr_dL = -(s_safe ** (1.0 / p - 1.0))[:, None] * (abs_q**p) / (lengths ** (p + 1.0))
        base = i * 9
        # center variables: q=(x-c)R, so dphi/dc_k = dr/dq dot R[k, :]
        dphi_all[i, :, base + 0] = dr_dq @ R[0, :]
        dphi_all[i, :, base + 1] = dr_dq @ R[1, :]
        dphi_all[i, :, base + 2] = dr_dq @ R[2, :]
        dphi_all[i, :, base + 3:base + 6] = -dr_dL
        dphi_all[i, :, base + 6] = -np.sum(dr_dq * (a @ dRa), axis=1)
        dphi_all[i, :, base + 7] = -np.sum(dr_dq * (a @ dRb), axis=1)
        dphi_all[i, :, base + 8] = -np.sum(dr_dq * (a @ dRg), axis=1)

    m = np.max(ks_rho * phi_all, axis=0, keepdims=True)
    exp_shift = np.exp(ks_rho * phi_all - m)
    weights = exp_shift / np.sum(exp_shift, axis=0, keepdims=True)
    phi_global = (np.log(np.sum(exp_shift, axis=0)) + m.ravel()) / ks_rho
    dphi_global = np.einsum("in,inp->np", weights, dphi_all)
    return phi_global, dphi_global


def compute_density_with_sensitivities(phi_nodes, dphi_nodes, elements, heaviside_params):
    """计算单元密度及其对设计变量的解析导数。"""
    rho_nodes = heaviside(phi_nodes, heaviside_params["eps"], heaviside_params["alpha"])
    drho_dphi = heaviside_derivative(phi_nodes, heaviside_params["eps"], heaviside_params["alpha"])
    densities = np.mean(rho_nodes[elements], axis=1)
    weighted_node_sens = drho_dphi[:, None] * dphi_nodes
    drho_dparams = np.mean(weighted_node_sens[elements], axis=1)
    return densities, drho_dparams
