"""平滑 Heaviside 映射。"""

import numpy as np


def heaviside(phi, eps, alpha):
    """将 TDF 映射为密度，输出范围限制在 [alpha, 1]。"""
    phi = np.asarray(phi, dtype=float)
    rho = np.empty_like(phi)
    rho[phi > eps] = 1.0
    rho[phi < -eps] = alpha
    mask = np.abs(phi) <= eps
    s = phi[mask] / eps
    rho[mask] = alpha + (1.0 - alpha) * (0.5 + 0.75 * s - 0.25 * s**3)
    return np.clip(rho, alpha, 1.0)


def heaviside_derivative(phi, eps, alpha):
    """平滑 Heaviside 对 phi 的导数。"""
    phi = np.asarray(phi, dtype=float)
    d = np.zeros_like(phi)
    mask = np.abs(phi) <= eps
    s = phi[mask] / eps
    d[mask] = (1.0 - alpha) * (0.75 - 0.75 * s**2) / eps
    return d
