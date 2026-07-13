"""轻量级可视化函数。"""

import os
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE_DIR / "results" / "mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _safe_savefig(save_path, dpi=160):
    """保存图片；沙箱或无字体缓存权限时跳过可视化输出。"""
    if not save_path:
        return
    try:
        plt.savefig(save_path, dpi=dpi)
    except OSError as exc:
        print(f"可视化保存失败，已跳过 {save_path}: {exc}")


def plot_density_slice(nodes, elements, densities, slice_axis="z", save_path=None):
    """绘制单元中心的密度散点切片。"""
    centers = nodes[elements].mean(axis=1)
    axis = {"x": 0, "y": 1, "z": 2}[slice_axis]
    mid = np.median(centers[:, axis])
    mask = np.abs(centers[:, axis] - mid) <= np.percentile(np.abs(centers[:, axis] - mid), 35)
    plt.figure(figsize=(6, 3))
    plt.scatter(centers[mask, 0], centers[mask, 1], c=densities[mask], cmap="viridis", s=30)
    plt.colorbar(label="density")
    plt.tight_layout()
    _safe_savefig(save_path)
    plt.close()


def plot_3d_density_voxels(densities, nelx, nely, nelz, save_path=None):
    """绘制密度大于 0.5 的体素中心散点。"""
    vals = densities.reshape(nelx, nely, nelz)
    pts = np.argwhere(vals > 0.5)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=vals[vals > 0.5], cmap="viridis", s=20)
    _safe_savefig(save_path)
    plt.close()


def plot_components_3d(components, save_path=None):
    """绘制 MMC 构件中心和主轴方向。"""
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    for c in components:
        x, y, z = c.center()
        ax.scatter([x], [y], [z], s=25)
        for v, l in zip(c.direction_vectors(), [c.L1, c.L2, c.L3]):
            ax.quiver(x, y, z, v[0], v[1], v[2], length=l, normalize=True)
    _safe_savefig(save_path)
    plt.close()


def plot_history(history, save_path=None):
    """绘制柔度和体积分数迭代曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(history["compliance"], marker="o")
    axes[0].set_title("Compliance")
    axes[1].plot(history["volume_fraction"], marker="o")
    axes[1].set_title("Volume fraction")
    plt.tight_layout()
    _safe_savefig(save_path)
    plt.close()
