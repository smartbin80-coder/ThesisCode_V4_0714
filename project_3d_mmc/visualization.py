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
from skimage.measure import marching_cubes

from tdf import compute_global_tdf


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


def _set_reference_axes(ax, config):
    """Style axes to match the reference MMC topology figures."""
    ax.set_xlim(0, config.DL)
    ax.set_ylim(0, config.DW)
    ax.set_zlim(0, config.DH)
    ax.set_xticks(np.arange(0, config.DL + 1e-9, 10.0))
    ax.set_yticks([0.0, config.DW / 2.0, config.DW])
    ax.set_zticks(np.arange(0, config.DH + 1e-9, 5.0))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((config.DL, config.DW, config.DH))
    ax.view_init(elev=22, azim=-55)
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
        axis.line.set_color("black")
        axis.line.set_linewidth(1.1)
    ax.tick_params(colors="black", labelsize=9, width=1.0)
    # Explicit black box axes. Matplotlib's pane edges are too faint when grid is off.
    ax.plot([0, config.DL], [0, 0], [0, 0], color="black", linewidth=1.0)
    ax.plot([0, 0], [0, config.DW], [0, 0], color="black", linewidth=1.0)
    ax.plot([0, 0], [0, 0], [0, config.DH], color="black", linewidth=1.0)


def _component_tube_surface(component, samples_u=28, samples_v=16):
    """Build the rendered surface from the actual MMC component dimensions."""
    u = np.linspace(-1.0, 1.0, samples_u)
    theta = np.linspace(0.0, 2.0 * np.pi, samples_v)
    uu, tt = np.meshgrid(u, theta, indexing="ij")
    cap_scale = np.sqrt(np.maximum(0.0, 1.0 - uu**2))
    local = np.stack(
        [
            component.L1 * uu,
            component.L2 * cap_scale * np.cos(tt),
            component.L3 * cap_scale * np.sin(tt),
        ],
        axis=-1,
    )
    pts = local @ component.rotation_matrix().T + component.center()
    return pts[..., 0], pts[..., 1], pts[..., 2]


def plot_topology_process(densities, config, title="", save_path=None, threshold=0.5, components=None):
    """Render a red 3D topology snapshot for initial/mid/final MMC states."""
    fig = plt.figure(figsize=(9, 5.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")

    if components:
        for component in components:
            xx, yy, zz = _component_tube_surface(component)
            ax.plot_surface(
                xx,
                yy,
                zz,
                color="#d00000",
                edgecolor="none",
                linewidth=0,
                antialiased=True,
                shade=True,
                alpha=1.0,
            )
    else:
        _plot_density_voxel_body(ax, densities, config, threshold)

    _set_reference_axes(ax, config)
    ax.set_title(title)
    plt.tight_layout()
    _safe_savefig(save_path, dpi=220)
    plt.close()


def plot_topology_isosurface(components, config, title="", save_path=None, resolution=None):
    """Render the true MMC implicit boundary phi=0 as a red isosurface."""
    if not components:
        return
    if resolution is None:
        resolution = getattr(config, "isosurface_resolution", (240, 24, 80))
    nx, ny, nz = (int(v) for v in resolution)
    x = np.linspace(0.0, config.DL, nx)
    y = np.linspace(0.0, config.DW, ny)
    z = np.linspace(0.0, config.DH, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    shape = str(getattr(config, "component_shape", "superellipsoid")).lower()
    p_norm = config.box_p_norm if shape == "box" else config.p_norm
    phi = compute_global_tdf(components, points, p_norm, config.ks_rho).reshape(nx, ny, nz)
    if np.min(phi) > 0.0 or np.max(phi) < 0.0:
        return

    spacing = (config.DL / (nx - 1), config.DW / (ny - 1), config.DH / (nz - 1))
    verts, faces, _, _ = marching_cubes(phi, level=0.0, spacing=spacing)
    fig = plt.figure(figsize=(9, 5.2), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        triangles=faces,
        color="#d00000",
        edgecolor="none",
        linewidth=0,
        antialiased=True,
        shade=True,
        alpha=1.0,
    )
    _set_reference_axes(ax, config)
    ax.set_title(title)
    plt.tight_layout()
    _safe_savefig(save_path, dpi=220)
    plt.close()


def _plot_density_voxel_body(ax, densities, config, threshold):
    """Fallback red body rendering when component geometry is unavailable."""
    vals = np.asarray(densities, dtype=float).reshape(config.nelx, config.nely, config.nelz)
    filled = vals >= float(threshold)
    if not np.any(filled):
        filled = vals >= np.percentile(vals, 75)

    x = np.linspace(0.0, config.DL, config.nelx + 1)
    y = np.linspace(0.0, config.DW, config.nely + 1)
    z = np.linspace(0.0, config.DH, config.nelz + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    facecolors = np.empty(filled.shape, dtype=object)
    facecolors[filled] = "#d40000"
    ax.voxels(xx, yy, zz, filled, facecolors=facecolors, edgecolor="#b00000", linewidth=0.08, shade=True)


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
