"""3D-MMC 悬臂梁拓扑优化入口。"""

from pathlib import Path
import tempfile

import numpy as np
from config import config
from fem3d import generate_hex_mesh, hex8_element_stiffness
from loads_boundary import define_cantilever_boundary_conditions
from mmc3d_components import create_initial_components, params_to_components
from optimizer import MMCOptimizer3D
from utils import ensure_dir, set_random_seed
from visualization import plot_components_3d, plot_density_slice, plot_history


def prepare_output_dirs(cfg):
    """优先使用项目内输出目录；若沙箱拒绝写入，则回退到 workspace 根目录。"""
    for attr in ("dataset_dir", "results_dir"):
        candidates = [
            Path(getattr(cfg, attr)),
            Path(__file__).resolve().parent.parent / Path(getattr(cfg, attr)).name,
            Path(tempfile.gettempdir()) / "project_3d_mmc" / Path(getattr(cfg, attr)).name,
        ]
        for path in candidates:
            try:
                path.mkdir(parents=True, exist_ok=True)
                probe = path / ".write_probe.npz"
                np.savez(probe, ok=np.array([1]))
                probe.unlink(missing_ok=True)
                setattr(cfg, attr, str(path))
                break
            except OSError:
                continue


def main():
    """执行完整优化流程。"""
    prepare_output_dirs(config)
    ensure_dir(config.dataset_dir)
    ensure_dir(config.results_dir)
    set_random_seed(config.seed)

    nodes, elements = generate_hex_mesh(config.DL, config.DW, config.DH, config.nelx, config.nely, config.nelz)
    fixed_dofs, force_vector = define_cantilever_boundary_conditions(
        nodes,
        config.DL,
        config.DW,
        config.DH,
        load_y=getattr(config, "load_y", None),
        load_z=getattr(config, "load_z", None),
    )
    ke = hex8_element_stiffness(config.E0, config.nu, config.DL / config.nelx, config.DW / config.nely, config.DH / config.nelz)

    components = create_initial_components(config)
    plot_components_3d(components, f"{config.results_dir}/initial_components.png")

    optimizer = MMCOptimizer3D(config, components, nodes, elements, fixed_dofs, force_vector, ke)
    res, final_eval = optimizer.run()

    final_components = params_to_components(res.x, components)
    plot_components_3d(final_components, f"{config.results_dir}/final_components.png")
    plot_density_slice(nodes, elements, final_eval[2], save_path=f"{config.results_dir}/final_density_slice.png")
    plot_history(optimizer.history, save_path=f"{config.results_dir}/history.png")
    print(f"运行完成。结果保存在 {config.results_dir}，GNN 图数据保存在 {config.dataset_dir}。")


if __name__ == "__main__":
    main()
