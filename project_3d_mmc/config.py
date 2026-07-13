"""3D-MMC 拓扑优化的全局配置。"""

from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass
class Config:
    """集中管理默认算例参数，默认规模面向普通电脑快速验证。"""

    # 设计域尺寸：长、宽、高
    DL: float = 4.0
    DW: float = 1.0
    DH: float = 1.0

    # 六面体网格数量
    nelx: int = 20
    nely: int = 6
    nelz: int = 6

    # 材料参数：实体弹性模量、弱材料模量、泊松比
    E0: float = 1.0
    Emin: float = 1e-3
    nu: float = 0.3

    # 体积分数上限
    volfrac: float = 0.4

    # MMC 构件与 TDF 参数
    num_components: int = 8
    min_components_for_dataset: int = 6
    max_components_for_dataset: int = 12
    p_norm: int = 6
    component_shape: str = "superellipsoid"
    box_p_norm: int = 24
    ks_rho: float = 60.0
    heaviside_eps: float = 0.08
    heaviside_alpha: float = 1e-3

    # 优化器参数
    max_iter: int = 20
    convergence_tol: float = 1e-3
    optimizer_type: str = "MMA"
    stop_on_convergence: bool = False

    # 数据保存参数
    save_every_iter: bool = True
    save_density: bool = True
    save_graph: bool = True
    eta_candidates: tuple = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
    feature_schema_version: int = 2
    load_prior_lambda: float = 2.0
    alpha_weight_decay: float = 1e-4
    alpha_freeze_epochs: int = 20
    trust_bias_eps: float = 1e-12
    trust_bias_threshold: float = 0.10
    node_step_label_mode: str = "global_broadcast"

    # 随机种子
    seed: int = 7
    trajectory_id: str = ""
    load_y: float = DW / 2
    load_z: float = DH / 2

    # 输出目录
    dataset_dir: str = str(PROJECT_DIR / "dataset")
    results_dir: str = str(PROJECT_DIR / "results")


config = Config()
