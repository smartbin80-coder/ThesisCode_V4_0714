"""3D-MMC optimizer with analytic sensitivities, MMA, and GNN step labels."""

from pathlib import Path
import heapq

import numpy as np
from scipy.ndimage import generate_binary_structure, label
from scipy.optimize import minimize

from fem3d import (
    assemble_global_stiffness,
    compute_compliance,
    compute_compliance_density_gradient,
    compute_element_strain_energy,
    compute_volume_fraction,
    solve_displacement,
)
from graph_export import build_component_graph, build_component_step_labels, build_graph_auxiliary_features, save_graph_npz
from mma import MMASolver
from mmc3d_components import component_half_extent, components_to_params, get_bounds, params_to_components, project_params_to_domain
from tdf import compute_density_with_sensitivities, compute_element_density_from_tdf, compute_global_tdf, compute_global_tdf_with_sensitivities
from utils import ensure_dir, save_json
from visualization import plot_topology_isosurface, plot_topology_process


def apply_gnn_step_scale(params_old, params_new, eta):
    """Scale a traditional optimizer step for future GNN step-size prediction."""
    return np.asarray(params_old) + float(eta) * (np.asarray(params_new) - np.asarray(params_old))


class MMCOptimizer3D:
    """MMC parameter evaluation, analytic-sensitivity optimization, and data export."""

    def __init__(self, config, components, nodes, elements, fixed_dofs, force_vector, ke):
        self.config = config
        self.components = components
        self.nodes = nodes
        self.elements = elements
        self.fixed_dofs = fixed_dofs
        self.F = force_vector
        self.ke = ke
        self.history = {"compliance": [], "volume_fraction": [], "params": []}
        self.iteration = 0
        self.prev_params = None
        self._cache_params = None
        self._cache_eval = None
        ensure_dir(config.dataset_dir)
        ensure_dir(config.results_dir)

    def evaluate(self, params):
        """Evaluate compliance, volume fraction, densities, displacement, and graph data."""
        params = self._project_params(params)
        comps = params_to_components(params, self.components)
        phi = compute_global_tdf(comps, self.nodes, self._tdf_p_norm(), self.config.ks_rho)
        densities = compute_element_density_from_tdf(
            phi, self.elements, {"eps": self.config.heaviside_eps, "alpha": self.config.heaviside_alpha}
        )
        K = assemble_global_stiffness(self.nodes, self.elements, densities, self.ke, self.config.E0, self.config.Emin)
        U = solve_displacement(K, self.F, self.fixed_dofs)
        compliance = compute_compliance(self.F, U)
        volume_fraction = compute_volume_fraction(densities)
        graph_data = build_component_graph(
            comps,
            self.config,
            iteration=self.iteration,
            compliance=compliance,
            volume_fraction=volume_fraction,
        )
        connectivity = self._connectivity_metrics(densities)
        return compliance, volume_fraction, densities, U, graph_data, comps, connectivity

    def evaluate_with_sensitivities(self, params):
        """Evaluate response and analytic gradients with respect to MMC design variables."""
        params = self._project_params(params)
        if self._cache_params is not None and np.array_equal(params, self._cache_params):
            return self._cache_eval

        comps = params_to_components(params, self.components)
        phi, dphi_dparams = compute_global_tdf_with_sensitivities(comps, self.nodes, self._tdf_p_norm(), self.config.ks_rho)
        densities, drho_dparams = compute_density_with_sensitivities(
            phi,
            dphi_dparams,
            self.elements,
            {"eps": self.config.heaviside_eps, "alpha": self.config.heaviside_alpha},
        )
        K = assemble_global_stiffness(self.nodes, self.elements, densities, self.ke, self.config.E0, self.config.Emin)
        U = solve_displacement(K, self.F, self.fixed_dofs)
        compliance = compute_compliance(self.F, U)
        volume_fraction = compute_volume_fraction(densities)

        dcompliance_drho = compute_compliance_density_gradient(U, self.elements, self.ke, self.config.E0, self.config.Emin)
        compliance_grad = dcompliance_drho @ drho_dparams
        volume_grad = np.mean(drho_dparams, axis=0)
        element_strain_energy = compute_element_strain_energy(U, self.elements, self.ke, densities)
        gap_penalty, gap_grad, gap_path = self._connection_gap_penalty_and_grad(comps)
        lambda_conn = self._connection_penalty_weight()
        volume_deficit = max(0.0, float(self.config.volfrac) - float(volume_fraction))
        volume_fill_weight = float(getattr(self.config, "volume_fill_weight", 0.0))
        volume_fill_penalty = volume_fill_weight * volume_deficit * volume_deficit
        objective_value = float(compliance + lambda_conn * gap_penalty + volume_fill_penalty)
        objective_grad = np.asarray(compliance_grad, dtype=float) + lambda_conn * np.asarray(gap_grad, dtype=float)
        if volume_deficit > 0.0 and volume_fill_weight > 0.0:
            objective_grad -= 2.0 * volume_fill_weight * volume_deficit * np.asarray(volume_grad, dtype=float)

        out = {
            "objective": objective_value,
            "compliance": float(compliance),
            "volume_fraction": float(volume_fraction),
            "densities": densities,
            "U": U,
            "components": comps,
            "connection_gap_penalty": float(gap_penalty),
            "connection_gap_path": np.asarray(gap_path, dtype=int),
            "connection_penalty_weight": float(lambda_conn),
            "volume_fill_penalty": float(volume_fill_penalty),
            "element_strain_energy": element_strain_energy,
            "component_strain_energy_norm": self._component_strain_energy_norm(densities, drho_dparams, element_strain_energy),
            "connectivity": self._connectivity_metrics(densities),
            "graph_data": build_component_graph(
                comps,
                self.config,
                compliance_grad=compliance_grad,
                volume_grad=volume_grad,
                iteration=self.iteration,
                compliance=compliance,
                volume_fraction=volume_fraction,
            ),
            "compliance_grad": np.asarray(compliance_grad, dtype=float),
            "objective_grad": np.asarray(objective_grad, dtype=float),
            "volume_grad": np.asarray(volume_grad, dtype=float),
        }
        self._cache_params = params.copy()
        self._cache_eval = out
        return out

    def objective(self, params):
        """SLSQP objective."""
        return self.evaluate_with_sensitivities(params)["objective"]

    def objective_jac(self, params):
        """Analytic objective gradient for SLSQP."""
        return self.evaluate_with_sensitivities(params)["objective_grad"]

    def constraint_volume(self, params):
        """SLSQP volume constraint: non-negative means feasible."""
        return self.config.volfrac - self.evaluate_with_sensitivities(params)["volume_fraction"]

    def constraint_volume_jac(self, params):
        """Analytic gradient of SLSQP volume constraint."""
        return -self.evaluate_with_sensitivities(params)["volume_grad"]

    def _evaluate_eta_candidates(self, params_old, params_new):
        """Evaluate candidate step scales and label the feasible minimum-compliance eta."""
        eta_candidates = np.asarray(self.config.eta_candidates, dtype=float)
        compliance_candidates = []
        volume_candidates = []
        for eta in eta_candidates:
            p_eta = self._project_params(apply_gnn_step_scale(params_old, params_new, eta))
            c_eta, v_eta, *_ = self.evaluate(p_eta)
            compliance_candidates.append(c_eta)
            volume_candidates.append(v_eta)
        compliance_candidates = np.asarray(compliance_candidates, dtype=float)
        volume_candidates = np.asarray(volume_candidates, dtype=float)
        feasible = volume_candidates <= self.config.volfrac + 1e-8
        if np.any(feasible):
            feasible_ids = np.where(feasible)[0]
            best_id = feasible_ids[np.argmin(compliance_candidates[feasible_ids])]
            eta_label = float(eta_candidates[best_id])
            failure_flag = 0
        else:
            best_id = -1
            eta_label = 0.0
            failure_flag = 1
        response_targets = np.column_stack([compliance_candidates, volume_candidates])
        return eta_candidates, eta_label, int(best_id), feasible.astype(np.int8), int(failure_flag), compliance_candidates, volume_candidates, response_targets

    def _trust_metrics(self, current_compliance, previous_compliance, eta_label_index, compliance_candidates):
        """Compute trust-bias fields for online adaptation triggers."""
        if previous_compliance is None or eta_label_index < 0 or eta_label_index >= len(compliance_candidates):
            return 0.0, 0.0, 0.0
        actual_delta = float(previous_compliance - current_compliance)
        predicted_delta = float(previous_compliance - compliance_candidates[eta_label_index])
        eps = float(getattr(self.config, "trust_bias_eps", 1e-12))
        trust_bias = abs(actual_delta - predicted_delta) / max(abs(predicted_delta), eps)
        return actual_delta, predicted_delta, float(trust_bias)

    def _component_strain_energy_norm(self, densities, drho_dparams, element_strain_energy):
        """Aggregate element strain energy to components using density sensitivities as soft support weights."""
        n_comp = len(self.components)
        if drho_dparams.size == 0 or element_strain_energy.size == 0:
            return np.zeros(n_comp, dtype=float)
        support = np.abs(np.asarray(drho_dparams, dtype=float).reshape(len(element_strain_energy), n_comp, 9)).sum(axis=2)
        weighted = support * np.asarray(element_strain_energy, dtype=float)[:, None]
        denom = np.sum(support, axis=0) + 1e-12
        comp_energy = np.sum(weighted, axis=0) / denom
        if np.max(comp_energy) > 0:
            comp_energy = comp_energy / np.max(comp_energy)
        return comp_energy.astype(float)

    def callback(self, params):
        """Save iteration history, density field, and graph npz labels."""
        params = self._project_params(params)
        eval_data = self.evaluate_with_sensitivities(params)
        compliance = eval_data["compliance"]
        volume_fraction = eval_data["volume_fraction"]
        densities = eval_data["densities"]
        self.history["compliance"].append(compliance)
        self.history["volume_fraction"].append(volume_fraction)
        self.history["params"].append(params.copy())

        if self.prev_params is None:
            previous_compliance = None
            eta_candidates = np.asarray(self.config.eta_candidates, dtype=float)
            eta_label = 1.0
            eta_label_index = int(np.argmin(np.abs(eta_candidates - eta_label)))
            eta_feasible_mask = np.ones_like(eta_candidates, dtype=np.int8)
            eta_failure_flag = 0
            compliance_candidates = np.full_like(eta_candidates, compliance, dtype=float)
            volume_candidates = np.full_like(eta_candidates, volume_fraction, dtype=float)
            response_targets = np.column_stack([compliance_candidates, volume_candidates])
            delta_params = np.zeros_like(params)
        else:
            previous_compliance = self.history["compliance"][-2] if len(self.history["compliance"]) >= 2 else None
            delta_params = params - self.prev_params
            (
                eta_candidates,
                eta_label,
                eta_label_index,
                eta_feasible_mask,
                eta_failure_flag,
                compliance_candidates,
                volume_candidates,
                response_targets,
            ) = self._evaluate_eta_candidates(self.prev_params, params)
        trust_actual_delta, trust_predicted_delta, trust_bias = self._trust_metrics(
            compliance,
            previous_compliance,
            eta_label_index,
            compliance_candidates,
        )

        graph_data = build_component_graph(
            eval_data["components"],
            self.config,
            compliance_grad=eval_data["compliance_grad"],
            volume_grad=eval_data["volume_grad"],
            delta_params=delta_params,
            iteration=self.iteration,
            compliance=compliance,
            volume_fraction=volume_fraction,
        )

        print(
            f"iter {self.iteration:04d} | compliance={compliance:.6e} | "
            f"volume={volume_fraction:.4f} | constraint={self.config.volfrac - volume_fraction:.4f} | "
            f"gap={eval_data['connection_gap_penalty']:.4e} | eta_label={eta_label:.2f}"
        )

        if self.config.save_graph:
            node_features, node_features_raw, edge_index, edge_attr, global_features = graph_data
            eta_node_label, eta_node_label_index = build_component_step_labels(
                eta_label,
                eta_label_index,
                len(eval_data["components"]),
            )
            auxiliary = build_graph_auxiliary_features(eval_data["components"], edge_index, self.config)
            labels = {
                "compliance": compliance,
                "volume_fraction": volume_fraction,
                "iteration": self.iteration,
                "params": params,
                "delta_params": delta_params,
                "eta_candidates": eta_candidates,
                "eta_label": eta_label,
                "eta_label_index": eta_label_index,
                "eta_feasible_mask": eta_feasible_mask,
                "eta_failure_flag": eta_failure_flag,
                "compliance_candidates": compliance_candidates,
                "volume_candidates": volume_candidates,
                "response_targets": response_targets,
                "eta_node_label": eta_node_label,
                "eta_node_label_index": eta_node_label_index,
                "trust_actual_delta": trust_actual_delta,
                "trust_predicted_delta": trust_predicted_delta,
                "trust_bias": trust_bias,
                "component_strain_energy_norm": eval_data["component_strain_energy_norm"],
                "connection_gap_penalty": float(eval_data["connection_gap_penalty"]),
                "connection_penalty_weight": float(eval_data["connection_penalty_weight"]),
                "connection_gap_path": eval_data["connection_gap_path"],
                "volume_fill_penalty": float(eval_data["volume_fill_penalty"]),
                "connected_to_load": int(eval_data["connectivity"]["connected_to_load"]),
                "spanning_ratio": float(eval_data["connectivity"]["spanning_ratio"]),
                "largest_component_ratio": float(eval_data["connectivity"]["largest_component_ratio"]),
                "trajectory_id": np.asarray(str(getattr(self.config, "trajectory_id", ""))),
                "seed": int(getattr(self.config, "seed", -1)),
                "num_components": int(getattr(self.config, "num_components", len(eval_data["components"]))),
                "load_y": float(getattr(self.config, "load_y", self.config.DW / 2)),
                "load_z": float(getattr(self.config, "load_z", self.config.DH / 2)),
                "DL": float(self.config.DL),
                "DW": float(self.config.DW),
                "DH": float(self.config.DH),
                "nelx": int(self.config.nelx),
                "nely": int(self.config.nely),
                "nelz": int(self.config.nelz),
                "E0": float(self.config.E0),
                "Emin": float(self.config.Emin),
                "nu": float(self.config.nu),
                "volfrac": float(self.config.volfrac),
                "max_iter": int(self.config.max_iter),
            }
            labels.update(auxiliary)
            save_graph_npz(
                Path(self.config.dataset_dir) / self._graph_file_name(),
                node_features,
                node_features_raw,
                edge_index,
                edge_attr,
                global_features,
                labels,
            )

        if self.config.save_density:
            np.save(Path(self.config.results_dir) / f"iter_{self.iteration:04d}_density.npy", densities)

        self._save_process_plot_if_needed(densities, eval_data["components"])

        self.prev_params = params.copy()
        self.iteration += 1

    def _save_process_plot_if_needed(self, densities, components):
        """Save initial/mid/final red 3D topology snapshots for one trajectory."""
        if not bool(getattr(self.config, "save_process_plots", True)):
            return
        max_iter = int(getattr(self.config, "max_iter", 0))
        mid_iter = max(1, max_iter // 2)
        stages = {0: "initial", mid_iter: "middle", max_iter: "final"}
        stage = stages.get(int(self.iteration))
        if stage is None:
            return
        out_dir = Path(self.config.results_dir) / "process_plots"
        ensure_dir(out_dir)
        threshold = float(
            getattr(
                self.config,
                "connectivity_density_threshold",
                getattr(self.config, "process_plot_density_threshold", 0.5),
            )
        )
        title = f"{stage.capitalize()} topology | iter {self.iteration:04d}"
        component_path = out_dir / f"{stage}_components_iter_{self.iteration:04d}.png"
        isosurface_path = out_dir / f"{stage}_isosurface_iter_{self.iteration:04d}.png"
        plot_topology_process(densities, self.config, title=title, save_path=component_path, threshold=threshold, components=components)
        plot_topology_isosurface(components, self.config, title=title, save_path=isosurface_path)

    def _save_final(self, result, final_params):
        """Save final arrays and summary."""
        final_eval = self.evaluate(final_params)
        np.save(Path(self.config.results_dir) / "final_params.npy", final_params)
        save_json(
            Path(self.config.results_dir) / "summary.json",
            {
                "success": bool(getattr(result, "success", False)),
                "message": str(getattr(result, "message", "")),
                "optimizer_type": self.config.optimizer_type,
                "final_compliance": float(final_eval[0]),
                "final_volume_fraction": float(final_eval[1]),
                "iterations_saved": self.iteration,
            },
        )
        return final_eval

    def _graph_file_name(self):
        """Build a graph filename that preserves trajectory identity."""
        trajectory_id = str(getattr(self.config, "trajectory_id", "") or "")
        if trajectory_id:
            return f"{trajectory_id}_iter_{self.iteration:04d}_graph.npz"
        return f"iter_{self.iteration:04d}_graph.npz"

    def _run_mma(self, x0):
        """Run the analytic-sensitivity MMA optimizer."""
        class Result:
            pass

        bounds = np.asarray(get_bounds(self.components, self.config), dtype=float)
        solver = MMASolver(bounds[:, 0], bounds[:, 1], move_limit=0.12)
        x = self._project_params(x0)
        success = True
        message = "MMA converged by max iteration."
        for _ in range(self.config.max_iter):
            data = self.evaluate_with_sensitivities(x)
            g = data["volume_fraction"] - self.config.volfrac
            raw = self._project_params(solver.update(x, data["objective"], data["objective_grad"], g, data["volume_grad"]))
            xnew = self._accept_mma_step(x, raw, data)
            change = float(np.max(np.abs(xnew - x)))
            x = xnew
            self.callback(x)
            if self.config.stop_on_convergence and change < self.config.convergence_tol:
                message = "MMA converged by parameter change tolerance."
                break
        res = Result()
        res.x = x
        res.success = success
        res.message = message
        return res

    def _accept_mma_step(self, x, raw, current_data):
        """Use real FEM responses to conservatively accept an MMA step."""
        old_v = current_data["volume_fraction"]
        old_c = current_data["compliance"]
        old_connectivity = current_data.get("connectivity", {"connected_to_load": False, "spanning_ratio": 0.0})
        old_gap = float(current_data.get("connection_gap_penalty", self._connection_gap_penalty(x)))
        old_violation = max(0.0, old_v - self.config.volfrac)
        candidates = []
        eta_list = tuple(float(eta) for eta in self.config.eta_candidates) + (0.0,)
        for eta in sorted(set(eta_list), reverse=True):
            trial = self._project_params(apply_gnn_step_scale(x, raw, eta))
            c_trial, v_trial, densities_trial, *_ = self.evaluate(trial)
            violation = max(0.0, v_trial - self.config.volfrac)
            connectivity = self._connectivity_metrics(densities_trial)
            gap_trial = self._connection_gap_penalty(trial)
            candidates.append((eta, trial, c_trial, v_trial, violation, connectivity, gap_trial))
        if not bool(old_connectivity["connected_to_load"]):
            repair = self._connection_repair_params(x)
            c_trial, v_trial, densities_trial, *_ = self.evaluate(repair)
            violation = max(0.0, v_trial - self.config.volfrac)
            connectivity = self._connectivity_metrics(densities_trial)
            gap_trial = self._connection_gap_penalty(repair)
            candidates.append((-1.0, repair, c_trial, v_trial, violation, connectivity, gap_trial))

        def connectivity_loss(item):
            conn = item[5]
            loss = max(0.0, float(old_connectivity["spanning_ratio"]) - float(conn["spanning_ratio"]))
            if bool(old_connectivity["connected_to_load"]) and not bool(conn["connected_to_load"]):
                loss += 1.0
            return loss

        def gap_loss(item):
            return max(0.0, float(item[6]) - old_gap)

        def volume_deficit(item):
            return max(0.0, float(self.config.volfrac) - float(item[3]))

        def nonworse_connectivity(item):
            conn = item[5]
            if bool(conn["connected_to_load"]) and not bool(old_connectivity["connected_to_load"]):
                return True
            if bool(old_connectivity["connected_to_load"]) and not bool(conn["connected_to_load"]):
                return False
            return float(conn["spanning_ratio"]) >= float(old_connectivity["spanning_ratio"]) - 1e-8

        def connected_key(item):
            conn = item[5]
            return (-int(bool(conn["connected_to_load"])), -float(conn["spanning_ratio"]), item[6], volume_deficit(item), item[2])

        feasible = [item for item in candidates if item[4] <= 1e-8]
        if old_violation <= 1e-8 and feasible:
            nonworse = [item for item in feasible if nonworse_connectivity(item)]
            if bool(old_connectivity["connected_to_load"]):
                connected = [item for item in feasible if bool(item[5]["connected_to_load"])]
                eta, trial, *_ = min(
                    connected or nonworse or feasible,
                    key=lambda item: (connectivity_loss(item), gap_loss(item), *connected_key(item)),
                )
            else:
                eta, trial, *_ = min(
                    nonworse or feasible,
                    key=lambda item: (gap_loss(item), item[6], connectivity_loss(item), *connected_key(item)),
                )
            return trial

        positive_candidates = [item for item in candidates if item[0] > 0.0]
        improving_violation = [item for item in positive_candidates if item[4] <= old_violation + 1e-10]
        if improving_violation:
            nonworse = [item for item in improving_violation if nonworse_connectivity(item)]
            eta, trial, *_ = min(
                nonworse or improving_violation,
                key=lambda item: (item[4], gap_loss(item), item[6], connectivity_loss(item), *connected_key(item)),
            )
            return trial

        _, trial, *_ = min(
            positive_candidates or candidates,
            key=lambda item: (gap_loss(item), item[6], connectivity_loss(item), item[4], *connected_key(item)),
        )
        return trial

    def _run_slsqp(self, x0):
        """Run SLSQP using the same analytic gradients."""
        return minimize(
            self.objective,
            x0,
            method="SLSQP",
            jac=self.objective_jac,
            bounds=get_bounds(self.components, self.config),
            constraints={"type": "ineq", "fun": self.constraint_volume, "jac": self.constraint_volume_jac},
            callback=self.callback,
            options={"maxiter": self.config.max_iter, "ftol": self.config.convergence_tol, "disp": True},
        )

    def run(self):
        """Execute optimization and save final results."""
        x0 = self._project_params(components_to_params(self.components))
        self.prev_params = None
        self.callback(x0)
        if self.config.optimizer_type.upper() == "MMA":
            result = self._run_mma(x0)
        elif self.config.optimizer_type.upper() == "SLSQP":
            result = self._run_slsqp(x0)
        else:
            raise ValueError(f"Unsupported optimizer_type: {self.config.optimizer_type}")
        final_params = self._project_params(result.x)
        final_eval = self._save_final(result, final_params)
        return result, final_eval

    def _project_params(self, params):
        """Project flattened MMC parameters to the strict design-domain constraint."""
        projected = project_params_to_domain(params, self.components, self.config)
        return np.asarray(projected, dtype=float)

    def _connection_penalty_weight(self):
        """Continuation schedule for the random-layout connection penalty."""
        initial = float(getattr(self.config, "connection_penalty_initial", 50.0))
        final = float(getattr(self.config, "connection_penalty_final", 1.0))
        max_iter = max(1, int(getattr(self.config, "max_iter", 1)))
        decay_fraction = float(getattr(self.config, "connection_penalty_decay_fraction", 0.7))
        decay_steps = max(1.0, decay_fraction * max_iter)
        t = min(1.0, max(0.0, float(self.iteration) / decay_steps))
        return (1.0 - t) * initial + t * final

    def _component_support_radius(self, component, direction):
        """Approximate component reach along a global direction."""
        d = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(d))
        if norm <= 1e-12:
            return 0.0
        d = d / norm
        q = d @ component.rotation_matrix()
        lengths = np.maximum(np.array([component.L1, component.L2, component.L3], dtype=float), 1e-9)
        p = max(2.0, float(self._tdf_p_norm()))
        denom = float(np.sum(np.abs(q / lengths) ** p) ** (1.0 / p))
        return 1.0 / max(denom, 1e-12)

    def _edge_gap(self, comps, a, b):
        """Return signed gap for one path edge; negative means overlap/reach."""
        n = len(comps)
        load = np.array([self.config.DL, float(getattr(self.config, "load_y", self.config.DW / 2)), float(getattr(self.config, "load_z", self.config.DH / 2))])
        fixed = np.array([0.0, load[1], load[2]])
        if a == -1 and 0 <= b < n:
            half = component_half_extent(comps[b])
            return float(comps[b].x0 - half[0])
        if b == -1 and 0 <= a < n:
            half = component_half_extent(comps[a])
            return float(comps[a].x0 - half[0])
        if a == -1:
            ca = fixed
            ra = 0.0
        elif a == n:
            ca = load
            ra = 0.0
        else:
            ca = comps[a].center()
            ra = None
        if b == -1:
            cb = fixed
            rb = 0.0
        elif b == n:
            cb = load
            rb = 0.0
        else:
            cb = comps[b].center()
            rb = None
        vec = cb - ca
        dist = float(np.linalg.norm(vec))
        if ra is None:
            ra = self._component_support_radius(comps[a], vec)
        if rb is None:
            rb = self._component_support_radius(comps[b], -vec)
        return dist - float(ra) - float(rb)

    def _minimum_gap_path(self, comps):
        """Find a left-to-right component chain with minimum positive gap."""
        n = len(comps)
        xs = np.array([comp.x0 for comp in comps], dtype=float)
        adjacency = {-1: []}
        for i in range(n):
            adjacency[i] = []
        for i in range(n):
            gap = max(0.0, self._edge_gap(comps, -1, i))
            adjacency[-1].append((gap, i))
            gap_load = max(0.0, self._edge_gap(comps, i, n))
            adjacency[i].append((gap_load, n))
        for i in range(n):
            for j in range(n):
                if xs[j] <= xs[i] + 1e-9:
                    continue
                gap = max(0.0, self._edge_gap(comps, i, j))
                distance_bias = 1e-4 * float(np.linalg.norm(comps[j].center() - comps[i].center())) / max(self.config.DL, 1e-9)
                adjacency[i].append((gap + distance_bias, j))

        heap = [(0.0, -1)]
        prev = {}
        best = {-1: 0.0}
        while heap:
            cost, node = heapq.heappop(heap)
            if node == n:
                break
            if cost > best.get(node, np.inf) + 1e-12:
                continue
            for weight, nxt in adjacency.get(node, []):
                new_cost = cost + float(weight)
                if new_cost < best.get(nxt, np.inf):
                    best[nxt] = new_cost
                    prev[nxt] = node
                    heapq.heappush(heap, (new_cost, nxt))
        if n not in best:
            return [(-1, n)]
        nodes = [n]
        while nodes[-1] != -1:
            nodes.append(prev[nodes[-1]])
        nodes.reverse()
        return list(zip(nodes[:-1], nodes[1:]))

    def _connection_gap_penalty_and_grad(self, comps):
        """Penalty that pulls one random component chain into a continuous load path."""
        n = len(comps)
        grad = np.zeros(n * 9, dtype=float)
        beta = float(getattr(self.config, "connection_gap_softplus_beta", 4.0))
        weight = float(getattr(self.config, "connection_gap_weight", 1.0))
        path = self._minimum_gap_path(comps)
        penalty = 0.0
        load = np.array([self.config.DL, float(getattr(self.config, "load_y", self.config.DW / 2)), float(getattr(self.config, "load_z", self.config.DH / 2))])
        fixed = np.array([0.0, load[1], load[2]])

        for a, b in path:
            gap = self._edge_gap(comps, a, b)
            soft = float(np.logaddexp(0.0, beta * gap) / beta)
            sigmoid = float(1.0 / (1.0 + np.exp(-np.clip(beta * gap, -60.0, 60.0))))
            penalty += weight * soft * soft
            coeff = weight * 2.0 * soft * sigmoid
            if coeff <= 1e-12:
                continue
            if a == -1:
                ca = fixed
            elif a == n:
                ca = load
            else:
                ca = comps[a].center()
            if b == -1:
                cb = fixed
            elif b == n:
                cb = load
            else:
                cb = comps[b].center()
            vec = cb - ca
            dist = float(np.linalg.norm(vec))
            if dist <= 1e-12:
                continue
            direction = vec / dist
            if 0 <= a < n:
                grad[9 * a : 9 * a + 3] -= coeff * direction
            if 0 <= b < n:
                grad[9 * b : 9 * b + 3] += coeff * direction
        flat_path = []
        for a, b in path:
            flat_path.extend([a, b])
        return float(penalty), grad, flat_path

    def _connection_gap_penalty(self, params):
        """Evaluate only the scalar connection gap penalty for candidate ranking."""
        comps = params_to_components(self._project_params(params), self.components)
        penalty, _, _ = self._connection_gap_penalty_and_grad(comps)
        return float(penalty)

    def _connection_repair_params(self, params):
        """Build a geometry-repair candidate that closes gaps along the current load path."""
        comps = params_to_components(self._project_params(params), self.components)
        n = len(comps)
        path = self._minimum_gap_path(comps)
        new_params = np.asarray([comp.get_params() for comp in comps], dtype=float)
        load = np.array([self.config.DL, float(getattr(self.config, "load_y", self.config.DW / 2)), float(getattr(self.config, "load_z", self.config.DH / 2))])
        fixed = np.array([0.0, load[1], load[2]])
        counts = np.ones(n, dtype=float)

        for a, b in path:
            gap = max(0.0, self._edge_gap(comps, a, b))
            if gap <= 1e-9:
                continue
            ca = fixed if a == -1 else load if a == n else comps[a].center()
            cb = fixed if b == -1 else load if b == n else comps[b].center()
            vec = cb - ca
            dist = float(np.linalg.norm(vec))
            if dist <= 1e-12:
                continue
            direction = vec / dist
            endpoint_edge = a in (-1, n) or b in (-1, n)
            effective_gap = max(gap, 4.0 if endpoint_edge else 0.75)
            shift = (0.65 if endpoint_edge else 0.40) * gap
            grow = (0.45 if endpoint_edge else 0.18) * effective_gap
            if 0 <= a < n:
                new_params[a, 0:3] += shift * direction / counts[a]
                new_params[a, 3:6] += grow * np.array([1.0, 0.35, 0.45])
                counts[a] += 1.0
            if 0 <= b < n:
                new_params[b, 0:3] -= shift * direction / counts[b]
                new_params[b, 3:6] += grow * np.array([1.0, 0.35, 0.45])
                counts[b] += 1.0
        return self._project_params(new_params.reshape(-1))

    def _connectivity_metrics(self, densities):
        """Measure whether the thresholded density connects the fixed side to the load side."""
        vals = np.asarray(densities, dtype=float).reshape(self.config.nelx, self.config.nely, self.config.nelz)
        threshold = float(
            getattr(
                self.config,
                "connectivity_density_threshold",
                getattr(self.config, "process_plot_density_threshold", 0.5),
            )
        )
        solid = vals >= threshold
        if not np.any(solid):
            return {"connected_to_load": False, "spanning_ratio": 0.0, "largest_component_ratio": 0.0}

        labels, count = label(solid, structure=generate_binary_structure(3, 1))
        if count == 0:
            return {"connected_to_load": False, "spanning_ratio": 0.0, "largest_component_ratio": 0.0}

        solid_count = float(np.sum(solid))
        component_sizes = np.bincount(labels.ravel(), minlength=count + 1).astype(float)
        largest_component_ratio = float(np.max(component_sizes[1:]) / solid_count)
        left_labels = set(int(v) for v in np.unique(labels[0, :, :]) if v > 0)
        j = int(np.clip(round((float(getattr(self.config, "load_y", self.config.DW / 2)) / self.config.DW) * (self.config.nely - 1)), 0, self.config.nely - 1))
        k = int(np.clip(round((float(getattr(self.config, "load_z", self.config.DH / 2)) / self.config.DH) * (self.config.nelz - 1)), 0, self.config.nelz - 1))
        j0, j1 = max(0, j - 1), min(self.config.nely, j + 2)
        k0, k1 = max(0, k - 1), min(self.config.nelz, k + 2)
        load_labels = set(int(v) for v in np.unique(labels[-1, j0:j1, k0:k1]) if v > 0)
        connected_labels = left_labels & load_labels
        if connected_labels:
            connected_mass = max(component_sizes[label_id] for label_id in connected_labels)
            return {
                "connected_to_load": True,
                "spanning_ratio": float(connected_mass / solid_count),
                "largest_component_ratio": largest_component_ratio,
            }
        endpoint_labels = left_labels | load_labels
        endpoint_mass = max((component_sizes[label_id] for label_id in endpoint_labels), default=0.0)
        return {
            "connected_to_load": False,
            "spanning_ratio": float(endpoint_mass / solid_count),
            "largest_component_ratio": largest_component_ratio,
        }

    def _tdf_p_norm(self):
        """Return the TDF p-norm for the configured component shape."""
        shape = self.config.component_shape.lower()
        if shape == "superellipsoid":
            return self.config.p_norm
        if shape == "box":
            return self.config.box_p_norm
        raise ValueError(f"Unsupported component_shape: {self.config.component_shape}")
