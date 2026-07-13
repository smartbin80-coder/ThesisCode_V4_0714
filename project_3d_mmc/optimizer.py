"""3D-MMC optimizer with analytic sensitivities, MMA, and GNN step labels."""

from pathlib import Path

import numpy as np
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
from mmc3d_components import components_to_params, get_bounds, params_to_components
from tdf import compute_density_with_sensitivities, compute_element_density_from_tdf, compute_global_tdf, compute_global_tdf_with_sensitivities
from utils import ensure_dir, save_json


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
        return compliance, volume_fraction, densities, U, graph_data, comps

    def evaluate_with_sensitivities(self, params):
        """Evaluate response and analytic gradients with respect to MMC design variables."""
        params = np.asarray(params, dtype=float)
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

        out = {
            "compliance": float(compliance),
            "volume_fraction": float(volume_fraction),
            "densities": densities,
            "U": U,
            "components": comps,
            "element_strain_energy": element_strain_energy,
            "component_strain_energy_norm": self._component_strain_energy_norm(densities, drho_dparams, element_strain_energy),
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
            "volume_grad": np.asarray(volume_grad, dtype=float),
        }
        self._cache_params = params.copy()
        self._cache_eval = out
        return out

    def objective(self, params):
        """SLSQP objective."""
        return self.evaluate_with_sensitivities(params)["compliance"]

    def objective_jac(self, params):
        """Analytic objective gradient for SLSQP."""
        return self.evaluate_with_sensitivities(params)["compliance_grad"]

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
            p_eta = apply_gnn_step_scale(params_old, params_new, eta)
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
        params = np.asarray(params, dtype=float)
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
            f"volume={volume_fraction:.4f} | constraint={self.config.volfrac - volume_fraction:.4f} | eta_label={eta_label:.2f}"
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
                "trajectory_id": np.asarray(str(getattr(self.config, "trajectory_id", ""))),
                "seed": int(getattr(self.config, "seed", -1)),
                "num_components": int(getattr(self.config, "num_components", len(eval_data["components"]))),
                "load_y": float(getattr(self.config, "load_y", self.config.DW / 2)),
                "load_z": float(getattr(self.config, "load_z", self.config.DH / 2)),
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

        self.prev_params = params.copy()
        self.iteration += 1

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
        x = np.asarray(x0, dtype=float).copy()
        success = True
        message = "MMA converged by max iteration."
        for _ in range(self.config.max_iter):
            data = self.evaluate_with_sensitivities(x)
            g = data["volume_fraction"] - self.config.volfrac
            raw = solver.update(x, data["compliance"], data["compliance_grad"], g, data["volume_grad"])
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
        old_violation = max(0.0, old_v - self.config.volfrac)
        candidates = []
        eta_list = tuple(float(eta) for eta in self.config.eta_candidates) + (0.0,)
        for eta in sorted(set(eta_list), reverse=True):
            trial = apply_gnn_step_scale(x, raw, eta)
            c_trial, v_trial, *_ = self.evaluate(trial)
            violation = max(0.0, v_trial - self.config.volfrac)
            candidates.append((eta, trial, c_trial, v_trial, violation))

        feasible = [item for item in candidates if item[4] <= 1e-8]
        if old_violation <= 1e-8 and feasible:
            eta, trial, *_ = min(feasible, key=lambda item: item[2])
            return trial

        improving_violation = [item for item in candidates if item[4] <= old_violation + 1e-10]
        if improving_violation:
            eta, trial, *_ = min(improving_violation, key=lambda item: (item[4], item[2]))
            return trial

        return x.copy()

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
        x0 = components_to_params(self.components)
        self.prev_params = None
        self.callback(x0)
        if self.config.optimizer_type.upper() == "MMA":
            result = self._run_mma(x0)
        elif self.config.optimizer_type.upper() == "SLSQP":
            result = self._run_slsqp(x0)
        else:
            raise ValueError(f"Unsupported optimizer_type: {self.config.optimizer_type}")
        final_params = result.x
        final_eval = self._save_final(result, final_params)
        return result, final_eval

    def _tdf_p_norm(self):
        """Return the TDF p-norm for the configured component shape."""
        shape = self.config.component_shape.lower()
        if shape == "superellipsoid":
            return self.config.p_norm
        if shape == "box":
            return self.config.box_p_norm
        raise ValueError(f"Unsupported component_shape: {self.config.component_shape}")
