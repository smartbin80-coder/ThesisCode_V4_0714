"""3D Moving Morphable Component geometry definitions."""

import numpy as np


def _domain_size(config):
    return np.array([config.DL, config.DW, config.DH], dtype=float)


def _minimum_length(config):
    return 0.5 * min(config.DL / config.nelx, config.DW / config.nely, config.DH / config.nelz)


class MMCComponent3D:
    """A movable, rotatable, scalable 3D MMC component."""

    def __init__(self, x0, y0, z0, L1, L2, L3, alpha=0.0, beta=0.0, gamma=0.0, active=True):
        self.x0, self.y0, self.z0 = float(x0), float(y0), float(z0)
        self.L1, self.L2, self.L3 = max(float(L1), 1e-6), max(float(L2), 1e-6), max(float(L3), 1e-6)
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self.active = bool(active)

    def get_params(self):
        """Return the 9 design variables."""
        return np.array([self.x0, self.y0, self.z0, self.L1, self.L2, self.L3, self.alpha, self.beta, self.gamma], dtype=float)

    def set_params(self, params):
        """Update the component from 9 design variables."""
        p = np.asarray(params, dtype=float)
        self.x0, self.y0, self.z0 = p[:3]
        self.L1, self.L2, self.L3 = np.maximum(p[3:6], 1e-6)
        self.alpha, self.beta, self.gamma = p[6:9]

    def rotation_matrix(self):
        """Build the rotation matrix from Euler angles alpha, beta, gamma."""
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        cb, sb = np.cos(self.beta), np.sin(self.beta)
        cg, sg = np.cos(self.gamma), np.sin(self.gamma)
        rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
        ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])
        return rz @ ry @ rx

    def global_to_local(self, points):
        """Transform global points to the component-local coordinate system."""
        pts = np.asarray(points, dtype=float) - self.center()
        return pts @ self.rotation_matrix()

    def tdf(self, points, p=6):
        """Compute the superellipsoid TDF; positive values are inside the component."""
        q = self.global_to_local(points)
        lengths = np.maximum(np.array([self.L1, self.L2, self.L3]), 1e-6)
        val = np.sum(np.abs(q / lengths) ** p, axis=1) ** (1.0 / p)
        return 1.0 - val if self.active else np.full(points.shape[0], -1e3)

    def volume(self):
        """Return a bounding-box based approximate component volume feature."""
        return 8.0 * self.L1 * self.L2 * self.L3

    def center(self):
        """Return the component center."""
        return np.array([self.x0, self.y0, self.z0], dtype=float)

    def direction_vectors(self):
        """Return the three local principal directions in global coordinates."""
        return self.rotation_matrix().T

    def as_node_feature(self):
        """Convert the component to raw GNN node features."""
        return np.r_[self.get_params(), self.volume(), float(self.active)]


def component_half_extent(component):
    """Return the rotated axis-aligned half extent of one MMC component."""
    lengths = np.maximum(np.array([component.L1, component.L2, component.L3], dtype=float), 1e-9)
    return np.abs(component.rotation_matrix()) @ lengths


def component_domain_margins(component, config):
    """Return [xmin, ymin, zmin, xmax, ymax, zmax] margins to domain faces."""
    center = component.center()
    half = component_half_extent(component)
    domain = _domain_size(config)
    return np.r_[center - half, domain - (center + half)]


def components_are_in_domain(components, config, tol=1e-9):
    """Check whether all rotated component AABBs are inside the design domain."""
    return all(np.min(component_domain_margins(comp, config)) >= -float(tol) for comp in components)


def project_component_to_domain(component, config, min_l=None):
    """Project one component so its rotated AABB lies inside the design domain."""
    domain = _domain_size(config)
    min_l = _minimum_length(config) if min_l is None else float(min_l)
    params = component.get_params().copy()
    lengths = np.maximum(params[3:6], min_l)
    eps = 1e-9

    for _ in range(8):
        trial = MMCComponent3D(params[0], params[1], params[2], *lengths, *params[6:9], active=component.active)
        half = component_half_extent(trial)
        limit = np.maximum(0.5 * domain - eps, min_l)
        if np.all(half <= limit + 1e-10):
            break
        scale = float(np.min(limit / np.maximum(half, eps)))
        lengths = np.maximum(lengths * min(scale, 1.0), min_l)

    trial = MMCComponent3D(params[0], params[1], params[2], *lengths, *params[6:9], active=component.active)
    half = component_half_extent(trial)
    lower = half
    upper = domain - half
    center = np.asarray(params[0:3], dtype=float)
    if np.any(lower > upper):
        center = 0.5 * domain
    else:
        center = np.clip(center, lower, upper)
    return MMCComponent3D(center[0], center[1], center[2], *lengths, *params[6:9], active=component.active)


def project_components_to_domain(components, config):
    """Project all components to the strict rotated-AABB design-domain constraint."""
    return [project_component_to_domain(comp, config) for comp in components]


def _segment_component(start, end, radius_y, radius_z):
    """Create one component whose local x-axis follows a segment in the x-z plane."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    center = 0.5 * (start + end)
    vector = end - start
    length = float(np.linalg.norm(vector))
    beta = float(np.arctan2(vector[2], vector[0]))
    return MMCComponent3D(center[0], center[1], center[2], 0.5 * length, radius_y, radius_z, 0.0, beta, 0.0)


def _cantilever_reference_components(config):
    """Build an overlapped cantilever reference layout for controlled comparisons."""
    y_mid = 0.5 * config.DW
    z_low = 0.18 * config.DH
    z_high = 0.82 * config.DH
    chord_radius_y = 0.34 * config.DW
    chord_radius_z = 0.08 * config.DH
    web_radius_y = 0.28 * config.DW
    web_radius_z = 0.07 * config.DH
    xs = np.linspace(0.0, config.DL, 7)
    comps = []

    for i in range(6):
        x0 = max(0.0, xs[i] - 0.8)
        x1 = min(config.DL, xs[i + 1] + 0.8)
        comps.append(_segment_component((x0, y_mid, z_low), (x1, y_mid, z_low), chord_radius_y, chord_radius_z))
        comps.append(_segment_component((x0, y_mid, z_high), (x1, y_mid, z_high), chord_radius_y, chord_radius_z))

    for i in range(5):
        x0 = max(0.0, xs[i] + 0.6)
        x1 = min(config.DL, xs[i + 1] - 0.6)
        comps.append(_segment_component((x0, y_mid, z_low), (x1, y_mid, z_high), web_radius_y, web_radius_z))
        comps.append(_segment_component((x0, y_mid, z_high), (x1, y_mid, z_low), web_radius_y, web_radius_z))

    for x in (0.0, config.DL):
        comps.append(_segment_component((x, y_mid, z_low), (x, y_mid, z_high), web_radius_y, web_radius_z))

    return comps[:24]


def _jitter_components(components, config, rng):
    """Apply small random perturbations while preserving the overlapped skeleton."""
    jittered = []
    for comp in components:
        params = comp.get_params()
        params[0] += rng.normal(0.0, 0.012 * config.DL)
        params[1] += rng.normal(0.0, 0.025 * config.DW)
        params[2] += rng.normal(0.0, 0.018 * config.DH)
        params[3:6] *= rng.uniform(0.92, 1.08, size=3)
        params[6] += rng.normal(0.0, 0.025)
        params[7] += rng.normal(0.0, 0.035)
        params[8] += rng.normal(0.0, 0.018)
        c = MMCComponent3D(*params, active=comp.active)
        jittered.append(project_component_to_domain(c, config))
    return jittered


def create_initial_components(config):
    """Generate the default deterministic scattered MMC component set."""
    rng = np.random.default_rng(getattr(config, "seed", 7))
    return create_random_components(config, rng)


def create_random_components(config, rng):
    """Generate randomized, scattered, strictly domain-feasible MMC components."""
    comps = []
    min_l = 0.7 * min(config.DL / config.nelx, config.DW / config.nely, config.DH / config.nelz)
    domain = _domain_size(config)
    for _ in range(config.num_components):
        for _attempt in range(160):
            L1 = rng.uniform(max(min_l, 0.05 * config.DL), 0.16 * config.DL)
            L2 = rng.uniform(max(min_l, 0.06 * config.DW), 0.18 * config.DW)
            L3 = rng.uniform(max(min_l, 0.04 * config.DH), 0.16 * config.DH)
            alpha = rng.uniform(-0.18 * np.pi, 0.18 * np.pi)
            beta = rng.uniform(-0.10 * np.pi, 0.10 * np.pi)
            gamma = rng.uniform(-0.06 * np.pi, 0.06 * np.pi)
            trial = MMCComponent3D(0.0, 0.0, 0.0, L1, L2, L3, alpha, beta, gamma)
            half = component_half_extent(trial)
            if np.all(half < 0.5 * domain):
                low = half
                high = domain - half
                center = rng.uniform(low, high)
                comps.append(MMCComponent3D(center[0], center[1], center[2], L1, L2, L3, alpha, beta, gamma))
                break
        else:
            L1 = rng.uniform(max(min_l, 0.04 * config.DL), 0.08 * config.DL)
            L2 = rng.uniform(max(min_l, 0.05 * config.DW), 0.12 * config.DW)
            L3 = rng.uniform(max(min_l, 0.04 * config.DH), 0.10 * config.DH)
            center = rng.uniform(0.35 * domain, 0.65 * domain)
            trial = MMCComponent3D(center[0], center[1], center[2], L1, L2, L3)
            comps.append(project_component_to_domain(trial, config, min_l=min_l))
    return comps


def params_to_components(params, template_components):
    """Convert flattened parameters to component objects."""
    params = np.asarray(params, dtype=float).reshape((-1, 9))
    comps = []
    for p, old in zip(params, template_components):
        c = MMCComponent3D(*old.get_params(), active=old.active)
        c.set_params(p)
        comps.append(c)
    return comps


def components_to_params(components):
    """Convert component objects to a flattened parameter array."""
    return np.concatenate([c.get_params() for c in components])


def project_params_to_domain(params, template_components, config):
    """Project flattened MMC parameters to strict domain-feasible components."""
    comps = params_to_components(params, template_components)
    projected = project_components_to_domain(comps, config)
    return components_to_params(projected)


def get_bounds(components, config):
    """Return simple box bounds for the MMC design variables."""
    min_l = _minimum_length(config)
    max_l = 0.5 * max(config.DL, config.DW, config.DH)
    bounds = []
    for _ in components:
        bounds += [
            (0, config.DL),
            (0, config.DW),
            (0, config.DH),
            (min_l, max_l),
            (min_l, max_l),
            (min_l, max_l),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
        ]
    return bounds
