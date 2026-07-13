"""Method of Moving Asymptotes (MMA) solver for one volume constraint."""

import numpy as np


class MMASolver:
    """A compact MMA implementation for bound-constrained problems with one inequality.

    The subproblem uses the standard separable reciprocal approximation. The single
    constraint dual variable is solved by bisection, which is robust for the
    topology-optimization volume constraint used in this project.
    """

    def __init__(self, xmin, xmax, move_limit=0.15, asyinit=0.5, asyincr=1.2, asydecr=0.7, eps=1e-8):
        self.xmin = np.asarray(xmin, dtype=float)
        self.xmax = np.asarray(xmax, dtype=float)
        self.move_limit = float(move_limit)
        self.asyinit = float(asyinit)
        self.asyincr = float(asyincr)
        self.asydecr = float(asydecr)
        self.eps = float(eps)
        self.low = None
        self.upp = None
        self.xold1 = None
        self.xold2 = None
        self.iteration = 0

    def _update_asymptotes(self, x):
        """Update lower and upper moving asymptotes."""
        span = np.maximum(self.xmax - self.xmin, self.eps)
        if self.iteration < 2 or self.xold1 is None or self.xold2 is None:
            self.low = x - self.asyinit * span
            self.upp = x + self.asyinit * span
        else:
            same_direction = (x - self.xold1) * (self.xold1 - self.xold2) > 0.0
            factor = np.where(same_direction, self.asyincr, self.asydecr)
            self.low = x - factor * (self.xold1 - self.low)
            self.upp = x + factor * (self.upp - self.xold1)

        min_gap = 0.01 * span
        max_gap = 10.0 * span
        self.low = np.minimum(self.low, x - min_gap)
        self.low = np.maximum(self.low, x - max_gap)
        self.upp = np.maximum(self.upp, x + min_gap)
        self.upp = np.minimum(self.upp, x + max_gap)

    def _subproblem_solution(self, lam, p0, q0, p1, q1, alpha, beta):
        """Closed-form primal minimizer for a fixed single dual variable."""
        p = np.maximum(p0 + lam * p1, self.eps)
        q = np.maximum(q0 + lam * q1, self.eps)
        sp = np.sqrt(p)
        sq = np.sqrt(q)
        x = (sp * alpha + sq * beta) / (sp + sq)
        return np.clip(x, alpha, beta)

    def _constraint_approx(self, xnew, x, g, p1, q1):
        """Evaluate MMA reciprocal approximation of the volume constraint."""
        return (
            g
            + np.sum(p1 / np.maximum(self.upp - xnew, self.eps) + q1 / np.maximum(xnew - self.low, self.eps))
            - np.sum(p1 / np.maximum(self.upp - x, self.eps) + q1 / np.maximum(x - self.low, self.eps))
        )

    def update(self, x, f, df, g, dg):
        """Return the next MMA iterate.

        Parameters follow the convention g <= 0 for feasibility.
        """
        del f
        x = np.asarray(x, dtype=float)
        df = np.asarray(df, dtype=float)
        dg = np.asarray(dg, dtype=float)
        g = float(g)
        self._update_asymptotes(x)

        span = np.maximum(self.xmax - self.xmin, self.eps)
        alpha = np.maximum.reduce([self.xmin, self.low + 0.001 * span, x - self.move_limit * span])
        beta = np.minimum.reduce([self.xmax, self.upp - 0.001 * span, x + self.move_limit * span])

        ux = np.maximum(self.upp - x, self.eps)
        xl = np.maximum(x - self.low, self.eps)
        regularization = 1e-5 * np.maximum(np.abs(df), 1.0)
        p0 = np.maximum(df, 0.0) * ux**2 + regularization
        q0 = np.maximum(-df, 0.0) * xl**2 + regularization
        p1 = np.maximum(dg, 0.0) * ux**2 + self.eps
        q1 = np.maximum(-dg, 0.0) * xl**2 + self.eps

        xnew = self._subproblem_solution(0.0, p0, q0, p1, q1, alpha, beta)
        if self._constraint_approx(xnew, x, g, p1, q1) > 0.0:
            lo, hi = 0.0, 1.0
            for _ in range(80):
                xhi = self._subproblem_solution(hi, p0, q0, p1, q1, alpha, beta)
                if self._constraint_approx(xhi, x, g, p1, q1) <= 0.0:
                    break
                hi *= 2.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                xm = self._subproblem_solution(mid, p0, q0, p1, q1, alpha, beta)
                if self._constraint_approx(xm, x, g, p1, q1) > 0.0:
                    lo = mid
                else:
                    hi = mid
            xnew = self._subproblem_solution(hi, p0, q0, p1, q1, alpha, beta)

        self.xold2 = None if self.xold1 is None else self.xold1.copy()
        self.xold1 = x.copy()
        self.iteration += 1
        return np.clip(xnew, self.xmin, self.xmax)
