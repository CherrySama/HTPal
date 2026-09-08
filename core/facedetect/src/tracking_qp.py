"""Small dense bound-constrained QP solver used by real-time tracking.

The tracking problem has six decision variables and only component-wise
velocity/acceleration/position bounds. A tiny dense active-set solver keeps
the real-time path deterministic without adding a runtime QP dependency.
"""

from __future__ import annotations

import numpy as np


def solve_bounded_qp(
    hessian: np.ndarray,
    gradient: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    tolerance: float = 1e-8,
    max_iterations: int = 50,
) -> tuple[np.ndarray, str, int, np.ndarray]:
    """Solve ``min .5*x'Hx + g'x`` subject to component-wise bounds.

    The Hessian is regularised and symmetrised. The returned active-set vector
    uses ``-1`` for lower, ``0`` for free and ``1`` for upper bounds.
    """
    h = np.asarray(hessian, dtype=np.float64)
    g = np.asarray(gradient, dtype=np.float64).reshape(-1)
    lo = np.asarray(lower, dtype=np.float64).reshape(-1)
    hi = np.asarray(upper, dtype=np.float64).reshape(-1)
    n = g.size
    if h.shape != (n, n) or lo.shape != (n,) or hi.shape != (n,):
        raise ValueError("QP 矩阵或边界维度错误")
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(g))
            and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        raise ValueError("QP 输入包含非有限值")
    if np.any(lo > hi):
        return np.clip(np.zeros(n), lo, hi), "BOUND_INVALID", 0, np.zeros(n, dtype=int)

    h = 0.5 * (h + h.T) + 1e-9 * np.eye(n)
    # Warm-start with the clipped unconstrained minimiser, then perform a
    # standard bounded active-set iteration.  With six variables this normally
    # converges in only a few small linear solves.
    try:
        x = np.clip(np.linalg.solve(h, -g), lo, hi)
    except np.linalg.LinAlgError:
        return np.clip(np.zeros(n), lo, hi), "NUMERICAL_FAILURE", 0, np.zeros(n, dtype=int)
    state = np.zeros(n, dtype=int)
    state[x <= lo + tolerance] = -1
    state[x >= hi - tolerance] = 1
    for iteration in range(1, max_iterations + 1):
        fixed = state != 0
        free = ~fixed
        x[state == -1] = lo[state == -1]
        x[state == 1] = hi[state == 1]
        if np.any(free):
            try:
                x[free] = np.linalg.solve(
                    h[np.ix_(free, free)],
                    -g[free] - h[np.ix_(free, fixed)] @ x[fixed],
                )
            except np.linalg.LinAlgError:
                return np.clip(np.zeros(n), lo, hi), "NUMERICAL_FAILURE", iteration, state
        below = free & (x < lo - tolerance)
        above = free & (x > hi + tolerance)
        if np.any(below) or np.any(above):
            state[below] = -1
            state[above] = 1
            continue
        x = np.clip(x, lo, hi)
        grad = h @ x + g
        release_lower = (state == -1) & (grad < -tolerance)
        release_upper = (state == 1) & (grad > tolerance)
        if np.any(release_lower) or np.any(release_upper):
            state[release_lower] = 0
            state[release_upper] = 0
            continue
        return x, "SOLVED", iteration, state
    return np.clip(x, lo, hi), "NUMERICAL_FAILURE", max_iterations, state


def bounded_qp_self_test() -> None:
    """Quick dependency-free checks used by CI and manual offline validation."""
    h = np.eye(2)
    x, status, _, active = solve_bounded_qp(h, np.array([-2.0, 0.25]), np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert status == "SOLVED" and np.allclose(x, [1.0, -0.25]) and active[0] == 1
    x, status, _, active = solve_bounded_qp(h, np.array([2.0, -0.25]), np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    assert status == "SOLVED" and np.allclose(x, [-1.0, 0.25]) and active[0] == -1
    x, status, _, _ = solve_bounded_qp(h, np.zeros(2), np.array([0.2, 0.3]), np.array([0.1, 0.4]))
    assert status == "BOUND_INVALID" and np.allclose(x, [0.1, 0.3])


def build_least_squares_qp(rows: list[np.ndarray], targets: list[np.ndarray]):
    """Convert stacked weighted least-squares rows into H/g form."""
    if not rows:
        raise ValueError("QP 至少需要一个目标项")
    a = np.vstack([np.asarray(row, dtype=np.float64) for row in rows])
    b = np.concatenate([np.asarray(target, dtype=np.float64).reshape(-1) for target in targets])
    if a.shape[0] != b.size:
        raise ValueError("QP 目标维度错误")
    return a.T @ a, -(a.T @ b)

