"""Physics-gap identification and calibrated-twin construction (claude_try2).

Data     : logs from fr3_twin.Plant.simulate on the 'real' system, with sensor noise.
Library  : fr3_twin.build_library (70 inertial + 35 friction columns), rows = 7 torque equations per sample.
Methods  : nominal | LS-payload | LS-struct (payload + Coulomb/viscous, dense) | LS-full (ridge, dense)
           | SINDy (STLSQ with leave-one-trajectory-out CV) | MLP residual | PINN-SR hybrid (pinnsr.py)
Twin     : identified physical coefficients are written back into a MuJoCo model (calibrated MJCF);
           terms that cannot be represented natively are applied as generalized forces.
Metrics  : torque NRMSE on held-out trajectories, closed-loop replay TCP RMSE, open-loop window divergence.
"""
import copy
import numpy as np
import mujoco
from scipy.signal import savgol_filter
import fr3_twin as ft

NAMES = ft.column_names()
N_INERT = 70
N_COLS = 112
IDX_PAYLOAD = list(range(60, 70))                                 # link7 = flange body (link 7 + payload)
IDX_COULVISC = [70 + 5 * j + b for j in range(7) for b in (0, 1)]
N_FRIC = 35
IDX_ARM = list(range(105, 112))
GROUPS = {f"link{i + 1}": list(range(10 * i, 10 * (i + 1))) for i in range(7)}
GROUPS.update({f"joint{j + 1}": list(range(70 + 5 * j, 70 + 5 * (j + 1))) for j in range(7)})
GROUPS["armature"] = IDX_ARM                                       # rotor inertia of all joints: one engine-level group

IDENTIFY_VERSION = "2026-09-25b"   # empty-model candidate, ddof=1 CV standard errors, resistive non-native friction, dense LMI baseline
FORWARD_ALLOW_EMPTY = True         # the uncorrected twin is a candidate of forward selection and of the 1-SE rule
RIDGE_GROUP = 1e-4                 # ridge of the bounded group fits (standardised units)


# ----------------------------------------------------------------------------- data
def collect(model_real, model_nom, seeds, T=20.0, noise_scale=1.0, noise_seed=0, log_every=1):
    """Run the tracking controller on the real plant for each Fourier seed.  Returns list of logs with
    measured (noisy) q, tau and the true derivatives kept for oracle experiments."""
    ctrl = ft.PD(model_nom)
    rng = np.random.default_rng(noise_seed)
    logs = []
    for sd in seeds:
        pl = ft.Plant(copy.copy(model_real))
        log = pl.simulate(ft.Fourier(sd), ctrl, T, log_every=log_every)
        log["q_meas"], log["tau_meas"] = ft.add_sensor_noise(log, rng, scale=noise_scale)
        log["seed"] = sd
        logs.append(log)
    return logs


def sg_derivatives(q_meas, dt, window=101, poly=3):
    """Savitzky-Golay smoothing and derivatives along axis 0."""
    q = savgol_filter(q_meas, window, poly, deriv=0, delta=dt, axis=0)
    qd = savgol_filter(q_meas, window, poly, deriv=1, delta=dt, axis=0)
    qdd = savgol_filter(q_meas, window, poly, deriv=2, delta=dt, axis=0)
    return q, qd, qdd


def make_dataset(model_nom, logs, source="sg", dec=5, window=101, trim=0.2):
    """source: 'oracle' (true q, qd, qdd from the simulator) or 'sg' (from noisy q_meas).
    Returns a list of per-trajectory dicts with q, qd, qdd, tau (measured), tau_nom, A (n,7,105), r (n,7)."""
    data = []
    for log in logs:
        dt = log["t"][1] - log["t"][0]
        if source == "oracle":
            q, qd, qdd = log["q"], log["qd"], log["qdd"]
        else:
            q, qd, qdd = sg_derivatives(log["q_meas"], dt, window=window)
        n_trim = int(round(trim / dt))
        sl = slice(n_trim, q.shape[0] - n_trim, dec)
        q, qd, qdd, tau = q[sl], qd[sl], qdd[sl], log["tau_meas"][sl]
        tau_nom, A = ft.build_library(model_nom, q, qd, qdd)
        data.append(dict(q=q, qd=qd, qdd=qdd, tau=tau, tau_nom=tau_nom, A=A, r=tau - tau_nom, seed=log["seed"],
                         t=log["t"][sl]))
    return data


def stack(data, cols=None, vmin=0.1):
    """Rows = joint equations.  Rows of joint j with |qd_j| < vmin are dropped (the simulator's friction knee, |qd| < f/(M_eff b) ~ 0.05 rad/s, is not in the library)."""
    A = np.concatenate([d["A"] for d in data], axis=0)
    r = np.concatenate([d["r"] for d in data], axis=0)
    qd = np.concatenate([d["qd"] for d in data], axis=0)
    keep = (np.abs(qd) >= vmin).ravel()
    A2 = A.reshape(-1, A.shape[2])[keep]
    if cols is not None:
        A2 = A2[:, cols]
    return A2, r.ravel()[keep]


# ----------------------------------------------------------------------------- regression
def ols(A, y, ridge=0.0):
    if ridge > 0:
        return np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y)
    return np.linalg.lstsq(A, y, rcond=None)[0]


def stlsq(A, y, thr, iters=20, ridge=1e-8):
    """Sequentially thresholded least squares on column-standardised A.  thr in standardised units."""
    scale = A.std(axis=0)
    scale[scale < 1e-12] = np.inf                      # dead columns
    As = A / scale
    xi = ols(As, y, ridge)
    big = np.abs(xi) >= thr
    for _ in range(iters):
        xi_new = np.zeros_like(xi)
        if big.any():
            xi_new[big] = ols(As[:, big], y, ridge)
        big_new = np.abs(xi_new) >= thr
        xi = xi_new
        if (big_new == big).all():
            break
        big = big_new
    xi[~big] = 0.0
    return xi / scale


def physical_bounds(model_nom, mass_frac=0.5):
    """Lower/upper bounds keeping the calibrated twin physical: total Coulomb, viscous, Stribeck and quadratic
    friction non-negative; body masses at least (1 - mass_frac) of nominal.  Other coefficients free."""
    lb = np.full(len(NAMES), -np.inf); ub = np.full(len(NAMES), np.inf)
    for i, bn in enumerate(ft.LINKS):
        lb[10 * i] = -mass_frac * ft.body_pi(model_nom, bn)[0]
    for j in range(7):
        dof = model_nom.jnt_dofadr[mujoco.mj_name2id(model_nom, mujoco.mjtObj.mjOBJ_JOINT, ft.JOINTS[j])]
        lb[70 + 5 * j + 0] = -model_nom.dof_frictionloss[dof]
        lb[70 + 5 * j + 1] = -model_nom.dof_damping[dof]
        lb[70 + 5 * j + 2] = 0.0
        lb[70 + 5 * j + 3] = 0.0
        lb[105 + j] = -model_nom.dof_armature[dof]                  # total armature stays non-negative
    return lb, ub


def stlsq_bounded(A, y, thr, lb, ub, iters=20, ridge=1e-8):
    """STLSQ with box constraints (physically feasible sparse regression).  Standardised columns."""
    from scipy.optimize import lsq_linear
    scale = A.std(axis=0)
    dead = scale < 1e-12
    scale[dead] = np.inf
    As = A / scale
    lbs, ubs = lb * scale, ub * scale                  # bounds in standardised units
    lbs[dead], ubs[dead] = -1e-9, 1e-9

    def solve(mask):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return np.zeros(A.shape[1])
        res = lsq_linear(As[:, idx], y, bounds=(lbs[idx], ubs[idx]), method="bvls", lsmr_tol="auto", max_iter=200)
        out = np.zeros(A.shape[1]); out[idx] = res.x
        return out

    big = ~dead
    xi = solve(big)
    big = np.abs(xi) >= thr
    for _ in range(iters):
        xi = solve(big)
        big_new = np.abs(xi) >= thr
        if (big_new == big).all():
            break
        big = big_new
    xi[~big] = 0.0
    return xi / scale


GROUP_INDEX = [GROUPS[f"link{i + 1}"] for i in range(7)] + [GROUPS[f"joint{j + 1}"] for j in range(7)]


def stlsq_group(A, y, thr, lb=None, ub=None, groups=GROUP_INDEX, iters=20, ridge=1e-8, thr_elem=None):
    """Group-sparse STLSQ: whole bodies / joints are kept or dropped by the RMS of their standardised coefficients,
    then (optionally) elementwise thresholding inside the kept groups.  Box constraints optional (BVLS)."""
    from scipy.optimize import lsq_linear
    scale = A.std(axis=0)
    dead = scale < 1e-12
    scale[dead] = np.inf
    As = A / scale
    bounded = lb is not None
    if bounded:
        lbs, ubs = lb * scale, ub * scale
        lbs[dead], ubs[dead] = -1e-9, 1e-9

    def solve(mask):
        idx = np.flatnonzero(mask)
        out = np.zeros(A.shape[1])
        if idx.size == 0:
            return out
        if bounded:
            out[idx] = lsq_linear(As[:, idx], y, bounds=(lbs[idx], ubs[idx]), method="bvls", lsmr_tol="auto", max_iter=200).x
        else:
            out[idx] = ols(As[:, idx], y, ridge)
        return out

    mask = ~dead
    for _ in range(iters):
        xi = solve(mask)
        new = np.zeros_like(mask)
        for g in groups:
            g = [c for c in g if not dead[c]]
            if g and np.sqrt(np.mean(xi[g] ** 2)) >= thr:
                new[g] = True
        if thr_elem is not None:
            new &= np.abs(xi) >= thr_elem
        if (new == mask).all():
            break
        mask = new
    xi = solve(mask)
    xi[~mask] = 0.0
    return xi / scale


def best_subset_groups(data, model_nom, kmax=4, criterion="cv", bounded=True, groups=None, verbose=None, allow_empty=None):
    """Physical best-subset: search all subsets of at most kmax groups (bodies / joints), and the empty subset when
    allow_empty, fit bounded LS with the group ridge on the subset's columns, score by leave-one-trajectory-out CV torque
    RMSE (or BIC on the training fit), and pick the smallest subset within one CV standard error of the best."""
    from itertools import combinations
    from scipy.optimize import lsq_linear
    allow_empty = FORWARD_ALLOW_EMPTY if allow_empty is None else allow_empty
    names = list(GROUPS.keys()) if groups is None else groups
    lb, ub = physical_bounds(model_nom)
    folds = [(stack([d for k, d in enumerate(data) if k != i]), stack([data[i]])) for i in range(len(data))]
    A_all, y_all = stack(data)

    def fit_cols(A, y, cols, ridge=RIDGE_GROUP):
        """Bounded LS with a light ridge (in standardised units)."""
        if not cols:
            return np.zeros(0)
        if not bounded:
            return ols(A[:, cols], y)
        scale = A[:, cols].std(axis=0); scale[scale < 1e-12] = np.inf
        As = A[:, cols] / scale
        Ar = np.vstack([As, np.sqrt(ridge * A.shape[0]) * np.eye(len(cols))]); yr = np.concatenate([y, np.zeros(len(cols))])
        lbs, ubs = np.nan_to_num(lb[cols] * scale, nan=-np.inf), np.nan_to_num(ub[cols] * scale, nan=np.inf)
        res = lsq_linear(Ar, yr, bounds=(lbs, ubs), method="bvls", lsmr_tol="auto", max_iter=200)
        return res.x / scale

    results = []
    for k in range(0 if allow_empty else 1, kmax + 1):
        for sub in combinations(names, k):
            cols = sorted(sum((GROUPS[g] for g in sub), []))
            if criterion == "cv":
                errs = []
                for (A_tr, y_tr), (A_va, y_va) in folds:
                    pred = A_va[:, cols] @ fit_cols(A_tr, y_tr, cols) if cols else 0.0
                    errs.append(np.sqrt(np.mean((y_va - pred) ** 2)))
                score, se = float(np.mean(errs)), float(np.std(errs, ddof=1) / np.sqrt(len(errs)))
            else:
                xi = fit_cols(A_all, y_all, cols)
                rss = np.sum((y_all - (A_all[:, cols] @ xi if cols else 0.0)) ** 2); n = y_all.size
                score, se = float(n * np.log(rss / n) + len(cols) * np.log(n)), 0.0
            results.append((score, se, sub, cols))
            if verbose:
                verbose(f"    subset {sub}: {criterion} {score:.4f}")
    results.sort(key=lambda r: r[0])
    best = results[0]
    if criterion == "cv":                                  # smallest subset within one SE of the best
        cand = [r for r in results if r[0] <= best[0] + best[1]]
        cand.sort(key=lambda r: (len(r[2]), r[0]))
        best = cand[0]
    xi = full_coef(fit_cols(A_all, y_all, best[3]), best[3])
    top = [(round(r[0], 4), r[2]) for r in results[:8]]
    return xi, dict(subset=list(best[2]), score=best[0], se=best[1], top=top, n_subsets=len(results))


def forward_groups(data, model_nom, kmax=8, groups=None, verbose=None, allow_empty=None):
    """Greedy forward selection over groups with leave-one-trajectory-out CV torque RMSE.  The path starts at the
    uncorrected (empty) model when allow_empty; at each step the group that lowers the CV error most is added; the search
    stops when no addition improves the score, and the final model is the smallest one on the path within one standard
    error of the best.  Bounded LS with the group ridge as in best_subset_groups."""
    from scipy.optimize import lsq_linear
    allow_empty = FORWARD_ALLOW_EMPTY if allow_empty is None else allow_empty
    names = list(GROUPS.keys()) if groups is None else groups
    lb, ub = physical_bounds(model_nom)
    folds = [(stack([d for k, d in enumerate(data) if k != i]), stack([data[i]])) for i in range(len(data))]
    A_all, y_all = stack(data)

    def fit_cols(A, y, cols, ridge=RIDGE_GROUP):
        if not cols:
            return np.zeros(0)
        scale = A[:, cols].std(axis=0); scale[scale < 1e-12] = np.inf
        As = A[:, cols] / scale
        Ar = np.vstack([As, np.sqrt(ridge * A.shape[0]) * np.eye(len(cols))]); yr = np.concatenate([y, np.zeros(len(cols))])
        lbs, ubs = np.nan_to_num(lb[cols] * scale, nan=-np.inf), np.nan_to_num(ub[cols] * scale, nan=np.inf)
        return lsq_linear(Ar, yr, bounds=(lbs, ubs), method="bvls", lsmr_tol="auto", max_iter=200).x / scale

    def score(sub):
        cols = sorted(sum((GROUPS[g] for g in sub), []))
        errs = []
        for (A_tr, y_tr), (A_va, y_va) in folds:
            pred = A_va[:, cols] @ fit_cols(A_tr, y_tr, cols) if cols else 0.0
            errs.append(np.sqrt(np.mean((y_va - pred) ** 2)))
        return float(np.mean(errs)), float(np.std(errs, ddof=1) / np.sqrt(len(errs))), cols

    chosen, path = [], []
    if allow_empty:
        s0, se0, _ = score([])
        path.append((s0, se0, [], []))
    while len(chosen) < kmax:
        best = None
        for g in names:
            if g in chosen:
                continue
            s, se, cols = score(chosen + [g])
            if best is None or s < best[0]:
                best = (s, se, g, cols)
        if path and best[0] >= path[-1][0]:
            break
        chosen.append(best[2]); path.append((best[0], best[1], list(chosen), best[3]))
        if verbose:
            verbose(f"    forward + {best[2]}: cv {best[0]:.4f}")
    smin = min(p[0] for p in path); se = [p for p in path if p[0] == smin][0][1]
    final = [p for p in path if p[0] <= smin + se][0]
    xi = full_coef(fit_cols(A_all, y_all, final[3]), final[3]) if final[3] else np.zeros(len(NAMES))
    return xi, dict(subset=final[2], score=final[0], se=final[1], path=[(round(p[0], 4), p[2]) for p in path])


def dense_fit(data, model_nom, ridge=RIDGE_GROUP, lmi=False, eps=1e-6):
    """Dense correction over all columns with SAPC's fit and bounds: bounded least squares with the same ridge in
    standardised units.  With lmi=True the pseudo-inertia of every updated link is constrained inside the fit,
    J(pi_nom + dpi) >= eps I (Wensing et al.), so the solution is physically consistent without projection."""
    from scipy.optimize import lsq_linear
    lb, ub = physical_bounds(model_nom)
    A, y = stack(data)
    n, p = A.shape
    scale = A.std(axis=0)
    live = scale >= 1e-12
    if not lmi:
        sc = np.where(live, scale, np.inf)
        As = A / sc
        Ar = np.vstack([As, np.sqrt(ridge * n) * np.eye(p)]); yr = np.concatenate([y, np.zeros(p)])
        lbs, ubs = np.nan_to_num(lb * sc, nan=-np.inf), np.nan_to_num(ub * sc, nan=np.inf)
        lbs[~live], ubs[~live] = -1e-12, 1e-12
        x = lsq_linear(Ar, yr, bounds=(lbs, ubs), method="bvls", lsmr_tol="auto", max_iter=500).x / sc
        x[~live] = 0.0
        return x
    import cvxpy as cp
    idx = np.flatnonzero(live)
    As = A[:, idx] / scale[idx]
    G = As.T @ As / n + ridge * np.eye(idx.size); g = As.T @ y / n
    L = np.linalg.cholesky(G)
    z = cp.Variable(idx.size)
    xi_live = cp.multiply(1.0 / scale[idx], z)
    S = np.zeros((p, idx.size)); S[idx, np.arange(idx.size)] = 1.0
    xi = S @ xi_live
    cons = []
    fl, fu = np.flatnonzero(np.isfinite(lb) & live), np.flatnonzero(np.isfinite(ub) & live)
    if fl.size:
        cons.append(xi[fl] >= lb[fl])
    if fu.size:
        cons.append(xi[fu] <= ub[fu])
    for b, bn in enumerate(ft.LINKS):
        pi = ft.body_pi(model_nom, bn) + xi[10 * b:10 * b + 10]
        m, hx, hy, hz, ixx, ixy, ixz, iyy, iyz, izz = [pi[k] for k in range(10)]
        tr = ixx + iyy + izz
        J = cp.Variable((4, 4), symmetric=True)
        cons += [J[0, 0] == 0.5 * tr - ixx, J[1, 1] == 0.5 * tr - iyy, J[2, 2] == 0.5 * tr - izz,
                 J[0, 1] == -ixy, J[0, 2] == -ixz, J[1, 2] == -iyz, J[0, 3] == hx, J[1, 3] == hy, J[2, 3] == hz, J[3, 3] == m,
                 J >> eps * np.eye(4)]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(L.T @ z) - 2 * g @ z), cons)
    for solver in ("CLARABEL", "SCS"):
        try:
            prob.solve(solver=solver)
        except Exception:
            continue
        if prob.status in ("optimal", "optimal_inaccurate"):
            break
    if z.value is None:
        raise RuntimeError(f"dense LMI fit failed: {prob.status}")
    out = np.zeros(p); out[idx] = z.value / scale[idx]
    return np.clip(out, lb, ub)


def cv_select(data, fit, grid, cols=None, one_se=True):
    """Leave-one-trajectory-out CV of a hyper-parameter grid.  fit(A, y, h) -> coefficients."""
    n = len(data)
    err = np.zeros((len(grid), n))
    for i in range(n):
        A_tr, y_tr = stack([d for k, d in enumerate(data) if k != i], cols)
        A_va, y_va = stack([data[i]], cols)
        for g, h in enumerate(grid):
            xi = fit(A_tr, y_tr, h)
            err[g, i] = np.sqrt(np.mean((y_va - A_va @ xi) ** 2))
    m, s = err.mean(axis=1), err.std(axis=1, ddof=1) / np.sqrt(n)
    best = int(np.argmin(m))
    if one_se:                                          # sparsest model within one SE of the best (grid ascending in sparsity)
        cand = np.flatnonzero(m <= m[best] + s[best])
        best = int(cand.max())
    return grid[best], dict(grid=list(grid), cv_rmse=m.tolist(), cv_se=s.tolist(), chosen=best)


def full_coef(xi_sub, cols):
    xi = np.zeros(len(NAMES))
    xi[cols] = xi_sub
    return xi


def identify(data, method, model_nom=None, thr_grid=None, ridge_full=1e-6):
    """Returns (xi (112,), info)."""
    if method == "nominal":
        return np.zeros(len(NAMES)), {}
    if method == "LS-payload":
        A, y = stack(data, IDX_PAYLOAD)
        return full_coef(ols(A, y), IDX_PAYLOAD), {}
    if method == "LS-struct":
        cols = IDX_PAYLOAD + IDX_COULVISC
        A, y = stack(data, cols)
        return full_coef(ols(A, y), cols), {}
    if method == "LS-full":
        A, y = stack(data)
        scale = A.std(axis=0); scale[scale < 1e-12] = np.inf
        xi = ols(A / scale, y, ridge_full * A.shape[0])
        return xi / scale, {}
    if method == "SINDy":
        grid = np.logspace(-3, 0.5, 15) if thr_grid is None else thr_grid
        thr, info = cv_select(data, stlsq, grid)
        A, y = stack(data)
        return stlsq(A, y, thr), dict(thr=float(thr), **info)
    if method == "SINDy-phys":
        grid = np.logspace(-3, 0.5, 15) if thr_grid is None else thr_grid
        lb, ub = physical_bounds(model_nom)
        fit = lambda A, y, h: stlsq_bounded(A, y, h, lb, ub)
        thr, info = cv_select(data, fit, grid)
        A, y = stack(data)
        return fit(A, y, thr), dict(thr=float(thr), **info)
    if method == "SINDy-group":
        grid = np.logspace(-3, 0.5, 15) if thr_grid is None else thr_grid
        lb, ub = physical_bounds(model_nom)
        fit = lambda A, y, h: stlsq_group(A, y, h, lb, ub)
        thr, info = cv_select(data, fit, grid)
        A, y = stack(data)
        return fit(A, y, thr), dict(thr=float(thr), **info)
    if method == "Forward":
        return forward_groups(data, model_nom)
    if method == "LS-bounded-r4":
        return dense_fit(data, model_nom, RIDGE_GROUP, lmi=False), {}
    if method == "LS-LMI":
        return dense_fit(data, model_nom, RIDGE_GROUP, lmi=True), {}
    if method == "BestSubset5-BIC":
        return best_subset_groups(data, model_nom, kmax=5, criterion="bic")
    if method == "BestSubset":
        return best_subset_groups(data, model_nom, kmax=4, criterion="cv")
    if method == "BestSubset-BIC":
        return best_subset_groups(data, model_nom, kmax=4, criterion="bic")
    if method == "LS-phys":
        lb, ub = physical_bounds(model_nom)
        A, y = stack(data)
        return stlsq_bounded(A, y, 0.0, lb, ub, iters=1), {}
    raise ValueError(method)


# ----------------------------------------------------------------------------- calibrated twin
def instantiate(model_nom, xi, verbose=False):
    """Write identified coefficients into a copy of the nominal model.  Returns (model, extra_fn, report).
    Inertial: pi_b + dpi_b if it is a valid rigid body, else applied as generalized force (lagged qdd).
    Coulomb -> frictionloss, viscous -> damping when the totals stay non-negative; other friction terms applied."""
    model = copy.copy(model_nom)
    rep = dict(inert_native=[], inert_applied=[], fric_native=[], fric_applied=[])
    xi = np.asarray(xi, float)
    dpi = xi[:N_INERT].reshape(7, 10)
    applied_bodies = []
    rep["projected"] = {}
    for i, bn in enumerate(ft.LINKS):
        if not np.any(dpi[i]):
            continue
        pi_new = ft.body_pi(model_nom, bn) + dpi[i]
        if not ft.params_from_pi(pi_new)[4]:
            pi_p = ft.project_pi(pi_new)
            rep["projected"][bn] = float(np.linalg.norm(pi_p - pi_new) / (np.linalg.norm(pi_new) + 1e-12))
            pi_new = pi_p
        ft.set_body_pi(model, bn, pi_new)
        rep["inert_native"].append(bn)
    xf = xi[N_INERT:N_INERT + N_FRIC].reshape(7, 5).copy()
    for j in range(7):
        dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ft.JOINTS[j])]
        if xf[j, 0] != 0 and model.dof_frictionloss[dof] + xf[j, 0] >= 0:
            model.dof_frictionloss[dof] += xf[j, 0]; xf[j, 0] = 0.0; rep["fric_native"].append(f"j{j + 1}:coul")
        if xf[j, 1] != 0 and model.dof_damping[dof] + xf[j, 1] >= 0:
            model.dof_damping[dof] += xf[j, 1]; xf[j, 1] = 0.0; rep["fric_native"].append(f"j{j + 1}:visc")
    rep["fric_applied"] = [f"j{j + 1}:{ft.FRIC_NAMES[b]}" for j in range(7) for b in range(5) if xf[j, b] != 0]
    xa = xi[105:112] if xi.size >= 112 else np.zeros(7)
    rep["armature"] = []
    for j in range(7):
        if xa[j] != 0:
            dof = model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ft.JOINTS[j])]
            model.dof_armature[dof] = max(model.dof_armature[dof] + xa[j], 0.0)
            rep["armature"].append(f"j{j + 1}")
    mujoco.mj_setConst(model, mujoco.MjData(model))
    extra = None
    if applied_bodies or np.any(xf):
        bodies = [ft.LINKS[i] for i in applied_bodies]
        dpi_app = dpi[applied_bodies].ravel() if applied_bodies else None
        cache = ft.RegressorCache(model, bodies) if bodies else None

        def extra(q, qd, qdd_prev):
            # The library coefficients are the extra torque the reference needs; in the twin they act as resistive
            # generalized forces, qfrc_applied = tau - Phi(qd) xi (same sign as the native friction-loss and damping).
            tau = -ft.friction_torque(xf, qd[None])[0]
            if bodies:
                tau = tau - cache(q[None], qd[None], qdd_prev[None])[0] @ dpi_app
            return tau
    if verbose:
        print(rep)
    return model, extra, rep


def twin_from(model_nom, xi, timestep=None):
    model, extra, rep = instantiate(model_nom, xi)
    if timestep is not None:
        model.opt.timestep = timestep
    return ft.Plant(model, extra), rep


# ----------------------------------------------------------------------------- metrics
def torque_metrics(data, xi):
    """Per-joint RMS of the remaining residual on data (all rows, no velocity mask) and the gap reduction."""
    A = np.concatenate([d["A"] for d in data], axis=0)
    r = np.concatenate([d["r"] for d in data], axis=0)
    res = r - A @ xi
    rms0 = np.sqrt((r ** 2).mean(axis=0)); rms = np.sqrt((res ** 2).mean(axis=0))
    return dict(rms_joint=rms.tolist(), rms_all=float(np.sqrt((res ** 2).mean())), rms0_all=float(np.sqrt((r ** 2).mean())),
                nrmse=float(np.sqrt((res ** 2).mean()) / np.sqrt((r ** 2).mean())),
                r2=float(1 - (res ** 2).sum() / ((r - r.mean(axis=0)) ** 2).sum()))


def twin_torque_metrics(model_nom, xi, data, vmin=0.1):
    """Inverse-dynamics residual of the executable twin (after projection and native write-back) on data.  The twin is
    driven by qfrc_applied = tau + extra, so the torque it needs is tau_twin = ID_twin(q, qd, qdd) - extra(q, qd).
    NRMSE uses the nominal residual of the same rows as denominator; 'masked' drops rows with |qd| < vmin."""
    model, extra, rep = instantiate(model_nom, xi)
    dd = mujoco.MjData(model)
    E, R, QD = [], [], []
    for d in data:
        tw = ft.inverse_batch(model, d["q"], d["qd"], d["qdd"], dd)
        if extra is not None:
            tw = tw - np.array([extra(q, v, np.zeros(ft.NJ)) for q, v in zip(d["q"], d["qd"])])
        E.append(d["tau"] - tw); R.append(d["r"]); QD.append(d["qd"])
    E, R, QD = np.concatenate(E), np.concatenate(R), np.concatenate(QD)
    keep = np.abs(QD) >= vmin
    eig = min(float(np.linalg.eigvalsh(ft.pseudo_inertia(ft.body_pi(model, bn))).min()) for bn in ft.LINKS)
    return dict(nrmse=float(np.sqrt((E ** 2).mean()) / np.sqrt((R ** 2).mean())), rms_all=float(np.sqrt((E ** 2).mean())),
                nrmse_masked=float(np.sqrt((E[keep] ** 2).mean()) / np.sqrt((R[keep] ** 2).mean())),
                projected=rep.get("projected", {}), n_applied_friction_terms=len(rep["fric_applied"]), min_pseudo_inertia_eig=eig)


def tcp_positions(model, q):
    d = mujoco.MjData(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    out = np.empty((q.shape[0], 3))
    for i in range(q.shape[0]):
        d.qpos[:] = q[i]; mujoco.mj_kinematics(model, d); out[i] = d.site_xpos[sid]
    return out


def replay_metrics(plant_twin, model_nom, log_real, T=None):
    """Closed-loop replay: the same controller and reference on the twin.  Compare joint and TCP trajectories."""
    T = log_real["t"][-1] + (log_real["t"][1] - log_real["t"][0]) if T is None else T
    ctrl = ft.PD(model_nom)
    log = plant_twin.simulate(ft.Fourier(log_real["seed"]), ctrl, T)
    n = min(len(log["t"]), len(log_real["t"]))
    dq = log["q"][:n] - log_real["q"][:n]
    p_t, p_r = tcp_positions(model_nom, log["q"][:n]), tcp_positions(model_nom, log_real["q"][:n])
    dtau = log["tau"][:n] - log_real["tau"][:n]
    return dict(q_rmse=np.sqrt((dq ** 2).mean(axis=0)).tolist(),
                tcp_rmse_mm=float(1e3 * np.sqrt(((p_t - p_r) ** 2).sum(axis=1).mean())),
                tcp_max_mm=float(1e3 * np.sqrt(((p_t - p_r) ** 2).sum(axis=1)).max()),
                tau_rmse=np.sqrt((dtau ** 2).mean(axis=0)).tolist())


def openloop_metrics(plant_twin, model_nom, log_real, window=0.5, stride=1.0):
    """Open-loop torque replay from the real state over windows; TCP error at the window end."""
    dt = log_real["t"][1] - log_real["t"][0]
    nw, ns = int(round(window / dt)), int(round(stride / dt))
    errs = []
    for k0 in range(0, len(log_real["t"]) - nw, ns):
        q = plant_twin.replay_torque(log_real["q"][k0], log_real["qd"][k0], log_real["tau"][k0:k0 + nw], dt_ctrl=dt)
        p_t = tcp_positions(model_nom, q[-1:]); p_r = tcp_positions(model_nom, log_real["q"][k0 + nw - 1:k0 + nw])
        errs.append(1e3 * np.linalg.norm(p_t - p_r))
    return dict(tcp_end_mm_mean=float(np.mean(errs)), tcp_end_mm_max=float(np.max(errs)), n_windows=len(errs))


def recovery_table(model_nom, xi, gap=ft.GAP_TRUE, true_groups=None):
    """Identified vs true physical quantities."""
    xi = np.asarray(xi, float)
    dpi = xi[:N_INERT].reshape(7, 10)
    dpi_true = ft.gap_true_dpi(model_nom, gap)
    xf = xi[N_INERT:N_INERT + N_FRIC].reshape(7, 5)
    xf_true = ft.gap_true_fric(model_nom, gap)
    xa = xi[105:112] if xi.size >= 112 else np.zeros(7)
    xa_true = ft.gap_true_arm(model_nom, gap)
    b = mujoco.mj_name2id(model_nom, mujoco.mjtObj.mjOBJ_BODY, "payload")
    p0 = model_nom.body_pos[b]
    out = {}
    m_hat = dpi[6, 0]
    out["payload_mass"] = (float(m_hat), float(dpi_true[6, 0]))
    com_hat = dpi[6, 1:4] / m_hat - p0 if abs(m_hat) > 1e-6 else np.full(3, np.nan)
    out["payload_com"] = (com_hat.tolist(), list(gap["payload"]["com"]))
    out["link3_mass_scale"] = (float(1 + dpi[2, 0] / ft.body_pi(model_nom, "fr3_link3")[0]),
                               float(gap["link_mass_scale"].get("fr3_link3", 1.0)))
    for j in (3, 5):
        out[f"j{j + 1}_coul"] = (float(xf[j, 0]), float(xf_true[j, 0]))
        out[f"j{j + 1}_visc"] = (float(xf[j, 1]), float(xf_true[j, 1]))
    out["armature"] = (xa.round(4).tolist(), xa_true.round(4).tolist())
    nz = np.flatnonzero(xi)
    if true_groups is None:
        true_groups = {"link3", "link7", "joint4", "joint6"} | ({"armature"} if np.any(xa_true) else set())
    found = {g for g, idx in GROUPS.items() if np.any(xi[idx])}
    out["n_nonzero"] = int(len(nz))
    out["groups_found"] = sorted(found)
    out["group_precision"] = float(len(found & true_groups) / max(len(found), 1))
    out["group_recall"] = float(len(found & true_groups) / len(true_groups))
    return out
