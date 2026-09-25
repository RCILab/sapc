"""Identifiability audit of the 70 inertial columns of the MuJoCo FR3 twin.

1. Structural rank on generic random states versus the rank reached by the identification excitation.
2. Physical meaning of link 3's null direction: a point mass at the intersection of joint axes 1-3, compared with the
   uniform scaling of link 3 and a point mass at its centre of mass.
3. Shared dimensions of neighbouring links (structural), and per-group identically zero columns.
4. How much of the injected payload torque other links can reproduce (the payload is identifiable only under the
   sparse-change assumption), and how closely link 4 can mimic the injected link-3 signal.
5. Torque difference between fitted and true corrections on generic states (dense LS, dense LMI, SAPC).
Writes results/identifiability_audit.json.   Usage: python run_identifiability_audit.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import json
import numpy as np
import mujoco
import fr3_twin as ft, identify as idf

nom = ft.make_nominal(); real = ft.make_real()
res = dict(identify_version=idf.IDENTIFY_VERSION)
rms = lambda v: float(np.sqrt(np.mean(np.asarray(v) ** 2)))


def std_cols(M):
    s = M.std(axis=0); keep = s > 1e-12
    return M[:, keep] / s[keep], keep, s


def rank(M, tol=1e-8):
    if M.shape[1] == 0:
        return 0
    s = np.linalg.svd(std_cols(M)[0], compute_uv=False)
    return int((s > tol * s[0]).sum())


# ---------------------------------------------------------------- 1. structural vs excitation rank
rng = np.random.default_rng(1)
lim = np.array([[-2.7, 2.7], [-1.7, 1.7], [-2.8, 2.8], [-3.0, -0.2], [-2.7, 2.7], [0.6, 4.4], [-2.9, 2.9]])
qg = rng.uniform(lim[:, 0], lim[:, 1], (400, 7)); qdg = rng.uniform(-2, 2, (400, 7)); qddg = rng.uniform(-8, 8, (400, 7))
Yg, _ = ft.inertial_regressor(nom, qg, qdg, qddg)
Ag = Yg.reshape(-1, 70)
zero_cols = [idf.NAMES[i] for i in range(70) if np.abs(Ag[:, i]).max() < 1e-9]
sg = np.linalg.svd(std_cols(Ag)[0], compute_uv=False); sg = sg / sg[0]
logs_tr = idf.collect(real, nom, [1, 2, 3], T=20.0, noise_scale=1.0, noise_seed=0)
data_or = idf.make_dataset(nom, logs_tr, source="oracle")
Ae, _ = idf.stack(data_or)                                   # identification rows (|qd| >= 0.1), oracle derivatives
se = np.linalg.svd(std_cols(Ae[:, :70])[0], compute_uv=False); se = se / se[0]
r_struct, r_exc = int((sg > 1e-8).sum()), int((se > 1e-8).sum())
res["rank"] = dict(structural=r_struct, excitation=r_exc, identically_zero_columns=zero_cols,
                   structural_singular_values_40_46=sg[40:46].tolist(), excitation_singular_values_40_46=se[40:46].tolist(),
                   excitation_condition_number_over_base=float(se[0] / se[r_exc - 1]))
print("rank", res["rank"], flush=True)

# ---------------------------------------------------------------- 2. link 3 null direction
d = mujoco.MjData(nom); d.qpos[:] = ft.Q_HOME; mujoco.mj_forward(nom, d)
Msum, bsum, lines = np.zeros((3, 3)), np.zeros(3), []
for jn in ft.JOINTS[:3]:
    j = mujoco.mj_name2id(nom, mujoco.mjtObj.mjOBJ_JOINT, jn)
    a = d.xaxis[j] / np.linalg.norm(d.xaxis[j]); c = d.xanchor[j].copy()
    P = np.eye(3) - np.outer(a, a); Msum += P; bsum += P @ c; lines.append((a, c))
p_star = np.linalg.solve(Msum, bsum)
b3 = mujoco.mj_name2id(nom, mujoco.mjtObj.mjOBJ_BODY, "fr3_link3")
p_loc = d.xmat[b3].reshape(3, 3).T @ (p_star - d.xpos[b3])
pi3 = ft.body_pi(nom, "fr3_link3")
dirs = {"point mass at axis intersection": ft.pi_from_params(1.0, p_loc, (0, 0, 0)),
        "point mass at link-3 CoM": ft.pi_from_params(1.0, nom.body_ipos[b3], (0, 0, 0)),
        "uniform scaling of link 3": pi3 / pi3[0]}
A3e = Ae[:, 20:30]; s3 = A3e.std(axis=0)
_, sv3, Vt3 = np.linalg.svd(A3e / s3, full_matrices=False)
nullv = Vt3[-1] / s3; nullv = nullv / nullv[0]
Y3g = Yg[:, :, 20:30]
res["link3"] = dict(axis_intersection_world=p_star.tolist(),
                    axis_line_distances_m=[float(np.linalg.norm((np.eye(3) - np.outer(a, a)) @ (p_star - c))) for a, c in lines],
                    intersection_in_link3_frame=p_loc.tolist(), relative_singular_values=(sv3 / sv3[0]).tolist(),
                    null_vector_per_kg=nullv.tolist(),
                    torque_rms_per_kg={k: rms(Y3g @ v) for k, v in dict(dirs, **{"excitation null vector": nullv}).items()},
                    cosine_null_vs_point_mass=float(abs(nullv @ dirs["point mass at axis intersection"]) /
                                                    np.linalg.norm(nullv) / np.linalg.norm(dirs["point mass at axis intersection"])))
print("link3", res["link3"]["torque_rms_per_kg"], flush=True)

# ---------------------------------------------------------------- 3. group ranks and shared dimensions (structural)
G = idf.GROUPS
ranks = {g: rank(Ag[:, G[g]]) for g in [f"link{i}" for i in range(1, 8)]}
pairs = {}
for a, b in (("link2", "link3"), ("link3", "link4"), ("link4", "link5"), ("link5", "link6"), ("link6", "link7")):
    rab = rank(Ag[:, G[a] + G[b]])
    pairs[f"{a}+{b}"] = dict(rank=rab, shared=ranks[a] + ranks[b] - rab)
others7 = [c for c in range(70) if c not in G["link7"]]
res["groups"] = dict(ranks=ranks, pairs=pairs,
                     link7_conditional_rank_given_all_other_links=r_struct - rank(Ag[:, others7]))
print("groups", res["groups"], flush=True)

# ---------------------------------------------------------------- 4. payload and link-3 signals
dpi_true = ft.gap_true_dpi(nom, ft.GAP_TRUE)
pay = Yg[:, :, 60:70] @ dpi_true[6]


def explained(sig, cols):
    M = Yg[:, :, cols].reshape(-1, len(cols)); M = M[:, np.abs(M).max(0) > 1e-9]
    Q, _ = np.linalg.qr(M / M.std(0)); v = sig.ravel()
    return float(1 - np.sum((v - Q @ (Q.T @ v)) ** 2) / np.sum(v ** 2))


res["payload_torque_explained_by"] = {"link6": explained(pay, G["link6"]), "links 1-6": explained(pay, list(range(60)))}
sig3 = Yg[:, :, 20:30] @ dpi_true[2]
res["link3_signal_explained_by"] = {"link4": explained(sig3, G["link4"]), "link2": explained(sig3, G["link2"]),
                                    "links 2 and 4": explained(sig3, G["link2"] + G["link4"])}
print("signals", res["payload_torque_explained_by"], res["link3_signal_explained_by"], flush=True)

# ---------------------------------------------------------------- 5. fitted corrections vs truth on generic states
rng17 = np.random.default_rng(17)
for lg in logs_tr:
    lg["q_meas"], lg["tau_meas"] = ft.add_sensor_noise(lg, rng17, scale=1.0)
data_sg = idf.make_dataset(nom, logs_tr, source="sg", window=101)
ref = rms(Yg @ dpi_true.ravel())
res["generic_torque_error"] = {"injected inertial torque rms": ref}
for name, meth in (("SAPC", "Forward"), ("dense LS", "LS-full"), ("dense, pseudo-inertia LMI", "LS-LMI")):
    xi, _ = idf.identify(data_sg, meth, model_nom=nom)
    dd = xi[:70] - dpi_true.ravel()
    res["generic_torque_error"][name] = dict(rms=rms(Yg @ dd), relative=rms(Yg @ dd) / ref, parameter_distance=float(np.linalg.norm(dd)),
                                             payload_mass=float(xi[60]))
print("generic", res["generic_torque_error"], flush=True)
json.dump(res, open("results/identifiability_audit.json", "w"), indent=1)
print("done", flush=True)
