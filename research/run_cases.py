"""Other injected discrepancies (same engine).  Seven cases, fixed before any fit: six sparse cases with supports,
sizes and locations different from the main study, and one distributed (non-sparse) case that violates SAPC's
sparse-change assumption.  Every method receives all 15 candidate groups; the injected locations are never given.
Three sensor-noise draws at x1 per case; validation on seeds 11-12 with oracle derivatives as in run_gate.py.
Methods: SAPC (Forward), dense with pseudo-inertia LMIs (LS-LMI), dense LS (paper default).
Writes results/cases.json.   Usage: python run_cases.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"
import sys, json, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import fr3_twin as ft, identify as idf

TRAIN, VAL, T, DRAWS = [1, 2, 3], [11, 12], 20.0, 3
J = lambda k: f"fr3_joint{k}"
CASES = {
    "C1 payload only": dict(payload=dict(mass=1.2, com=(-0.03, 0.02, 0.05), inertia=(2e-3, 2e-3, 1e-3))),
    "C2 friction only, joint 2": dict(joints={J(2): dict(frictionloss=1.137 + 1.5, damping=0.21 + 0.4)}),
    "C3 link only, link 5": dict(link_mass_scale={"fr3_link5": 1.25}),
    "C4 payload + joints 1 and 7": dict(payload=dict(mass=0.4, com=(0.01, 0.02, 0.08), inertia=(5e-4, 5e-4, 3e-4)),
                                        joints={J(1): dict(frictionloss=1.137 + 2.0), J(7): dict(frictionloss=0.248 + 0.3)}),
    "C5 payload + link 6 + joint 3": dict(payload=dict(mass=1.0, com=(0.0, 0.0, 0.04), inertia=(1e-3, 1e-3, 6e-4)),
                                          link_mass_scale={"fr3_link6": 1.2}, joints={J(3): dict(damping=0.21 + 0.5)}),
    "C6 armature + joint 4 + link 2": dict(joints={J(5): dict(armature=0.074 + 0.05), J(4): dict(frictionloss=1.137 + 1.0)},
                                           link_mass_scale={"fr3_link2": 1.15}),
    "C7 distributed (non-sparse)": dict(payload=ft.GAP_TRUE["payload"], joints=ft.GAP_TRUE["joints"],
                                        link_mass_scale={f"fr3_link{i + 1}": s for i, s in
                                                         enumerate((1.04, 0.97, 1.05, 0.96, 1.03, 0.95, 1.02))}),
}
METHODS = {"SAPC": "Forward", "dense, pseudo-inertia LMI": "LS-LMI", "dense LS": "LS-full"}


def true_xi(nom, gap):
    xi = np.concatenate([ft.gap_true_dpi(nom, gap).ravel(), ft.gap_true_fric(nom, gap).ravel(), np.zeros(7)])
    for jn, mods in gap.get("joints", {}).items():
        if "armature" in mods:
            j = ft.JOINTS.index(jn); dof = nom.jnt_dofadr[nom.joint(jn).id]
            xi[105 + j] = mods["armature"] - nom.dof_armature[dof]
    return xi


def groups_of(xi, tol=0.0):
    return sorted(g for g, idx in idf.GROUPS.items() if np.any(np.abs(np.asarray(xi)[idx]) > tol))


def run_case(item):
    ci, (name, gap) = item
    nom = ft.make_nominal(); real = ft.make_real(gap=gap)
    xt = true_xi(nom, gap); true_groups = groups_of(xt)
    logs_tr = idf.collect(real, nom, TRAIN, T=T, noise_scale=1.0, noise_seed=0)
    logs_va = idf.collect(real, nom, VAL, T=T, noise_scale=1.0, noise_seed=1)
    data_va = idf.make_dataset(nom, logs_va, source="oracle")
    tw_true, _ = idf.twin_from(nom, xt)
    out = dict(case=name, gap=gap, true_groups=true_groups, xi_true=xt.tolist(),
               true_replay_mm=float(np.mean([idf.replay_metrics(tw_true, nom, lg)["tcp_rmse_mm"] for lg in logs_va])),
               nominal_replay_mm=float(np.mean([idf.replay_metrics(ft.Plant(ft.make_nominal()), nom, lg)["tcp_rmse_mm"] for lg in logs_va])),
               runs=[])
    for draw in range(DRAWS):
        rng = np.random.default_rng(7000 + 100 * ci + draw)
        for lg in logs_tr:
            lg["q_meas"], lg["tau_meas"] = ft.add_sensor_noise(lg, rng, scale=1.0)
        data_tr = idf.make_dataset(nom, logs_tr, source="sg", window=101)
        for label, meth in METHODS.items():
            t0 = time.time(); xi, info = idf.identify(data_tr, meth, model_nom=nom); fit_s = time.time() - t0
            found = groups_of(xi)
            tw, _ = idf.twin_from(nom, xi)
            dm = np.asarray(xi[:70]).reshape(7, 10)[:, 0]; dmt = xt[:70].reshape(7, 10)[:, 0]
            row = dict(draw=draw, method=label, groups=found, subset=info.get("subset"),
                       precision=len(set(found) & set(true_groups)) / max(len(found), 1),
                       recall=len(set(found) & set(true_groups)) / len(true_groups),
                       exact=set(found) == set(true_groups),
                       link_mass_increments=dm.tolist(), link_mass_increments_true=dmt.tolist(),
                       friction=np.asarray(xi[70:105]).reshape(7, 5)[:, :2].tolist(),
                       friction_true=xt[70:105].reshape(7, 5)[:, :2].tolist(),
                       armature=np.asarray(xi[105:112]).tolist(), armature_true=xt[105:112].tolist(),
                       lib_nrmse=idf.torque_metrics(data_va, xi)["nrmse"], twin_nrmse=idf.twin_torque_metrics(nom, xi, data_va)["nrmse"],
                       replay_mm=float(np.mean([idf.replay_metrics(tw, nom, lg)["tcp_rmse_mm"] for lg in logs_va])), fit_s=fit_s,
                       xi=np.asarray(xi).tolist())
            if "payload" in gap:
                com = np.asarray(xi[61:64]) / xi[60] - nom.body_pos[nom.body("payload").id] if abs(xi[60]) > 1e-6 else np.full(3, np.nan)
                row.update(payload_mass=float(xi[60]), payload_mass_true=float(gap["payload"]["mass"]),
                           payload_com_error_mm=float(1e3 * np.linalg.norm(com - np.asarray(gap["payload"]["com"]))))
            out["runs"].append(row)
            print(f"[{name}] draw {draw} {label:26s} groups {found} | P/R {row['precision']:.2f}/{row['recall']:.2f} "
                  f"| payload {row.get('payload_mass', float('nan')):.3f} | replay {row['replay_mm']:.4f} mm", flush=True)
    return out


if __name__ == "__main__":
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(CASES)) as pool:
        results = list(pool.map(run_case, list(enumerate(CASES.items()))))
    json.dump(dict(identify_version=idf.IDENTIFY_VERSION, draws=DRAWS, cases=results, wall_s=time.time() - t0),
              open("results/cases.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s", flush=True)
