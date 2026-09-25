"""Task C: cross-engine decomposition and the 'calibration absorbs engine discrepancy' analysis.

Three levels on the validation trajectories (seeds 11, 12), all with the same controller (KD_CROSS):
  L1 engine-only        : Gazebo nominal  vs MuJoCo nominal
  L2 engine + physical  : Gazebo perturbed vs MuJoCo nominal
  L3 post-calibration   : Gazebo perturbed vs MuJoCo calibrated (Forward, LS-full)
Ablations:
  (a) Forward calibrated on Gazebo NOMINAL data (no injected gap): which groups are selected and how much of L1 is absorbed.
  (b) Forward on perturbed data restricted to the injected + armature groups (no nuisance groups): TCP / NRMSE change.
  (c) Forward without the armature group.
Needs results/gz/nom_seed{1,2,3,11}.npz and ref_seed*.npz.
"""
import sys, json, time, numpy as np
import fr3_twin as ft, identify as idf


def gz_log(path, seed, noise_scale=1.0, noise_seed=0):
    G = np.load(path)
    log = dict(t=G["t"], q=G["q"], qd=G["qd"], qdd=np.zeros_like(G["q"]), tau=G["tau"], q_ref=G["q_ref"], seed=seed)
    log["q_meas"], log["tau_meas"] = ft.add_sensor_noise(log, np.random.default_rng(noise_seed + seed), scale=noise_scale)
    return log


logf = open("results/crossengine_ablation_log.txt", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush(); sys.stdout.flush()


ft.DEFAULT_KD = ft.KD_CROSS
nom = ft.make_nominal()
ref_tr = [gz_log(f"results/gz/ref_seed{s}.npz", s) for s in (1, 2, 3)]
ref_va = [gz_log(f"results/gz/ref_seed{s}.npz", s) for s in (11, 12)]
nom_tr = [gz_log(f"results/gz/nom_seed{s}.npz", s) for s in (1, 2, 3)]
nom_va = [gz_log(f"results/gz/nom_seed{s}.npz", s) for s in (11,)]
data_ref = idf.make_dataset(nom, ref_tr, source="sg", window=101)
data_ref_va = idf.make_dataset(nom, ref_va, source="sg", window=101)
data_nom = idf.make_dataset(nom, nom_tr, source="sg", window=101)
data_nom_va = idf.make_dataset(nom, nom_va, source="sg", window=101)
out = {}


def tcp(twin, logs):
    return float(np.mean([idf.replay_metrics(twin, nom, lg)["tcp_rmse_mm"] for lg in logs]))


# L1 / L2
tw0 = ft.Plant(ft.make_nominal())
out["L1_engine_only_tcp"] = tcp(tw0, nom_va); out["L1_nrmse"] = idf.torque_metrics(data_nom_va, np.zeros(112))["nrmse"]
out["L2_engine_plus_physical_tcp"] = tcp(tw0, ref_va)
say(f"L1 engine-only: TCP {out['L1_engine_only_tcp']:.3f} mm (nominal Gazebo vs nominal MuJoCo, seed 11); torque residual RMS {idf.torque_metrics(data_nom_va, np.zeros(112))['rms0_all']:.3f} Nm")
say(f"L2 engine+physical: TCP {out['L2_engine_plus_physical_tcp']:.3f} mm")

# (a) calibrate on nominal Gazebo data: what does the twin absorb?
t0 = time.time(); xi_n, info_n = idf.identify(data_nom, "Forward", model_nom=nom)
tw_n, _ = idf.twin_from(nom, xi_n); rec_n = idf.recovery_table(nom, xi_n)
out["a_nominal_calibration"] = dict(subset=info_n["subset"], path=info_n["path"], tcp_nom_va=tcp(tw_n, nom_va),
                                    nrmse_nom_va=idf.torque_metrics(data_nom_va, xi_n)["nrmse"], armature=rec_n["armature"][0],
                                    payload_mass=rec_n["payload_mass"][0], xi=xi_n.tolist())
say(f"(a) Forward on NOMINAL Gazebo data: groups {info_n['subset']} | TCP on nominal val {out['a_nominal_calibration']['tcp_nom_va']:.3f} mm (was {out['L1_engine_only_tcp']:.3f}) "
    f"| NRMSE {out['a_nominal_calibration']['nrmse_nom_va']:.3f} | armature {np.round(rec_n['armature'][0], 3).tolist()} | spurious payload {rec_n['payload_mass'][0]:.3f} kg | {time.time() - t0:.0f} s")

# L3 and ablations (b), (c) on the perturbed reference
xi_f, info_f = idf.identify(data_ref, "Forward", model_nom=nom)
tw_f, _ = idf.twin_from(nom, xi_f)
out["L3_forward"] = dict(subset=info_f["subset"], tcp=tcp(tw_f, ref_va), nrmse=idf.torque_metrics(data_ref_va, xi_f)["nrmse"])
say(f"L3 Forward ({info_f['subset']}): TCP {out['L3_forward']['tcp']:.3f} mm, val NRMSE {out['L3_forward']['nrmse']:.3f}")
xi_d, _ = idf.identify(data_ref, "LS-full", model_nom=nom); tw_d, _ = idf.twin_from(nom, xi_d)
out["L3_dense"] = dict(tcp=tcp(tw_d, ref_va), nrmse=idf.torque_metrics(data_ref_va, xi_d)["nrmse"])
say(f"L3 dense: TCP {out['L3_dense']['tcp']:.3f} mm, val NRMSE {out['L3_dense']['nrmse']:.3f}")
for label, groups in (("b_injected_plus_armature", ["link3", "link7", "joint4", "joint6", "armature"]),
                      ("b_injected_only", ["link3", "link7", "joint4", "joint6"]),
                      ("c_forward_minus_armature", [g for g in info_f["subset"] if g != "armature"]),
                      ("c_forward_minus_nuisance", [g for g in info_f["subset"] if g not in ("link2", "joint7")])):
    xi_r, info_r = idf.forward_groups(data_ref, nom, groups=groups, kmax=len(groups))
    # force the full given group set (forward may stop early): refit bounded LS on all of them
    cols = sorted(sum((idf.GROUPS[g] for g in groups), []))
    from scipy.optimize import lsq_linear
    A, y = idf.stack(data_ref); lb, ub = idf.physical_bounds(nom)
    sc = A[:, cols].std(0); sc[sc < 1e-12] = np.inf
    Ar = np.vstack([A[:, cols] / sc, np.sqrt(1e-4 * A.shape[0]) * np.eye(len(cols))]); yr = np.concatenate([y, np.zeros(len(cols))])
    xi_r = idf.full_coef(lsq_linear(Ar, yr, bounds=(lb[cols] * sc, ub[cols] * sc), method="bvls", lsmr_tol="auto", max_iter=200).x / sc, cols)
    tw_r, _ = idf.twin_from(nom, xi_r); rec_r = idf.recovery_table(nom, xi_r)
    out[label] = dict(groups=groups, tcp=tcp(tw_r, ref_va), nrmse=idf.torque_metrics(data_ref_va, xi_r)["nrmse"], payload_mass=rec_r["payload_mass"][0],
                      com=rec_r["payload_com"][0], armature=rec_r["armature"][0])
    say(f"{label:26s} groups {groups}: TCP {out[label]['tcp']:.3f} mm, val NRMSE {out[label]['nrmse']:.3f}, payload {rec_r['payload_mass'][0]:.3f} kg, armature {np.round(rec_r['armature'][0], 3).tolist()}")
json.dump(out, open("results/crossengine_ablation.json", "w"), indent=1)
say("done")
