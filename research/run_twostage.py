"""Optional two-stage cross-engine calibration (paper_plan v3, item 16).

Stage 1: calibrate the nominal MuJoCo twin on NOMINAL Gazebo data -> engine-description mismatch (expected: armature,
         possibly joint 7).  The result becomes the engine-calibrated nominal twin T0'.
Stage 2: with T0' as the nominal model (regressor and residual both from T0'), identify the injected physical
         discrepancy from the PERTURBED Gazebo data.  Question: does removing the engine mismatch first reduce the
         nuisance groups (link 2, joint 7) and recover the injected groups more cleanly?
Reports support, payload/CoM/friction/armature estimates, validation torque NRMSE and replay TCP RMSE for
one-stage SAPC, two-stage SAPC and the stage-2 dense fit.
"""
import sys, json, time, numpy as np
import fr3_twin as ft, identify as idf

logf = open("results/twostage_log.txt", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush(); sys.stdout.flush()


def gz_log(path, seed, noise_scale=1.0, noise_seed=0):
    G = np.load(path)
    log = dict(t=G["t"], q=G["q"], qd=G["qd"], qdd=np.zeros_like(G["q"]), tau=G["tau"], q_ref=G["q_ref"], seed=seed)
    log["q_meas"], log["tau_meas"] = ft.add_sensor_noise(log, np.random.default_rng(noise_seed + seed), scale=noise_scale)
    return log


ft.DEFAULT_KD = ft.KD_CROSS
nom = ft.make_nominal()
nom_tr = [gz_log(f"results/gz/nom_seed{s}.npz", s) for s in (1, 2, 3)]
ref_tr = [gz_log(f"results/gz/ref_seed{s}.npz", s) for s in (1, 2, 3)]
ref_va = [gz_log(f"results/gz/ref_seed{s}.npz", s) for s in (11, 12)]


def tcp(twin, logs):
    return float(np.mean([idf.replay_metrics(twin, nom, lg)["tcp_rmse_mm"] for lg in logs]))


out = {}
# ---- stage 1
t0 = time.time()
data_nom = idf.make_dataset(nom, nom_tr, source="sg", window=101)
xi1, info1 = idf.identify(data_nom, "Forward", model_nom=nom)
T0p, extra1, rep1 = idf.instantiate(nom, xi1)
rec1 = idf.recovery_table(nom, xi1)
say(f"stage 1 (nominal Gazebo -> engine mismatch): groups {info1['subset']} | armature {np.round(rec1['armature'][0], 3).tolist()} | extras {rep1['fric_applied']} | {time.time() - t0:.0f} s")
out["stage1"] = dict(subset=info1["subset"], armature=rec1["armature"][0], xi=xi1.tolist(), fric_applied=rep1["fric_applied"])
if extra1 is not None:
    say("  note: stage-1 twin carries non-native friction terms; stage 2 uses the native part of T0' as the nominal model")

# ---- stage 2: T0' (native part) as the nominal model for the perturbed data
data_ref2 = idf.make_dataset(T0p, ref_tr, source="sg", window=101)
data_va2 = idf.make_dataset(T0p, ref_va, source="sg", window=101)
t0 = time.time()
xi2, info2 = idf.identify(data_ref2, "Forward", model_nom=T0p)
rec2 = idf.recovery_table(T0p, xi2)
tw2, rep2 = idf.twin_from(T0p, xi2)
res2 = dict(subset=info2["subset"], path=info2["path"], nrmse=idf.torque_metrics(data_va2, xi2)["nrmse"], tcp=tcp(tw2, ref_va),
            payload_mass=rec2["payload_mass"][0], com=rec2["payload_com"][0], j4=(rec2["j4_coul"][0], rec2["j4_visc"][0]), j6=rec2["j6_coul"][0],
            armature=rec2["armature"][0], link3=rec2["link3_mass_scale"][0])
out["stage2_forward"] = res2
say(f"stage 2 SAPC on T0': groups {info2['subset']} | val NRMSE {res2['nrmse']:.3f} | TCP {res2['tcp']:.3f} mm | payload {res2['payload_mass']:.3f} kg com {np.round(res2['com'], 4).tolist()} "
    f"| j4 {res2['j4'][0]:.2f}/{res2['j4'][1]:.2f} j6 {res2['j6']:.2f} | armature delta {np.round(res2['armature'], 3).tolist()} | link3 {res2['link3']:.3f} | {time.time() - t0:.0f} s")
xi2d, _ = idf.identify(data_ref2, "LS-full", model_nom=T0p)
tw2d, _ = idf.twin_from(T0p, xi2d); rec2d = idf.recovery_table(T0p, xi2d)
out["stage2_dense"] = dict(nrmse=idf.torque_metrics(data_va2, xi2d)["nrmse"], tcp=tcp(tw2d, ref_va), payload_mass=rec2d["payload_mass"][0])
say(f"stage 2 dense on T0': val NRMSE {out['stage2_dense']['nrmse']:.3f} | TCP {out['stage2_dense']['tcp']:.3f} mm | payload {rec2d['payload_mass'][0]:.3f} kg")

# ---- one-stage reference (from run_crossengine): SAPC directly on the perturbed data with the nominal twin
data_ref1 = idf.make_dataset(nom, ref_tr, source="sg", window=101)
data_va1 = idf.make_dataset(nom, ref_va, source="sg", window=101)
xi_one, info_one = idf.identify(data_ref1, "Forward", model_nom=nom)
tw_one, _ = idf.twin_from(nom, xi_one); rec_one = idf.recovery_table(nom, xi_one)
out["one_stage"] = dict(subset=info_one["subset"], nrmse=idf.torque_metrics(data_va1, xi_one)["nrmse"], tcp=tcp(tw_one, ref_va), payload_mass=rec_one["payload_mass"][0])
say(f"one-stage SAPC: groups {info_one['subset']} | val NRMSE {out['one_stage']['nrmse']:.3f} | TCP {out['one_stage']['tcp']:.3f} mm | payload {rec_one['payload_mass'][0]:.3f} kg")
json.dump(out, open("results/twostage.json", "w"), indent=1)
say("done")
