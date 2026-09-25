"""Forward group selection (SAPC) on the exact noise realisation of run_gate.py (E2, SG101, noise x0/x1/x3).

run_gate.py draws the sensor noise with a fresh generator per trajectory (np.random.default_rng(int(10*scale)+7) inside
the log loop), whereas run_forward_compare.py uses one generator for all three training logs, so the two scripts see
different noise realisations at x1 and x3.  This script reproduces the run_gate.py realisation so that Table 2 of the
paper (dense LS, bounded elementwise sparse, forward selection, exhaustive best-subset) is reported on one realisation.
Writes results/forward_gate.json.
"""
import sys, json, time, numpy as np
import fr3_twin as ft, identify as idf

TRAIN, VAL, T = [1, 2, 3], [11, 12], 20.0
logf = open("results/forward_gate_log.txt", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush(); sys.stdout.flush()


nom = ft.make_nominal(); real = ft.make_real()
t0 = time.time()
logs_tr = idf.collect(real, nom, TRAIN, T=T, noise_scale=1.0, noise_seed=0)
logs_va = idf.collect(real, nom, VAL, T=T, noise_scale=1.0, noise_seed=1)
say(f"collected {len(TRAIN)} train + {len(VAL)} val trajectories, {time.time() - t0:.0f} s")
data_va = idf.make_dataset(nom, logs_va, source="oracle")
out = {}
for scale in (0.0, 1.0, 3.0):
    tag = f"E2 noise x{scale} SG101"
    for lg in logs_tr:                                   # identical to run_gate.py: fresh generator per trajectory
        lg["q_meas"], lg["tau_meas"] = ft.add_sensor_noise(lg, np.random.default_rng(int(10 * scale) + 7), scale=scale)
    data_tr = idf.make_dataset(nom, logs_tr, source="sg", window=101)
    t1 = time.time(); xi, info = idf.identify(data_tr, "Forward", model_nom=nom); dt = time.time() - t1
    rec = idf.recovery_table(nom, xi); tm = idf.torque_metrics(data_va, xi)
    twin, rep = idf.twin_from(nom, xi)
    replay = [idf.replay_metrics(twin, nom, lg) for lg in logs_va]
    openl = [idf.openloop_metrics(twin, nom, lg, window=0.2) for lg in logs_va]
    tcp = float(np.mean([r["tcp_rmse_mm"] for r in replay])); ol = float(np.mean([o["tcp_end_mm_mean"] for o in openl]))
    com_err = 1e3 * float(np.linalg.norm(np.array(rec["payload_com"][0]) - np.array(rec["payload_com"][1])))
    out[tag] = dict(subset=info.get("subset"), path=info.get("path"), fit_s=dt, val_nrmse=tm["nrmse"], tcp=tcp, openloop_0p2s=ol,
                    payload_mass=rec["payload_mass"][0], payload_com=rec["payload_com"][0], com_err_mm=com_err,
                    j4_coul=rec["j4_coul"][0], j4_visc=rec["j4_visc"][0], j6_coul=rec["j6_coul"][0], link3_scale=rec["link3_mass_scale"][0],
                    group_precision=rec["group_precision"], group_recall=rec["group_recall"], n_nonzero=rec["n_nonzero"],
                    replay=replay, openloop=openl, twin=rep, xi=xi.tolist())
    say(f"[{tag}] Forward {dt:5.0f} s | subset {info.get('subset')} | val NRMSE {tm['nrmse']:.4f} | TCP {tcp:.4f} mm | open-loop {ol:.3f} mm"
        f" | payload {rec['payload_mass'][0]:.4f} kg, CoM err {com_err:.2f} mm | j4 {rec['j4_coul'][0]:.3f}/{rec['j4_visc'][0]:.3f}"
        f" j6 {rec['j6_coul'][0]:.3f} | link3 {rec['link3_mass_scale'][0]:.3f} | P/R {rec['group_precision']:.2f}/{rec['group_recall']:.2f}")
    json.dump(out, open("results/forward_gate.json", "w"), indent=1)
say("done")
