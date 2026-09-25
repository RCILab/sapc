"""Task A: proposed method = physics-constrained Forward group selection.  Compare with the exhaustive BestSubset
benchmark on the same-engine data at the four E1/E2 conditions: support, parameter errors, validation torque NRMSE,
replay TCP RMSE, runtime.  Noise realisations reproduce run_gate.py exactly."""
import sys, json, time, numpy as np
import fr3_twin as ft, identify as idf

TRAIN, VAL, T = [1, 2, 3], [11, 12], 20.0
CONDS = [("E1 oracle", "oracle", None), ("E2 x0", "sg", 0.0), ("E2 x1", "sg", 1.0), ("E2 x3", "sg", 3.0)]
logf = open("results/forward_compare_log.txt", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush(); sys.stdout.flush()


nom = ft.make_nominal(); real = ft.make_real()
logs_tr = idf.collect(real, nom, TRAIN, T=T, noise_scale=1.0, noise_seed=0)
logs_va = idf.collect(real, nom, VAL, T=T, noise_scale=1.0, noise_seed=1)
data_va = idf.make_dataset(nom, logs_va, source="oracle")
dpi_true = ft.gap_true_dpi(nom, ft.GAP_TRUE); xf_true = ft.gap_true_fric(nom, ft.GAP_TRUE)
out = {}
for tag, source, scale in CONDS:
    if source == "sg":
        rng = np.random.default_rng(int(10 * scale) + 7)
        for lg in logs_tr:
            lg["q_meas"], lg["tau_meas"] = ft.add_sensor_noise(lg, rng, scale=scale)
    data_tr = idf.make_dataset(nom, logs_tr, source=source, window=101)
    out[tag] = {}
    for meth in ("Forward", "BestSubset"):
        t0 = time.time(); xi, info = idf.identify(data_tr, meth, model_nom=nom); dt = time.time() - t0
        rec = idf.recovery_table(nom, xi); tm = idf.torque_metrics(data_va, xi)
        twin, rep = idf.twin_from(nom, xi)
        tcp = float(np.mean([idf.replay_metrics(twin, nom, lg)["tcp_rmse_mm"] for lg in logs_va]))
        com_err = float(np.linalg.norm(np.array(rec["payload_com"][0]) - np.array(rec["payload_com"][1])))
        out[tag][meth] = dict(subset=info.get("subset"), fit_s=dt, val_nrmse=tm["nrmse"], tcp=tcp, payload_mass=rec["payload_mass"][0],
                              com_err_mm=1e3 * com_err, j4_coul=rec["j4_coul"][0], j4_visc=rec["j4_visc"][0], j6_coul=rec["j6_coul"][0],
                              link3_scale=rec["link3_mass_scale"][0], group_precision=rec["group_precision"], group_recall=rec["group_recall"],
                              path=info.get("path"), top=info.get("top"))
        say(f"[{tag}] {meth:10s} {dt:6.0f} s | subset {info.get('subset')} | val NRMSE {tm['nrmse']:.3f} | TCP {tcp:.3f} mm | payload {rec['payload_mass'][0]:.3f} kg, CoM err {1e3 * com_err:.2f} mm | j4 {rec['j4_coul'][0]:.2f}/{rec['j4_visc'][0]:.2f} j6 {rec['j6_coul'][0]:.2f} | P/R {rec['group_precision']:.2f}/{rec['group_recall']:.2f}")
    json.dump(out, open("results/forward_compare.json", "w"), indent=1)
say("done")
