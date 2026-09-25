"""E3: PINN-SR hybrid vs Savitzky-Golay derivatives at heavy sensor noise (scale 3): derivative error, identification,
twin fidelity.  Same trajectories as run_gate.py."""
import sys, json, time, numpy as np
import fr3_twin as ft, identify as idf, pinnsr
SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
logf = open("results/pinnsr_log.txt", "a", encoding="utf-8")
def say(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush(); sys.stdout.flush()
nom = ft.make_nominal(); real = ft.make_real()
logs_tr = idf.collect(real, nom, [1, 2, 3], T=20.0, noise_scale=SCALE, noise_seed=int(10 * SCALE) + 7)
logs_va = idf.collect(real, nom, [11, 12], T=20.0, noise_scale=1.0, noise_seed=1)
data_va = idf.make_dataset(nom, logs_va, source="oracle")
say(f"=== E3 noise x{SCALE}: PINN-SR surrogate vs SG101 ===")
out = {}
# SG reference
data_sg = idf.make_dataset(nom, logs_tr, source="sg", window=101)
def deriv_err(data):
    e = []
    for k, d in enumerate(data):
        dt = logs_tr[k]["t"][1] - logs_tr[k]["t"][0]; n0 = int(round(0.2 / dt)); sl = slice(n0, len(logs_tr[k]["t"]) - n0, 5)
        e.append(dict(qd=float(np.sqrt(((d["qd"] - logs_tr[k]["qd"][sl]) ** 2).mean())), qdd=float(np.sqrt(((d["qdd"] - logs_tr[k]["qdd"][sl]) ** 2).mean()))))
    return e
say("SG derivative errors:", deriv_err(data_sg))
t0 = time.time()
ps = pinnsr.PINNSR(nom, sigma_q=1e-4 * SCALE, pre_steps=1500, ref_steps=150, rounds=2, batch_phys=192, verbose=say).fit(logs_tr)
say(f"PINN-SR fit {time.time() - t0:.0f} s; history {ps.history}; smoothing {ps.lam_s}")
say("PINN-SR derivative errors:", deriv_err(ps.data))
for name, data in (("SG101", data_sg), ("PINN-SR", ps.data)):
    for meth in ("SINDy-phys", "Forward"):     # Forward = SAPC, the proposed method
        t0 = time.time(); xi, info = idf.identify(data, meth, model_nom=nom)
        tw, rep = idf.twin_from(nom, xi); rec = idf.recovery_table(nom, xi)
        replay = [idf.replay_metrics(tw, nom, lg) for lg in logs_va]
        out[f"{name}/{meth}"] = dict(val=idf.torque_metrics(data_va, xi), recovery=rec, replay=replay, info={k: v for k, v in info.items() if k in ("subset", "thr")})
        say(f"[{name}] {meth:10s} val NRMSE {out[f'{name}/{meth}']['val']['nrmse']:.3f} replay TCP {np.mean([r['tcp_rmse_mm'] for r in replay]):.3f} mm | payload m {rec['payload_mass'][0]:.3f} com {np.round(rec['payload_com'][0], 4).tolist()} j4 {rec['j4_coul'][0]:.2f}/{rec['j4_visc'][0]:.2f} groups {rec['groups_found']} | {time.time() - t0:.0f} s")
json.dump(dict(scale=SCALE, sg_deriv=deriv_err(data_sg), pinnsr_deriv=deriv_err(ps.data), history=ps.history, results=out), open(f"results/pinnsr_x{SCALE:g}.json", "w"), indent=1)
say("done")
