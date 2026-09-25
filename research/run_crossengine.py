"""E6: cross-engine calibration.  Reference = Gazebo Harmonic (DART) FR3 with the injected physics gap (rotor inertia
folded into link inertia, DART joint friction/damping, 1 ms).  Twin = MuJoCo nominal FR3; regressor extracted from
MuJoCo; identified coefficients written back into the MuJoCo model; validation against the Gazebo trajectories.

Columns of the paper figure: nominal | dense calibrated (LS-full) | structure-aware calibrated (Forward) | reference.
Rows: trajectory fidelity (closed-loop replay TCP RMSE vs the Gazebo log), torque fidelity (validation NRMSE),
recovered physical parameters, contact onset / hold force / impulse in the press task.
Also reports the pure engine gap: Gazebo nominal vs MuJoCo nominal on the same controller and reference.

Usage: python run_crossengine.py      (needs results/gz/*.npz from gazebo/collect.sh)
"""
import sys, json, time, numpy as np
import fr3_twin as ft, identify as idf, contact_stage as cs

TRAIN, VAL = [1, 2, 3], [11, 12]
METHODS = ["nominal", "LS-payload", "LS-full", "SINDy-phys", "Forward"]
logf = open("results/crossengine_log.txt", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush(); sys.stdout.flush()


def gz_log(path, seed, noise_scale=1.0, noise_seed=0):
    """Gazebo npz -> identify log dict (the engine's own numerics plus the usual sensor noise model)."""
    G = np.load(path)
    log = dict(t=G["t"], q=G["q"], qd=G["qd"], qdd=np.zeros_like(G["q"]), tau=G["tau"], q_ref=G["q_ref"], seed=seed)
    log["q_meas"], log["tau_meas"] = ft.add_sensor_noise(log, np.random.default_rng(noise_seed + seed), scale=noise_scale)
    return log


ft.DEFAULT_KD = ft.KD_CROSS                      # same controller gains as the Gazebo runs (gazebo/make_traj.py)
nom = ft.make_nominal()
logs_tr = [gz_log(f"results/gz/ref_seed{s}.npz", s) for s in TRAIN]
logs_va = [gz_log(f"results/gz/ref_seed{s}.npz", s) for s in VAL]
say("=== E6 cross-engine: Gazebo/DART reference -> MuJoCo twin ===")

# --- pure engine gap: Gazebo nominal vs MuJoCo nominal, same controller and reference
for s in (1, 11):
    G = np.load(f"results/gz/nom_seed{s}.npz")
    M = ft.Plant(ft.make_nominal()).simulate(ft.Fourier(s), ft.PD(nom), 20.0)
    n = min(len(G["t"]), len(M["t"]))
    p_g, p_m = idf.tcp_positions(nom, G["q"][:n]), idf.tcp_positions(nom, M["q"][:n])
    say(f"engine gap (nominal vs nominal, seed {s}): joint RMSE {np.sqrt(((G['q'][:n] - M['q'][:n]) ** 2).mean(0)).round(5).tolist()} rad, "
        f"TCP RMSE {1e3 * np.sqrt(((p_g - p_m) ** 2).sum(1).mean()):.3f} mm, torque RMSE {np.sqrt(((G['tau'][:n] - M['tau'][:n]) ** 2).mean(0)).round(3).tolist()} Nm")

# --- identification on the Gazebo reference
data_tr = idf.make_dataset(nom, logs_tr, source="sg", window=101)
data_va = idf.make_dataset(nom, logs_va, source="sg", window=101)
gap_gz = dict(ft.GAP_TRUE)               # rotor inertia sits in the reference links; the twin keeps it as armature
out = {}
press_ref = np.load("results/gz/ref_press.npz")
press = cs.Press(); ctrl = ft.PD(nom)
for meth in METHODS:
    t0 = time.time()
    xi, info = idf.identify(data_tr, meth, model_nom=nom)
    if "subset" in info:
        say(f"      {meth} chose {info['subset']} path {info.get('path', '')}")
    tm_va = idf.torque_metrics(data_va, xi)
    twin, rep = idf.twin_from(nom, xi)
    replay = [idf.replay_metrics(twin, nom, lg) for lg in logs_va]
    rec = idf.recovery_table(nom, xi, gap=gap_gz)
    # contact transfer: same press in the MuJoCo twin (nominal MuJoCo plate) vs the Gazebo reference press
    model_cal, extra_cal, _ = idf.instantiate(nom, xi)
    log_c = cs.make_plant(model_cal, cs.PLATE_NOM, extra_cal).simulate(press, ctrl, 4.0)
    ref_c = dict(t=press_ref["t"], F=press_ref["F"], p=idf.tcp_positions(nom, press_ref["q"]) * 0 + np.nan)
    Fg, Ft = press_ref["F"], log_c["F"]; dt = 1e-3
    hold = slice(2200, 2900)
    onset_g = float(press_ref["t"][np.argmax(Fg > 0.05)]) if (Fg > 0.05).any() else None
    onset_t = float(log_c["t"][np.argmax(Ft > 0.05)]) if (Ft > 0.05).any() else None
    contact = dict(onset=onset_t, onset_ref=onset_g, F_hold=float(Ft[hold].mean()), F_hold_ref=float(Fg[hold].mean()),
                   impulse=float(Ft.sum() * dt), impulse_ref=float(Fg.sum() * dt), F_peak=float(Ft.max()), F_peak_ref=float(Fg.max()),
                   F_trace=Ft.tolist())
    out[meth] = dict(val=tm_va, replay=replay, recovery=rec, twin=rep, contact=contact, xi=xi.tolist(),
                     info={k: v for k, v in info.items() if k in ("subset", "path", "thr")}, fit_s=time.time() - t0)
    say(f"[E6] {meth:11s} val NRMSE {tm_va['nrmse']:.3f} R2 {tm_va['r2']:.3f} | replay TCP {np.mean([r['tcp_rmse_mm'] for r in replay]):6.2f} mm "
        f"| payload m {rec['payload_mass'][0]:.3f} com {np.round(rec['payload_com'][0], 4).tolist()} j4 {rec['j4_coul'][0]:.2f}/{rec['j4_visc'][0]:.2f} j6 {rec['j6_coul'][0]:.2f} "
        f"| arm {np.round(rec['armature'][0], 3).tolist()} | groups {rec['groups_found']} | contact onset {onset_t} (ref {onset_g}) hold {contact['F_hold']:.2f} (ref {contact['F_hold_ref']:.2f}) N impulse {contact['impulse']:.2f} (ref {contact['impulse_ref']:.2f}) | {time.time() - t0:.0f} s")
# engine-only contact gap: the MuJoCo 'real' system (identical injected parameters, MuJoCo engine) on the same press
log_r = cs.make_plant(ft.make_real(), cs.PLATE_TRUE).simulate(press, ctrl, 4.0)
Fr, Fg = log_r["F"], press_ref["F"]
out["mujoco_real_same_params"] = dict(onset=float(log_r["t"][np.argmax(Fr > 0.05)]), onset_ref=float(press_ref["t"][np.argmax(Fg > 0.05)]),
                                      F_hold=float(Fr[2200:2900].mean()), F_hold_ref=float(Fg[2200:2900].mean()), impulse=float(Fr.sum() * 1e-3), impulse_ref=float(Fg.sum() * 1e-3), F_trace=Fr.tolist())
say(f"engine-only contact gap (MuJoCo real vs Gazebo ref, same parameters): onset {out['mujoco_real_same_params']['onset']:.3f} vs {out['mujoco_real_same_params']['onset_ref']:.3f}, "
    f"hold {out['mujoco_real_same_params']['F_hold']:.2f} vs {out['mujoco_real_same_params']['F_hold_ref']:.2f} N, impulse {out['mujoco_real_same_params']['impulse']:.2f} vs {out['mujoco_real_same_params']['impulse_ref']:.2f} N s")
out["press_ref"] = dict(t=press_ref["t"].tolist(), F=press_ref["F"].tolist())
json.dump(out, open("results/crossengine_results.json", "w"), indent=1)
say("done")
