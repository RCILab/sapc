"""Consequence of attribution: edit each calibrated twin as a maintenance engineer would, then compare it with the
reference that has undergone the same physical change.  Same data and noise realisations as run_gate.py (E2).

Edits   : (a) remove the identified payload (the link-7 increment)  -> reference without the payload;
          (b) replace it by a new, known tool                       -> reference carrying that tool.
Transfer: unedited twins on faster and larger held-out motions than the identification data (no edit).
Twins   : nominal | true parameters | SAPC | dense LS (paper default) | dense with SAPC's ridge and bounds
          | dense with pseudo-inertia LMIs | dense, raw-unit minimum norm.
Writes results/counterfactual.json.   Usage: python run_counterfactual.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
import sys, json, time, copy
import numpy as np
import fr3_twin as ft, identify as idf

TRAIN, VAL, T = [1, 2, 3], [11, 12], 20.0
NEW_TOOL = dict(mass=0.5, com=(-0.03, 0.02, 0.04), inertia=(1.0e-3, 1.0e-3, 5.0e-4))
EXTRAP = dict(amp=0.45, vmax=2.5, amax=12.0)          # identification: amp 0.35, vmax 1.6 rad/s, amax 8 rad/s^2
logf = open("results/counterfactual_log.txt", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush(); sys.stdout.flush()


nom = ft.make_nominal(); real = ft.make_real()
p0 = nom.body_pos[nom.body("payload").id]
pi_tool = ft.pi_from_params(NEW_TOOL["mass"], p0 + np.asarray(NEW_TOOL["com"]), NEW_TOOL["inertia"])
gap_remove = {k: v for k, v in ft.GAP_TRUE.items() if k != "payload"}
gap_swap = dict(ft.GAP_TRUE); gap_swap["payload"] = NEW_TOOL
xi_true = np.concatenate([ft.gap_true_dpi(nom, ft.GAP_TRUE).ravel(), ft.gap_true_fric(nom, ft.GAP_TRUE).ravel(), np.zeros(7)])

refs = {"remove": idf.collect(ft.make_real(gap=gap_remove), nom, VAL, T=T, noise_scale=0.0),
        "swap": idf.collect(ft.make_real(gap=gap_swap), nom, VAL, T=T, noise_scale=0.0)}
trajs, ex_seeds = [], []
for sd in range(21, 60):
    try:
        trajs.append(ft.Fourier(sd, **EXTRAP)); ex_seeds.append(sd)
    except AssertionError:
        continue
    if len(trajs) == 2:
        break
refs_ex = [ft.Plant(copy.copy(real)).simulate(tr, ft.PD(nom), T) for tr in trajs]


def edit(xi, kind):
    x = np.array(xi, float).copy()
    x[60:70] = 0.0 if kind == "remove" else pi_tool
    return x


def tcp_rmse(la, lb):
    k = min(len(la["q"]), len(lb["q"]))
    pa, pb = idf.tcp_positions(nom, la["q"][:k]), idf.tcp_positions(nom, lb["q"][:k])
    return float(1e3 * np.sqrt(((pa - pb) ** 2).sum(axis=1).mean()))


def evaluate(xi, nominal_edit=False):
    row = {}
    for kind in ("remove", "swap"):
        x = edit(np.zeros(112), kind) if nominal_edit else edit(xi, kind)
        tw, _ = idf.twin_from(nom, x)
        row[kind] = float(np.mean([idf.replay_metrics(tw, nom, lg)["tcp_rmse_mm"] for lg in refs[kind]]))
    tw, _ = idf.twin_from(nom, np.zeros(112) if nominal_edit else xi)
    row["extrapolation"] = float(np.mean([tcp_rmse(tw.simulate(tr, ft.PD(nom), T), lr) for tr, lr in zip(trajs, refs_ex)]))
    return row


def masses(xi):
    dm = np.asarray(xi[:70]).reshape(7, 10)[:, 0]
    return dict(link7_mass_increment=float(dm[6]), other_links_abs_mass=float(np.abs(dm[:6]).sum()),
                link_mass_increments=dm.tolist())


out = dict(meta=dict(identify_version=idf.IDENTIFY_VERSION, new_tool=NEW_TOOL, extrapolation=EXTRAP,
                     extrapolation_seeds=ex_seeds, noise="as run_gate.py E2"))
out["nominal"] = evaluate(None, nominal_edit=True); out["nominal"]["note"] = "nominal twin; the edits add or remove only the known tool"
out["true"] = evaluate(xi_true); out["true"].update(masses(xi_true))
say("nominal", out["nominal"]); say("true", out["true"])
logs_tr = idf.collect(real, nom, TRAIN, T=T, noise_scale=1.0, noise_seed=0)
for scale in (0.0, 1.0, 3.0):
    rng = np.random.default_rng(int(10 * scale) + 7)
    for lg in logs_tr:
        lg["q_meas"], lg["tau_meas"] = ft.add_sensor_noise(lg, rng, scale=scale)
    data_tr = idf.make_dataset(nom, logs_tr, source="sg", window=101)
    A, y = idf.stack(data_tr)
    fits = {"SAPC": idf.identify(data_tr, "Forward", model_nom=nom)[0],
            "dense LS": idf.identify(data_tr, "LS-full", model_nom=nom)[0],
            "dense, SAPC ridge and bounds": idf.identify(data_tr, "LS-bounded-r4", model_nom=nom)[0],
            "dense, pseudo-inertia LMI": idf.identify(data_tr, "LS-LMI", model_nom=nom)[0],
            "dense, raw-unit minimum norm": np.linalg.lstsq(A, y, rcond=None)[0]}
    out[f"x{scale:g}"] = {}
    for name, xi in fits.items():
        t0 = time.time()
        row = evaluate(xi); row.update(masses(xi)); row["xi"] = np.asarray(xi).tolist()
        out[f"x{scale:g}"][name] = row
        say(f"[x{scale:g}] {name:30s} remove {row['remove']:.4f} mm | swap {row['swap']:.4f} mm | extrapolation {row['extrapolation']:.4f} mm "
            f"| link7 dm {row['link7_mass_increment']:.4f} kg | other |dm| {row['other_links_abs_mass']:.4f} kg | {time.time() - t0:.0f} s")
        json.dump(out, open("results/counterfactual.json", "w"), indent=1)
say("done")
