"""Same-engine benchmark E1 (oracle derivatives) and E2 (noisy measurements, Savitzky-Golay derivatives), final protocol.

Real system R = FR3 + GAP_TRUE (implicit 0.5 ms).  Nominal twin T0 = Menagerie FR3, implicitfast 1 ms.
Train: Fourier seeds 1,2,3 x 20 s.  Validation: seeds 11,12 x 20 s, oracle derivatives, measured torque (noise x1).
Library: 112 columns (70 inertial, 35 friction, 7 armature) in 15 groups (identify.GROUPS).
Noise (E2): one generator per condition, default_rng(int(10*scale) + 7), drawn sequentially over the three training
trajectories, so every trajectory receives an independent noise stream.
Methods: see METHODS.  Every calibrated twin is written back natively (identify.instantiate) and evaluated by the
linear-library torque NRMSE, the executable-twin torque NRMSE, closed-loop replay and 0.2 s open-loop divergence.

Usage: python run_gate.py e1|x0|x1|x3   (one condition -> results/gate_parts/<cond>.json)
       python run_gate.py merge          (-> results/gate_results.json)
"""
import sys, os, json, time
import numpy as np
import fr3_twin as ft, identify as idf, learned

TRAIN, VAL, T = [1, 2, 3], [11, 12], 20.0
METHODS = ["nominal", "LS-payload", "LS-struct", "LS-full", "LS-bounded-r4", "LS-LMI", "LS-phys", "SINDy", "SINDy-phys",
           "SINDy-group", "Forward", "BestSubset", "MLP"]
CONDS = {"e1": ("E1", "oracle", None), "x0": ("E2 noise x0.0 SG101", "sg", 0.0), "x1": ("E2 noise x1.0 SG101", "sg", 1.0),
         "x3": ("E2 noise x3.0 SG101", "sg", 3.0)}
MODE = sys.argv[1] if len(sys.argv) > 1 else "x1"
os.makedirs("results/gate_parts", exist_ok=True)


def merge():
    out = {}
    for key in ("e1", "x0", "x1", "x3"):
        part = json.load(open(f"results/gate_parts/{key}.json"))
        out[CONDS[key][0]] = part["results"]
        out.setdefault("_meta", {})[CONDS[key][0]] = part["meta"]
    json.dump(out, open("results/gate_results.json", "w"), indent=1)
    print("merged -> results/gate_results.json")


if MODE == "merge":
    merge(); sys.exit(0)

tag, source, scale = CONDS[MODE]
logf = open(f"results/gate_parts/{MODE}_log.txt", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush(); sys.stdout.flush()


nom = ft.make_nominal(); real = ft.make_real()
t0 = time.time()
logs_tr = idf.collect(real, nom, TRAIN, T=T, noise_scale=1.0, noise_seed=0)
logs_va = idf.collect(real, nom, VAL, T=T, noise_scale=1.0, noise_seed=1)
say(f"[{tag}] collected {len(TRAIN)} train + {len(VAL)} val trajectories, {time.time() - t0:.0f} s")
data_va = idf.make_dataset(nom, logs_va, source="oracle")
if source == "sg":
    rng = np.random.default_rng(int(10 * scale) + 7)
    for lg in logs_tr:
        lg["q_meas"], lg["tau_meas"] = ft.add_sensor_noise(lg, rng, scale=scale)
data_tr = idf.make_dataset(nom, logs_tr, source=source, window=101)

res = {}
for meth in METHODS:
    t0 = time.time()
    if meth == "MLP":
        mdl = learned.MLPResidual(nom).fit(data_tr)
        fit_s = time.time() - t0
        tm_tr, tm_va = mdl.torque_metrics(data_tr), mdl.torque_metrics(data_va)
        twin = ft.Plant(ft.make_nominal(), mdl.extra_fn()); rep = {"learned": True}; rec = {}; xi = None; info = {}; tw = None
    else:
        xi, info = idf.identify(data_tr, meth, model_nom=nom)
        fit_s = time.time() - t0
        tm_tr, tm_va = idf.torque_metrics(data_tr, xi), idf.torque_metrics(data_va, xi)
        tw = idf.twin_torque_metrics(nom, xi, data_va)
        twin, rep = idf.twin_from(nom, xi)
        rec = idf.recovery_table(nom, xi)
    replay =[idf.replay_metrics(twin, nom, lg) for lg in logs_va]
    openl = [idf.openloop_metrics(twin, nom, lg, window=0.2) for lg in logs_va]
    keep = {k: v for k, v in info.items() if k in ("subset", "path", "top", "score", "se", "n_subsets", "thr")}
    res[meth] = dict(train=tm_tr, val=tm_va, val_twin=tw, recovery=rec, twin=rep, replay=replay, openloop=openl,
                     xi=None if xi is None else xi.tolist(), fit_s=fit_s, info=keep)
    tw_txt = "-" if tw is None else f"{tw['nrmse']:.4f}"
    say(f"[{tag}] {meth:13s} {fit_s:6.1f} s | lib NRMSE {tm_va['nrmse']:.4f} | twin NRMSE {tw_txt} "
        f"| replay {np.mean([r['tcp_rmse_mm'] for r in replay]):.4f} mm | open-loop {np.mean([o['tcp_end_mm_mean'] for o in openl]):.3f} mm "
        f"| groups {rec.get('groups_found', '-')} | payload {rec['payload_mass'][0] if rec else float('nan'):.4f}")
    json.dump(dict(meta=dict(tag=tag, source=source, scale=scale, identify_version=idf.IDENTIFY_VERSION,
                             allow_empty=idf.FORWARD_ALLOW_EMPTY, noise="default_rng(int(10*scale)+7), sequential over trajectories"),
                   results=res), open(f"results/gate_parts/{MODE}.json", "w"), indent=1)
say(f"[{tag}] done")
