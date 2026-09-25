"""Physically consistent dense baselines versus SAPC, paired on the same data, within and across engines.

Engines : MuJoCo reference with the injected gap (validation with oracle derivatives, as run_gate.py) and the
          Isaac Sim/PhysX reference (validation with SG derivatives, as run_isaac.py).
Noise   : x1 and x3, three training-noise draws each, fixed in advance and paired with the first three realisations
          of run_stability.py (MuJoCo) and run_isaac_stability.py (PhysX).
Methods : SAPC | dense LS (paper default, ridge 1e-6) | dense with SAPC's ridge and bounds | dense with pseudo-inertia LMIs.
Metrics : linear-library and executable-twin torque NRMSE, replay TCP RMSE, payload mass and CoM, injected friction
          increments, mass moved to non-injected links, projection distances, minimum pseudo-inertia eigenvalue.
Writes results/lmi_compare.json.   Usage: python run_lmi_compare.py
"""
import os
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"
import json, time, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import fr3_twin as ft, identify as idf

ROOT = Path(__file__).resolve().parent
METHODS = {"SAPC": "Forward", "dense LS": "LS-full", "dense, SAPC ridge and bounds": "LS-bounded-r4",
           "dense, pseudo-inertia LMI": "LS-LMI"}
TASKS = [(eng, scale, k) for eng in ("MuJoCo", "PhysX") for scale in (1.0, 3.0) for k in range(3)]


def load(engine):
    nom = ft.make_nominal()
    if engine == "MuJoCo":
        real = ft.make_real()
        train = idf.collect(real, nom, [1, 2, 3], T=20.0, noise_scale=1.0, noise_seed=0)
        val = idf.collect(real, nom, [11, 12], T=20.0, noise_scale=1.0, noise_seed=1)
        return nom, train, val, idf.make_dataset(nom, val, source="oracle")
    from run_isaac import read_log
    manifest = ROOT / "isaac" / "inputs" / "manifest.json"
    h = hashlib.sha256(manifest.read_bytes()).hexdigest()
    data = ROOT / "results" / "isaac"
    train = [read_log(data / f"ref_seed{s}.npz", s, 0.0, h) for s in (1, 2, 3)]
    val = [read_log(data / f"ref_seed{s}.npz", s, 1.0, h) for s in (11, 12)]
    return nom, train, val, idf.make_dataset(nom, val, source="sg", window=101)


def one(task):
    engine, scale, k = task
    nom, train, val, data_va = load(engine)
    if engine == "MuJoCo":                          # run_stability.py realisation k
        rng = np.random.default_rng(1000 + 100 * int(scale) + k)
        for lg in train:
            lg["q_meas"], lg["tau_meas"] = ft.add_sensor_noise(lg, rng, scale=scale)
        logs = train
    else:                                           # run_isaac_stability.py realisation k
        from run_isaac_stability import noisy_training
        logs, _ = noisy_training(train, scale, k, 20260925)
    data_tr = idf.make_dataset(nom, logs, source="sg", window=101)
    rows = []
    for label, meth in METHODS.items():
        t0 = time.time(); xi, info = idf.identify(data_tr, meth, model_nom=nom); fit_s = time.time() - t0
        twin, rep = idf.twin_from(nom, xi)
        if engine == "MuJoCo":
            tcp = float(np.mean([idf.replay_metrics(twin, nom, lg)["tcp_rmse_mm"] for lg in val]))
        else:
            from run_isaac import replay
            tcp = float(np.mean([replay(twin, nom, ref)["tcp_rmse_mm"] for ref in val]))
        rec = idf.recovery_table(nom, xi)
        dm = np.asarray(xi[:70]).reshape(7, 10)[:, 0]
        com = np.asarray(rec["payload_com"][0], float)
        tw = idf.twin_torque_metrics(nom, xi, data_va)
        rows.append(dict(engine=engine, scale=scale, draw=k, method=label, subset=info.get("subset"), fit_s=fit_s,
                         lib_nrmse=idf.torque_metrics(data_va, xi)["nrmse"], twin_nrmse=tw["nrmse"], twin_nrmse_masked=tw["nrmse_masked"],
                         replay_mm=tcp, payload_mass=rec["payload_mass"][0],
                         payload_com_error_mm=float(1e3 * np.linalg.norm(com - np.asarray(ft.GAP_TRUE["payload"]["com"]))),
                         j4_coul=rec["j4_coul"][0], j4_visc=rec["j4_visc"][0], j6_coul=rec["j6_coul"][0],
                         link3_mass_increment=float(dm[2]), other_links_abs_mass=float(np.abs(dm[[0, 1, 3, 4, 5]]).sum()),
                         groups=rec["groups_found"], projected=tw["projected"], min_pseudo_inertia_eig=tw["min_pseudo_inertia_eig"],
                         xi=np.asarray(xi).tolist()))
        print(f"[{engine} x{scale:g} #{k}] {label:30s} payload {rows[-1]['payload_mass']:.4f} kg | CoM err {rows[-1]['payload_com_error_mm']:.1f} mm "
              f"| lib {rows[-1]['lib_nrmse']:.4f} twin {rows[-1]['twin_nrmse']:.4f} | replay {tcp:.4f} mm | other |dm| {rows[-1]['other_links_abs_mass']:.3f} kg", flush=True)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=len(TASKS)) as pool:
        rows = [r for part in pool.map(one, TASKS) for r in part]
    json.dump(dict(identify_version=idf.IDENTIFY_VERSION, rows=rows, wall_s=time.time() - t0),
              open(ROOT / "results" / "lmi_compare.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f} s", flush=True)
