"""Isaac/PhysX -> MuJoCo calibration, independent of the existing Gazebo outputs.

Host Python: python run_isaac.py [--data results/isaac] [--decompose]
All results go under --data. Uses existing identification functions unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

import fr3_twin as ft
import identify as idf

ROOT = Path(__file__).resolve().parent
TRAIN, VAL = (1, 2, 3), (11, 12)


def read_log(path, seed=None, noise_scale=1.0, expected_hash=None):
    with np.load(path, allow_pickle=False) as archive:
        log = {k: archive[k] for k in archive.files}
    metadata = json.loads(str(log.pop("metadata")))
    if metadata.get("schema_version") != 2 or not metadata.get("fixed_base") or metadata.get("engine") != "Isaac Sim 4.5.0 / PhysX":
        raise ValueError(f"Not a supported PhysX experiment log: {path}")
    if expected_hash is not None and metadata["manifest_sha256"] != expected_hash:
        raise ValueError(f"Mixed experiment configurations: {path}")
    n = len(log["t"])
    if n < 401 or not np.allclose(log["t"], np.arange(n) * 0.001, rtol=0, atol=1e-10):
        raise ValueError(f"Expected complete, uniformly sampled 1 kHz pre-step states: {path}")
    for key in ("q", "qd", "tau", "q_ref"):
        if log[key].shape != (n, 7) or not np.isfinite(log[key]).all():
            raise ValueError(f"Invalid {key} in {path}")
    if log["F"].shape != (n,) or not np.isfinite(log["F"]).all() or np.min(log["F"]) < 0:
        raise ValueError(f"Invalid contact force in {path}")
    if log["p"].shape != (n, 3) or not np.isfinite(log["p"]).all():
        raise ValueError(f"Invalid PhysX TCP in {path}")
    fk_errors = np.asarray(metadata["fk_error_m"])
    if metadata["dof_names"] != ft.JOINTS or not np.isfinite(fk_errors).all() or len(fk_errors) == 0 or max(fk_errors) > 2e-5:
        raise ValueError(f"Kinematic checks failed: {path}")
    if not np.allclose(log["kp"], ft.DEFAULT_KP) or not np.allclose(log["kd"], ft.KD_CROSS):
        raise ValueError(f"Controller mismatch: {path}")
    if np.any(np.abs(log["tau"]) > log["tau_lim"] + 1e-6):
        raise ValueError(f"Command exceeds torque limits: {path}")
    log["meta"] = metadata
    if seed is not None:
        if metadata["task"] != f"seed{seed}":
            raise ValueError(f"Wrong trajectory seed in {path}")
        if not np.allclose(log["q_ref"], ft.Fourier(seed)(log["t"])[0], atol=1e-10):
            raise ValueError(f"Reference trajectory differs from MuJoCo replay: {path}")
        log["seed"] = seed
        log["qdd"] = np.zeros_like(log["q"])
        log["q_meas"], log["tau_meas"] = ft.add_sensor_noise(log, np.random.default_rng(seed), scale=noise_scale)
    return log


def replay(twin, nominal, reference):
    ctrl = ft.PD(nominal, kp=reference["kp"], kd=reference["kd"])
    log = twin.simulate(ft.Fourier(reference["seed"]), ctrl, len(reference["t"]) * .001)
    if log["q"].shape != reference["q"].shape:
        raise ValueError("Replay and reference lengths differ")
    p = idf.tcp_positions(nominal, log["q"])
    pr = idf.tcp_positions(nominal, reference["q"])
    return dict(tcp_rmse_mm=float(1000*np.sqrt(np.mean(np.sum((p-pr)**2, axis=1)))),
                q_rmse=np.sqrt(np.mean((log["q"]-reference["q"])**2, axis=0)).tolist())


def event_metrics(t, force):
    mask = (t >= 2.2) & (t < 2.9)
    active = np.flatnonzero(force > .05)
    return dict(onset=float(t[active[0]]) if len(active) else None,
                F_hold=float(force[mask].mean()) if mask.any() else None,
                impulse=float(force.sum() * (t[1]-t[0])), F_peak=float(force.max()))


def contact_replay(model, extra, nominal, config, reference, plate_true=False):
    """Same probe and press as E4, without importing its log-writing script.

MuJoCo force samples are aligned with the Isaac log: previous control interval's
mean at pre-step t[k]. Contact parameters are never fitted to PhysX observations.
"""
    p = dict(pos=config["plate_top"], half=.01, r_probe=.01,
             solref=(.002, .6) if plate_true else (.005, 1.),
             solref_probe=(.005, 1.), mu=.3 if plate_true else .6)
    m = ft.load_model(contact=True, plate=p, timestep=float(model.opt.timestep),
                      integrator="implicit" if model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT else "implicitfast")
    for name in ("body_mass", "body_ipos", "body_inertia", "body_iquat"):
        getattr(m, name)[:model.nbody] = getattr(model, name)
    for name in ("dof_frictionloss", "dof_damping", "dof_armature"):
        getattr(m, name)[:] = getattr(model, name)
    mujoco.mj_setConst(m, mujoco.MjData(m))
    plant = ft.Plant(m, extra)
    plant.reset(ft.Q_HOME)
    d = plant.data
    # Derive desired velocity/acceleration from the exact prepared controller input.
    with np.load(config["_inputs"] / "press.npz", allow_pickle=False) as tr:
        qd, vd, ad = tr["q_d"], tr["qd_d"], tr["qdd_d"]
    ctrl = ft.PD(nominal, kp=reference["kp"], kd=reference["kd"])
    n = len(reference["t"])
    if len(qd) != n:
        raise ValueError("Contact trajectory length mismatch")
    Q, F = np.zeros((n, 7)), np.zeros(n)
    acc, last_force = np.zeros(7), 0.0
    nsub = round(.001 / m.opt.timestep)
    for k in range(n):
        Q[k], F[k] = d.qpos.copy(), last_force
        tau = ctrl(d.qpos, d.qvel, qd[k], vd[k], ad[k])
        forces = []
        for _ in range(nsub):
            d.qfrc_applied[:] = tau + plant._extra(d.qpos, d.qvel, acc, tau)
            mujoco.mj_step(m, d)
            acc = d.qacc.copy()
            forces.append(d.sensordata[0])
        last_force = float(np.mean(forces))
    metrics = event_metrics(reference["t"], F)
    metrics.update({k + "_ref": v for k, v in event_metrics(reference["t"], reference["F"]).items()})
    dp = idf.tcp_positions(nominal, Q) - idf.tcp_positions(nominal, reference["q"])
    metrics["tcp_rmse_mm"] = float(1000*np.sqrt(np.mean(np.sum(dp**2, axis=1))))
    return metrics, F


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results" / "isaac")
    parser.add_argument("--inputs", type=Path, default=ROOT / "isaac" / "inputs")
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--methods", nargs="+", choices=["nominal", "LS-payload", "LS-full", "SINDy-phys", "Forward"],
                        default=["nominal", "LS-payload", "LS-full", "SINDy-phys", "Forward"])
    parser.add_argument("--decompose", action="store_true", help="Also fit nominal engine discrepancy and a two-stage model")
    a = parser.parse_args()
    if not np.isfinite(a.noise) or a.noise < 0:
        parser.error("--noise must be finite and nonnegative")
    manifest = a.inputs / "manifest.json"
    config = json.loads(manifest.read_text(encoding="utf-8"))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    for filename, expected in config["source_sha256"].items():
        if hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"MuJoCo source changed since collection: {filename}; use the matching experiment sources")
    config["_inputs"] = a.inputs.resolve()
    def logs(variant, seeds):
        return [read_log(a.data / f"{variant}_seed{s}.npz", s, a.noise, digest) for s in seeds]
    train, val = logs("ref", TRAIN), logs("ref", VAL)
    nominal_val = logs("nom", VAL)
    press = read_log(a.data / "ref_press.npz", expected_hash=digest)
    nominal = ft.make_nominal()
    data = idf.make_dataset(nominal, train, source="sg", window=101)
    data_val = idf.make_dataset(nominal, val, source="sg", window=101)
    out = {"status": "running", "metadata": dict(reference_engine=config["engine"], target_engine="MuJoCo " + mujoco.__version__,
                             manifest_sha256=digest, noise_scale=a.noise, train_seeds=TRAIN, val_seeds=VAL,
                             friction_mode=config["friction_mode"], armature_mode=config["armature_mode"],
                             note="Controlled simulated reference; explicit SI friction, native PhysX rigid-body/contact dynamics.")}
    out["engine_only"] = [replay(ft.Plant(ft.make_nominal()), nominal, r) for r in nominal_val]
    out["same_physical_parameters"] = [replay(ft.Plant(ft.make_real()), nominal, r) for r in val]
    out["press_ref"] = event_metrics(press["t"], press["F"])
    traces = {"t": press["t"], "reference": press["F"]}
    results = {}
    def save():
        out["methods"] = results
        tmp = a.data / "analysis.tmp.json"
        tmp.write_text(json.dumps(clean_json(out), indent=2, allow_nan=False), encoding="utf-8")
        tmp.replace(a.data / "analysis.json")
    for method in a.methods:
        start = time.monotonic()
        xi, info = idf.identify(data, method, model_nom=nominal)
        fit_seconds = time.monotonic() - start
        twin, report = idf.twin_from(nominal, xi)
        contact, forces = contact_replay(twin.model, twin.extra, nominal, config, press)
        traces[method] = forces
        results[method] = dict(val=idf.torque_metrics(data_val, xi),
                               replay=[replay(twin, nominal, r) for r in val],
                               recovery=idf.recovery_table(nominal, xi), contact=contact,
                               xi=xi.tolist(), info=info, twin=report, fit_s=fit_seconds)
        row = results[method]
        print(f"{method:12s} NRMSE {row['val']['nrmse']:.4f}, TCP {np.mean([r['tcp_rmse_mm'] for r in row['replay']]):.3f} mm, "
              f"payload {row['recovery']['payload_mass'][0]:.4f} kg, groups {row['recovery']['groups_found']}", flush=True)
        save()
    out["same_physical_contact"], traces["same_physical_parameters"] = contact_replay(ft.make_real(), None, nominal, config, press, plate_true=True)
    if a.decompose:
        nominal_train = logs("nom", TRAIN)
        nd = idf.make_dataset(nominal, nominal_train, source="sg")
        nv = idf.make_dataset(nominal, nominal_val, source="sg")
        xi1, info1 = idf.identify(nd, "Forward", model_nom=nominal)
        twin1, report1 = idf.twin_from(nominal, xi1)
        out["nominal_calibration"] = dict(info=info1, xi=xi1, twin=report1,
                                          val=idf.torque_metrics(nv, xi1), replay=[replay(twin1, nominal, r) for r in nominal_val])
        # Like the Gazebo two-stage test: carry native fields, disclose omitted external friction.
        base = twin1.model
        d2 = idf.make_dataset(base, train, source="sg")
        v2 = idf.make_dataset(base, val, source="sg")
        xi2, info2 = idf.identify(d2, "Forward", model_nom=base)
        twin2, report2 = idf.twin_from(base, xi2)
        out["two_stage"] = dict(info=info2, xi=xi2, twin=report2, omitted_stage1_friction=report1["fric_applied"],
                                val=idf.torque_metrics(v2, xi2), replay=[replay(twin2, nominal, r) for r in val],
                                note="Stage-2 NRMSE uses the stage-1 baseline denominator; compare absolute rms_all across stages.")
    out["status"] = "complete"
    save()
    np.savez_compressed(a.data / "contact_traces.npz", **traces)
    lines = ["# Isaac/PhysX -> MuJoCo", "", "Saved experiment results; no manuscript files changed.", "",
             "| Method | Torque NRMSE | TCP RMSE [mm] | Payload [kg] | Groups |",
             "|---|---:|---:|---:|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['val']['nrmse']:.4f} | {np.mean([x['tcp_rmse_mm'] for x in r['replay']]):.3f} | "
                     f"{r['recovery']['payload_mass'][0]:.4f} | {', '.join(r['recovery']['groups_found']) or '-'} |")
    lines += ["", f"Nominal engine discrepancy: {np.mean([r['tcp_rmse_mm'] for r in out['engine_only']]):.3f} mm.",
              "Native armature is preserved. Coulomb/viscous friction is an explicit passive torque, not PhysX jointFriction.",
              "Contact uses a rigid PhysX solver, not a translation of MuJoCo solref. No contact parameters are fitted."]
    (a.data / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {a.data / 'analysis.json'}", flush=True)


if __name__ == "__main__":
    main()
