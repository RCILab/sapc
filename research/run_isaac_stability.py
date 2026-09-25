"""Minimal Forward-only PhysX stability study on fixed, already collected logs.

python run_isaac_stability.py
Defaults: reference x1/x3, 10 training-noise draws each; nominal x1, 10 draws.
No PhysX simulation, contact experiment, or other identification method is run.
MuJoCo closed-loop replay is evaluated for each fitted twin on seeds 11 and 12.
"""
import os
# Each independent job uses one BLAS thread; this does not change the estimator.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import fr3_twin as ft
import identify as idf
from run_isaac import read_log, replay, clean_json

ROOT = Path(__file__).resolve().parent
INJECTED = {"link3", "link7", "joint4", "joint6"}
METRICS = ("payload_mass_kg", "com_error_mm", "com_x_m", "com_y_m", "com_z_m",
           "j4_coulomb_delta_Nm", "j4_viscous_delta_Nms", "j6_coulomb_delta_Nm", "j6_viscous_delta_Nms",
           "val_nrmse", "val_rms_Nm", "tcp_rmse_mm", "cv_rmse_Nm", "fit_seconds", "total_seconds")
STATE = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def noisy_training(logs, scale, replicate, seed_base):
    """Independent trajectory streams, paired across reference/nominal variants."""
    output, entropy = [], []
    for source in logs:
        parts = [seed_base, int(scale), replicate, source["seed"]]
        rng = np.random.default_rng(np.random.SeedSequence(parts))
        log = dict(source)
        log["q_meas"], log["tau_meas"] = ft.add_sensor_noise(source, rng, scale=scale)
        output.append(log)
        entropy.append(parts)
    return output, entropy


def initialize(data_dir, manifest_hash):
    global STATE
    threadpool_limits(limits=1)
    nominal = ft.make_nominal()
    STATE = {"model": nominal, "variants": {}}
    for variant in ("ref", "nom"):
        # Validation noise stays exactly as in run_isaac.py: x1, seed=trajectory seed.
        train = [read_log(Path(data_dir) / f"{variant}_seed{s}.npz", s, 0., manifest_hash) for s in (1, 2, 3)]
        val = [read_log(Path(data_dir) / f"{variant}_seed{s}.npz", s, 1., manifest_hash) for s in (11, 12)]
        STATE["variants"][variant] = dict(train=train, val=val,
                                          validation=idf.make_dataset(nominal, val, source="sg", window=101))


def one_run(task):
    case, variant, scale, replicate, seed_base = task
    start = time.perf_counter()
    nominal = STATE["model"]
    source = STATE["variants"][variant]
    logs, entropy = noisy_training(source["train"], scale, replicate, seed_base)
    data = idf.make_dataset(nominal, logs, source="sg", window=101)
    fit_start = time.perf_counter()
    xi, info = idf.identify(data, "Forward", model_nom=nominal)
    fit_seconds = time.perf_counter() - fit_start
    twin, report = idf.twin_from(nominal, xi)
    validation = idf.torque_metrics(source["validation"], xi)
    replay_rows = [replay(twin, nominal, ref) for ref in source["val"]]
    mass = float(xi[60])
    com = xi[61:64] / mass - nominal.body_pos[nominal.body("payload").id] if abs(mass) > 1e-6 else np.full(3, np.nan)
    com_error = float(1000*np.linalg.norm(com - ft.GAP_TRUE["payload"]["com"])) if variant == "ref" else None
    groups = info["subset"]
    row = dict(case=case, variant=variant, noise_scale=scale, replicate=replicate,
               training_noise_seed_entropy=entropy, groups=groups,
               exact_injected_support=(set(groups) == INJECTED) if variant == "ref" else None,
               injected_groups_selected=sorted(set(groups) & INJECTED),
               payload_mass_kg=mass, com_error_mm=com_error,
               com_x_m=float(com[0]), com_y_m=float(com[1]), com_z_m=float(com[2]),
               j4_coulomb_delta_Nm=float(xi[85]), j4_viscous_delta_Nms=float(xi[86]),
               j6_coulomb_delta_Nm=float(xi[95]), j6_viscous_delta_Nms=float(xi[96]),
               val_nrmse=validation["nrmse"], val_rms_Nm=validation["rms_all"],
               tcp_rmse_mm=float(np.mean([r["tcp_rmse_mm"] for r in replay_rows])),
               cv_rmse_Nm=info["score"], cv_se_Nm=info["se"],
               fit_seconds=fit_seconds, total_seconds=time.perf_counter()-start,
               xi=xi.tolist(), selection_path=info["path"], validation=validation,
               replay=replay_rows, native_writeback=report)
    numbers = [row[k] for k in ("payload_mass_kg", "val_nrmse", "val_rms_Nm", "tcp_rmse_mm")]
    if not np.isfinite(numbers).all() or (variant == "ref" and not np.isfinite(com_error)):
        raise ValueError(f"Invalid fitted-model evaluation: {case}/{replicate}")
    return clean_json(row)


def stats(values):
    valid = np.array([v for v in values if v is not None and np.isfinite(v)], float)
    if len(valid) == 0:
        return dict(n_valid=0, mean=None, std=None, min=None, max=None, p5=None, p95=None)
    return dict(n_valid=len(valid), mean=float(valid.mean()),
                std=float(valid.std(ddof=1)) if len(valid) > 1 else 0.,
                min=float(valid.min()), max=float(valid.max()),
                p5=float(np.percentile(valid, 5)), p95=float(np.percentile(valid, 95)))


def summarize(rows):
    n = len(rows)
    counts = {g: sum(g in row["groups"] for row in rows) for g in idf.GROUPS}
    return dict(n=n, group_counts=counts, group_frequency={g: v/n for g, v in counts.items()},
                support_patterns={" + ".join(groups): sum(tuple(sorted(r["groups"])) == groups for r in rows)
                                  for groups in sorted({tuple(sorted(r["groups"])) for r in rows})},
                exact_support_count=sum(r["exact_injected_support"] is True for r in rows),
                any_injected_group_count=sum(bool(r["injected_groups_selected"]) for r in rows),
                empty_support_count=sum(not r["groups"] for r in rows),
                metrics={key: stats([r[key] for r in rows]) for key in METRICS})


def write_json(path, value):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_reports(out, result):
    rows = [row for case in result["cases"].values() for row in case["runs"]]
    with (out / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case", "replicate", "groups", *METRICS])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in ["case", "replicate", *METRICS]}, "groups": ";".join(row["groups"])})
    lines = ["# PhysX → MuJoCo: Forward stability and nominal negative control", "",
             f"Status: {result['status']}. Only Forward was fitted. No PhysX trajectories were rerun.", "",
             "## Protocol", "",
             "- Fixed training trajectories: seeds 1, 2, 3; fixed held-out trajectories: 11, 12 (20 s each).",
             "- Only training measurement noise is redrawn; validation measurement noise remains fixed at ×1, exactly as in run_isaac.py.",
             "- SG101 derivatives, original Forward + bounded refit + native write-back, with no estimator changes.",
             "- Paired noise streams for reference and nominal controls; exact RNG entropy and all 112 coefficients are saved per run.",
             "- TCP RMSE averages the two held-out closed-loop MuJoCo replays against the stored PhysX states.",
             "- Standard deviations use ddof=1. These distributions are conditional on fixed excitation trajectories, not robot/trajectory resampling.",
             "", "## Group selection", "",
             "| Group | Reference ×1 | Reference ×3 | Nominal ×1 |", "|---|---:|---:|---:|"]
    def count(case, group):
        s = result["cases"].get(case, {}).get("summary")
        return f"{s['group_counts'][group]}/{s['n']}" if s else "—"
    for group in idf.GROUPS:
        lines.append(f"| {group} | {count('ref_x1', group)} | {count('ref_x3', group)} | {count('nom_x1', group)} |")
    lines += ["", "## Metric distributions (mean ± sample standard deviation)", "",
              "| Metric | Reference ×1 | Reference ×3 | Nominal ×1 |", "|---|---:|---:|---:|"]
    def cell(case, key):
        s = result["cases"].get(case, {}).get("summary")
        if not s or s["metrics"][key]["n_valid"] == 0:
            return "N/A"
        s = s["metrics"][key]
        return f"{s['mean']:.6g} ± {s['std']:.3g}"
    for key in METRICS:
        lines.append(f"| {key} | {cell('ref_x1', key)} | {cell('ref_x3', key)} | {cell('nom_x1', key)} |")
    lines += ["", "## Ground truth and interpretation", "",
              "Reference payload is 0.8 kg at (0.02, −0.015, 0.06) m. Friction estimates above are changes relative to the nominal model, not total friction.",
              "Reference joint-4 Coulomb/viscous changes: +1.663 Nm / +0.34 Nm s; joint-6: +0.66 Nm / 0 Nm s.",
              "Nominal control has zero physical changes and no payload: its payload CoM error is undefined, not zero.",
              "", "The existing Forward implementation does not evaluate empty support and must select at least one group. No empty-support claim is possible from this unchanged estimator.",
              "Any selection of link3/link7/joint4/joint6 in the nominal control is reported without thresholding, even if its coefficient is small.",
              "A nominal control may still contain engine/model-form discrepancy. Selection frequency and coefficient magnitude must be interpreted together.",
              "PhysX uses preserved armature and explicit tanh/viscous friction, as documented in isaac/README.md. This study alone does not prove universal engine-independent identifiability."]
    if result["status"] == "complete":
        lines += ["", "## Observed support patterns", ""]
        for name, case in result["cases"].items():
            s = case["summary"]
            lines.append(f"- {name}: {s['support_patterns']}; injected-group selection in {s['any_injected_group_count']}/{s['n']} runs.")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "results/isaac")
    p.add_argument("--out", type=Path, default=ROOT / "results/isaac_stability")
    p.add_argument("--n-x1", type=int, default=10)
    p.add_argument("--n-x3", type=int, default=10)
    p.add_argument("--n-control", type=int, default=10)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed-base", type=int, default=20260925)
    a = p.parse_args()
    if min(a.n_x1, a.n_x3, a.n_control, a.seed_base) < 0 or a.workers < 1:
        p.error("Counts/seed must be nonnegative; workers must be positive")
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Choose a new --out directory; existing stability results are not overwritten")
    manifest = ROOT / "isaac/inputs/manifest.json"
    config = json.loads(manifest.read_text(encoding="utf-8"))
    for path, expected in config["source_sha256"].items():
        if digest(ROOT / path) != expected:
            raise ValueError(f"Original model source changed: {path}")
    cases = [("ref_x1", "ref", 1., a.n_x1), ("ref_x3", "ref", 3., a.n_x3), ("nom_x1", "nom", 1., a.n_control)]
    tasks = [(case, variant, scale, k, a.seed_base) for case, variant, scale, n in cases for k in range(n)]
    protected = {str(path.relative_to(ROOT)): digest(path)
                 for path in ROOT.rglob("*") if path.is_file() and
                 ("paper" in path.relative_to(ROOT).parts or path.suffix in (".py", ".md"))}
    input_hashes = {f"{v}_seed{s}.npz": digest(a.data / f"{v}_seed{s}.npz") for v in ("nom", "ref") for s in (1, 2, 3, 11, 12)}
    a.out.mkdir(parents=True, exist_ok=True)
    result = dict(status="running", metadata=dict(method="Forward", train_seeds=[1, 2, 3], validation_seeds=[11, 12],
                                                  validation_noise="fixed x1, RNG=trajectory seed, original Isaac benchmark",
                                                  training_noise_seed_base=a.seed_base, workers=a.workers,
                                                  manifest_sha256=digest(manifest), input_sha256=input_hashes,
                                                  source_sha256={name: digest(ROOT / name) for name in ("identify.py", "fr3_twin.py", "run_isaac.py", "run_isaac_stability.py")},
                                                  plans={c: dict(variant=v, scale=s, n=n) for c,v,s,n in cases},
                                                  empty_support_candidate=bool(getattr(idf, "FORWARD_ALLOW_EMPTY", False)),
                                                  identify_version=getattr(idf, "IDENTIFY_VERSION", None)),
                  cases={case: dict(runs=[]) for case, _, _, n in cases if n})
    write_json(a.out / "stability.json", result)
    start = time.perf_counter()
    try:
        with ProcessPoolExecutor(max_workers=a.workers, initializer=initialize,
                                 initargs=(str(a.data.resolve()), digest(manifest))) as pool:
            pending = {pool.submit(one_run, task): task for task in tasks}
            for number, future in enumerate(as_completed(pending), 1):
                row = future.result()
                case = result["cases"][row["case"]]
                case["runs"].append(row)
                case["runs"].sort(key=lambda r: r["replicate"])
                case["summary"] = summarize(case["runs"])
                write_json(a.out / f"{row['case']}_{row['replicate']:02d}.json", row)
                write_json(a.out / "stability.json", result)
                print(f"[{number}/{len(tasks)}] {row['case']} #{row['replicate']:02d}: {row['groups']} | "
                      f"mass {row['payload_mass_kg']:.5f} kg | CoM {row['com_error_mm']} mm | "
                      f"NRMSE {row['val_nrmse']:.5f} | TCP {row['tcp_rmse_mm']:.4f} mm | {row['total_seconds']:.1f}s", flush=True)
        changed = [name for name, value in protected.items() if digest(ROOT / name) != value]
        changed += [name for name, value in input_hashes.items() if digest(a.data / name) != value]
        if changed:
            raise RuntimeError(f"Protected inputs/source/paper changed during study: {changed}")
        result["status"] = "complete"
        result["wall_seconds"] = time.perf_counter()-start
        result["protected_files_unchanged"] = True
        result["protected_file_count"] = len(protected)
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        write_json(a.out / "stability.json", result)
        raise
    write_json(a.out / "stability.json", result)
    write_reports(a.out, result)
    print(f"COMPLETE: {len(tasks)} Forward fits, {result['wall_seconds']:.1f}s. Results: {a.out}", flush=True)


if __name__ == "__main__":
    main()
