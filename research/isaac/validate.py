"""Audit real PhysX logs against input hashes, FK, torque accounting and contact events."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fr3_twin as ft
import identify as idf
from run_isaac import read_log, event_metrics


def validate(data, inputs):
    manifest = inputs / "manifest.json"
    config = json.loads(manifest.read_text(encoding="utf-8"))
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    model = ft.make_nominal()
    report = {}
    files = sorted(list(data.glob("nom_*.npz")) + list(data.glob("ref_*.npz")))
    if not files:
        raise ValueError("No PhysX rollouts to validate")
    for path in files:
        variant, task = path.stem.split("_", 1)
        seed = int(task[4:]) if task.startswith("seed") else None
        log = read_log(path, seed=seed, expected_hash=digest)
        if log["meta"]["variant"] != variant or log["meta"]["task"] != task:
            raise ValueError(f"File/metadata mismatch: {path}")
        source = config["trajectories"][task]
        if log["meta"]["trajectory_sha256"] != source["sha256"]:
            raise ValueError(f"Input hash mismatch: {path}")
        with np.load(inputs / source["file"], allow_pickle=False) as tr:
            if len(log["t"]) != len(tr["t"]) or not np.allclose(log["q"][0], tr["q_d"][0], atol=1e-6):
                raise ValueError(f"Initial state or sample count mismatch: {path}")
            command = np.clip(tr["tau_ff"] + log["kp"]*(tr["q_d"]-log["q"]) + log["kd"]*(tr["qd_d"]-log["qd"]), -tr["tau_lim"], tr["tau_lim"])
        if not np.allclose(command, log["tau"], atol=1e-10):
            raise ValueError(f"Controller torque accounting failed: {path}")
        params = config["nominal" if variant == "nom" else "reference"]
        coul = np.array([j["coulomb"] for j in params["joints"]])
        visc = np.array([j["viscous"] for j in params["joints"]])
        passive = coul*np.tanh(log["qd"]/config["friction_velocity"]) + visc*log["qd"]
        if not np.allclose(passive, log["tau_passive"], atol=1e-10) or not np.allclose(log["tau_net"], log["tau"]-passive):
            raise ValueError(f"Passive torque accounting failed: {path}")
        fk = idf.tcp_positions(model, log["q"])
        error = float(np.max(np.linalg.norm(fk-log["p"], axis=1)))
        if error > 2e-5:
            raise ValueError(f"Dynamic FK mismatch {error} m: {path}")
        row = dict(samples=len(log["t"]), fk_max_m=error, tracking_max_rad=float(np.max(np.abs(log["q"]-log["q_ref"]))))
        if task == "press":
            row["contact"] = event_metrics(log["t"], log["F"])
            # Confirms that the contact sensor works, and there was no initial collision.
            if row["contact"]["onset"] is None or row["contact"]["onset"] < 1.0:
                raise ValueError(f"Missing or premature press contact: {path}")
        elif np.any(log["F"] != 0):
            raise ValueError(f"Unexpected free-space contact: {path}")
        report[path.name] = row
        print(f"PASS {path.name}: FK {error*1e6:.2f} um, {len(log['t'])} samples", flush=True)
    (data / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "results" / "isaac")
    p.add_argument("--inputs", type=Path, default=ROOT / "isaac" / "inputs")
    a = p.parse_args()
    validate(a.data, a.inputs)
