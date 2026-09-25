"""True-parameter reference row for the benchmark tables (Tables 2 and 8).

Rebuilds the validation logs of run_gate.py exactly (seeds 11, 12 x 20 s, measured torque with noise x1, oracle
derivatives) and evaluates the true correction, written back through the same instantiate path as every calibrated
twin: linear-library NRMSE, executable-twin NRMSE, closed-loop replay TCP RMSE and 0.2 s open-loop divergence.
The true correction does not depend on the training noise, so one row serves every noise level.
Also the replay against the Isaac Sim reference (held-out logs of run_lmi_compare.py and Table 7).
Output: results/true_reference.json
"""
import json
import numpy as np
import fr3_twin as ft, identify as idf

VAL, T = [11, 12], 20.0
nom = ft.make_nominal(); real = ft.make_real()
logs_va = idf.collect(real, nom, VAL, T=T, noise_scale=1.0, noise_seed=1)
data_va = idf.make_dataset(nom, logs_va, source="oracle")
xi_true = np.concatenate([ft.gap_true_dpi(nom, ft.GAP_TRUE).ravel(), ft.gap_true_fric(nom, ft.GAP_TRUE).ravel(), np.zeros(7)])
tm_va = idf.torque_metrics(data_va, xi_true)
tw = idf.twin_torque_metrics(nom, xi_true, data_va)
twin, rep = idf.twin_from(nom, xi_true)
replay = [idf.replay_metrics(twin, nom, lg) for lg in logs_va]
openl = [idf.openloop_metrics(twin, nom, lg, window=0.2) for lg in logs_va]
out = dict(identify_version=idf.IDENTIFY_VERSION, lib_nrmse=tm_va["nrmse"], twin_nrmse=tw["nrmse"],
           replay_tcp_mm=float(np.mean([r["tcp_rmse_mm"] for r in replay])),
           openloop_mm=float(np.mean([o["tcp_end_mm_mean"] for o in openl])), twin=rep)
# Isaac Sim reference: the held-out logs of run_lmi_compare.py / Table 7 (seeds 11, 12, noise x1)
import hashlib
from pathlib import Path
from run_isaac import read_log
h = hashlib.sha256(open(Path("isaac") / "inputs" / "manifest.json", "rb").read()).hexdigest()
val_is = [read_log(Path("results") / "isaac" / f"ref_seed{s}.npz", s, 1.0, h) for s in (11, 12)]
out["isaac_replay_tcp_mm"] = float(np.mean([idf.replay_metrics(twin, nom, lg)["tcp_rmse_mm"] for lg in val_is]))
json.dump(out, open("results/true_reference.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "twin"})
