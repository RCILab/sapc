"""The one figure of the paper: nominal | dense calibrated | structure-aware calibrated | reference,
rows = same-engine (MuJoCo -> MuJoCo, ground truth known), cross-engine (Gazebo/DART -> MuJoCo) and cross-engine
(Isaac Sim/PhysX -> MuJoCo).
Panels per row: trajectory fidelity (replay TCP RMSE), torque fidelity (validation NRMSE), recovered physics
(payload mass, CoM z, joint-4 Coulomb), contact prediction (onset, hold force, impulse relative to the reference).
Reads results/gate_results.json, results/contact_BestSubset.json, results/contact_LS-full.json, results/crossengine_results.json,
results/isaac/analysis.json.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_NOM, C_DENSE, C_SPARSE, INK, INK2, GRID = "#eb6834", "#1baf7a", "#2a78d6", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 8.5})
COLS = [("nominal", "nominal", C_NOM), ("dense", "dense LS", C_DENSE), ("sparse", "SAPC", C_SPARSE)]
LEGEND = ["nominal twin", "dense LS (all groups)", "SAPC (group selection)"]
TRUE = dict(payload_mass=0.8, com_z=0.06, j4_coul=1.663)


def same_engine():
    R = json.load(open("results/gate_results.json"))["E2 noise x1.0 SG101"]
    m = {"nominal": R["nominal"], "dense": R["LS-full"], "sparse": R["Forward"]}
    row = {}
    for k, r in m.items():
        rec = r["recovery"] or {}
        row[k] = dict(tcp=np.mean([x["tcp_rmse_mm"] for x in r["replay"]]), nrmse=r["val"]["nrmse"],
                      payload_mass=rec.get("payload_mass", [0.0])[0], com_z=(rec.get("payload_com", [[np.nan] * 3])[0][2]),
                      j4_coul=rec.get("j4_coul", [0.0])[0])
    cfiles = {"nominal": ("results/contact_Forward.json", "nominal arm + nominal contact"),
              "dense": ("results/contact_LS-full.json", "calibrated arm + nominal contact"),
              "sparse": ("results/contact_Forward.json", "calibrated arm + nominal contact")}
    for k, (f, key) in cfiles.items():
        if os.path.exists(f):
            c = json.load(open(f))[key]
            row[k].update(onset=c["onset"], onset_ref=c["onset_ref"], hold=c["F_hold"], hold_ref=c["F_hold_ref"], impulse=c["impulse"], impulse_ref=c["impulse_ref"])
    return row


def cross_engine():
    if not os.path.exists("results/crossengine_results.json"):
        return None
    R = json.load(open("results/crossengine_results.json"))
    m = {"nominal": R["nominal"], "dense": R["LS-full"], "sparse": R["Forward"]}
    row = {}
    for k, r in m.items():
        rec = r["recovery"]; c = r["contact"]
        row[k] = dict(tcp=np.mean([x["tcp_rmse_mm"] for x in r["replay"]]), nrmse=r["val"]["nrmse"],
                      payload_mass=rec["payload_mass"][0], com_z=rec["payload_com"][0][2], j4_coul=rec["j4_coul"][0],
                      onset=c["onset"], onset_ref=c["onset_ref"], hold=c["F_hold"], hold_ref=c["F_hold_ref"], impulse=c["impulse"], impulse_ref=c["impulse_ref"])
    if "mujoco_real_same_params" in R:
        row["engine_only"] = R["mujoco_real_same_params"]
    return row


def isaac_engine():
    """Isaac Sim / PhysX reference (results/isaac/analysis.json, written by run_isaac.py): same panels as the Gazebo row."""
    if not os.path.exists("results/isaac/analysis.json"):
        return None
    A = json.load(open("results/isaac/analysis.json"))
    m = {"nominal": A["methods"]["nominal"], "dense": A["methods"]["LS-full"], "sparse": A["methods"]["Forward"]}
    row = {}
    for k, r in m.items():
        rec = r["recovery"]; c = r["contact"]
        com_z = rec["payload_com"][0][2]
        row[k] = dict(tcp=np.mean([x["tcp_rmse_mm"] for x in r["replay"]]), nrmse=r["val"]["nrmse"],
                      payload_mass=rec["payload_mass"][0], com_z=np.nan if com_z is None else com_z, j4_coul=rec["j4_coul"][0],
                      onset=c["onset"], onset_ref=c["onset_ref"], hold=c["F_hold"], hold_ref=c["F_hold_ref"], impulse=c["impulse"], impulse_ref=c["impulse_ref"])
    if "same_physical_contact" in A:                      # MuJoCo engine with the true parameters vs the PhysX press
        row["engine_only"] = A["same_physical_contact"]
    return row


def draw_row(axes, row, title, oracle_tcp=None):
    x = np.arange(3)
    # A trajectory fidelity
    ax = axes[0]
    vals = [row[k]["tcp"] for k, _, _ in COLS]
    ax.bar(x, vals, color=[c for _, _, c in COLS], width=0.6, linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.25, f"{v:.3f}" if v < 0.1 else f"{v:.2f}", ha="center", fontsize=7, color=INK)
    if oracle_tcp:
        ax.axhline(oracle_tcp, color=INK2, lw=1, ls=":", label="true parameters"); ax.legend(frameon=False, fontsize=6, loc="upper right")
    ax.set_yscale("log"); ax.set_ylim(0.01, 60); ax.set_ylabel("replay TCP RMSE [mm]"); ax.set_title("trajectory fidelity")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in COLS], fontsize=7); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    # B torque fidelity
    ax = axes[1]
    vals = [row[k]["nrmse"] for k, _, _ in COLS]
    ax.bar(x, vals, color=[c for _, _, c in COLS], width=0.6, linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.25, f"{v:.3f}", ha="center", fontsize=7, color=INK)
    ax.set_yscale("log"); ax.set_ylim(0.01, 3); ax.set_ylabel("validation torque NRMSE"); ax.set_title("torque fidelity")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in COLS], fontsize=7); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    # C recovered physics (normalised to truth)
    ax = axes[2]
    keys = [("payload_mass", "payload\nmass"), ("com_z", "payload\nCoM z"), ("j4_coul", "joint-4\nCoulomb")]
    w = 0.25
    for i, (k, lab, c) in enumerate(COLS):
        vals = [row[k][kk] / TRUE[kk] for kk, _ in keys]
        ax.bar(np.arange(3) + (i - 1) * w, np.clip(vals, -0.6, 1.8), width=w - 0.03, color=c, linewidth=0)
        for j, v in enumerate(vals):
            if v > 1.8:
                ax.text(j + (i - 1) * w, 1.62, f"{v:.1f}", ha="center", fontsize=6, color=INK, rotation=90)
    ax.axhline(1.0, color=INK, lw=1, ls="--", label="true value"); ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_ylim(-0.6, 1.8); ax.set_ylabel("estimate / true"); ax.set_title("recovered physical parameters")
    ax.set_xticks(np.arange(3)); ax.set_xticklabels([l for _, l in keys], fontsize=7); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    # D contact prediction (relative to reference)
    ax = axes[3]
    if "onset" in row["nominal"]:
        keys = [("onset", "contact\nonset"), ("hold", "hold\nforce"), ("impulse", "impulse")]
        for i, (k, lab, c) in enumerate(COLS):
            vals = [(row[k][kk] or np.nan) / row[k][kk + "_ref"] for kk, _ in keys]
            ax.bar(np.arange(3) + (i - 1) * w, vals, width=w - 0.03, color=c, linewidth=0)
        ax.axhline(1.0, color=INK, lw=1, ls="--", label="reference")
        if "engine_only" in row:                        # same parameters, other engine: the engine-level residual
            e = row["engine_only"]
            ax.plot(np.arange(3), [e["onset"] / e["onset_ref"], e["F_hold"] / e["F_hold_ref"], e["impulse"] / e["impulse_ref"]],
                    "o", mfc="white", mec=INK, ms=6, mew=1.2, label="MuJoCo, true parameters")
        ax.legend(frameon=False, fontsize=6, loc="upper left")
        ax.set_ylim(0, 1.8); ax.set_ylabel("prediction / reference"); ax.set_title("contact event (unseen task)")
        ax.set_xticks(np.arange(3)); ax.set_xticklabels([l for _, l in keys], fontsize=7); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    else:
        ax.text(0.5, 0.5, "contact results pending", ha="center", va="center", transform=ax.transAxes)


ORACLE_TCP = json.load(open("results/true_reference.json"))["replay_tcp_mm"]   # run_true_reference.py
rows = [("Same simulator: MuJoCo reference with injected discrepancies", same_engine(), ORACLE_TCP)]
xe = cross_engine()
if xe is not None:
    rows.append(("Cross simulator: Gazebo reference, MuJoCo twin", xe, None))
ie = isaac_engine()
if ie is not None:
    rows.append(("Cross simulator: Isaac Sim reference, MuJoCo twin", ie, None))
fig, axes = plt.subplots(len(rows), 4, figsize=(11, 2.9 * len(rows)), squeeze=False)
for r, (title, row, orc) in enumerate(rows):
    draw_row(axes[r], row, title, orc)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in COLS]
fig.legend(handles, LEGEND, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.04, 1, 0.97), w_pad=1.2, h_pad=3.0)
for r, (title, _, _) in enumerate(rows):                 # row titles above the first panel, after layout
    b = axes[r][0].get_position()
    fig.text(b.x0, b.y1 + 0.04, title, ha="left", va="bottom", fontsize=9.5, fontweight="bold", color=INK)
fig.savefig("results/fig_story.png", dpi=220); fig.savefig("results/fig_story.pdf")
print("saved results/fig_story.png")
