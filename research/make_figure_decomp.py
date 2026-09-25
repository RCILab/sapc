"""Cross-engine decomposition of the replay TCP RMSE on the held-out trajectories.
Left: Gazebo/DART reference (results/crossengine_ablation.json).  Right: Isaac/PhysX reference (results/isaac/analysis.json).
Log axis, values labelled; the rotor inertia found on nominal Gazebo data is listed in the left panel."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C_NOM, C_DENSE, C_SPARSE, C_ENG, C_TRUE = "#eb6834", "#1baf7a", "#2a78d6", "#7a7873", "#b9b7b1"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
R = json.load(open("results/crossengine_ablation.json"))
A = json.load(open("results/isaac/analysis.json"))
mean = lambda rows: float(np.mean([r["tcp_rmse_mm"] for r in rows]))
n_sapc = len(R["L3_forward"]["subset"])
gz = [("simulator only", R["L1_engine_only_tcp"], C_ENG),
      ("simulator only,\ncalibrated on\nnominal data", R["a_nominal_calibration"]["tcp_nom_va"], C_SPARSE),
      ("nominal twin", R["L2_engine_plus_physical_tcp"], C_NOM),
      ("dense LS", R["L3_dense"]["tcp"], C_DENSE),
      (f"SAPC\n({n_sapc} groups)", R["L3_forward"]["tcp"], C_SPARSE),
      ("injected\ngroups and\narmature", R["b_injected_plus_armature"]["tcp"], C_SPARSE),
      ("injected\ngroups,\nno armature", R["b_injected_only"]["tcp"], C_SPARSE)]
ph = [("simulator only", mean(A["engine_only"]), C_ENG),
      ("simulator only,\ncalibrated on\nnominal data", mean(A["nominal_calibration"]["replay"]), C_SPARSE),
      ("nominal twin", mean(A["methods"]["nominal"]["replay"]), C_NOM),
      ("dense LS", mean(A["methods"]["LS-full"]["replay"]), C_DENSE),
      ("SAPC", mean(A["methods"]["Forward"]["replay"]), C_SPARSE),
      ("true\nparameters", mean(A["same_physical_parameters"]), C_TRUE)]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw=dict(width_ratios=[len(gz), len(ph)]))
for ax, items, title in ((axes[0], gz, "Gazebo reference"), (axes[1], ph, "Isaac Sim reference")):
    x = np.arange(len(items)); vals = [v for _, v, _ in items]
    ax.bar(x, vals, color=[c for _, _, c in items], width=0.62, linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.2, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_yscale("log"); ax.set_ylim(0.05, 60); ax.set_ylabel("replay TCP RMSE [mm]")
    ax.set_xticks(x); ax.set_xticklabels([l for l, _, _ in items], fontsize=7); ax.set_title(title)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
arm = R["a_nominal_calibration"]["armature"]
axes[0].text(0.99, 0.97, "rotor inertia found on nominal data [kg m$^2$]:\n" + "  ".join(f"{a:.3f}" for a in arm),
             transform=axes[0].transAxes, ha="right", va="top", fontsize=6.5, color=INK2)
fig.tight_layout(w_pad=2.0); fig.savefig("results/fig_decomposition.png", dpi=220); fig.savefig("results/fig_decomposition.pdf")
print("saved results/fig_decomposition.png")
