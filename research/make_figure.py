"""Figures for claude_try2 (matplotlib, print-oriented).  Reads results/gate_results.json, results/contact_*.json.

Fig. 1  A: closed-loop replay TCP RMSE per identification method at three sensor-noise levels (log axis),
           with the nominal twin and the oracle-twin floor as reference lines.
        B: payload mass and CoM-z estimate per method (true values as lines) - physical-coordinate recovery.
        C: contact stage - probe force traces on the real system and on twins.
Palette: dataviz reference categorical order (blue, orange, aqua), text in ink, thin marks, one axis per panel.
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 9})

R = json.load(open("results/gate_results.json"))
tags = [t for t in R if t.startswith("E2")]
methods = ["LS-payload", "LS-struct", "LS-full", "LS-phys", "SINDy", "SINDy-phys", "BestSubset", "MLP"]
methods = [m for m in methods if m in R[tags[0]]]
labels = {"LS-payload": "LS\npayload", "LS-struct": "LS\nstruct.", "LS-full": "LS\nfull", "LS-phys": "LS\nbounded",
          "SINDy": "STLSQ", "SINDy-phys": "STLSQ\nbounded", "BestSubset": "best\nsubset", "MLP": "MLP\nresidual"}
ORACLE_TCP = 0.028

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw=dict(width_ratios=[1.35, 1.0, 1.1]))

# ---- A
ax = axes[0]
x = np.arange(len(methods)); w = 0.26
for i, tag in enumerate(tags):
    vals = [np.mean([r["tcp_rmse_mm"] for r in R[tag][m]["replay"]]) if m in R[tag] else np.nan for m in methods]
    scale = tag.split("x")[1].split(" ")[0]
    ax.bar(x + (i - 1) * w, vals, width=w - 0.03, color=C[i], label=f"noise ×{scale}", linewidth=0)
nominal = np.mean([r["tcp_rmse_mm"] for r in R[tags[0]]["nominal"]["replay"]])
ax.axhline(nominal, color=INK2, lw=1, ls="--"); ax.text(len(methods) - 0.5, nominal * 1.15, "nominal twin", ha="right", color=INK2)
ax.axhline(ORACLE_TCP, color=INK2, lw=1, ls=":"); ax.text(len(methods) - 0.5, ORACLE_TCP * 1.15, "oracle twin (numerical gap)", ha="right", color=INK2)
ax.set_yscale("log"); ax.set_ylim(0.01, 40)
ax.set_xticks(x); ax.set_xticklabels([labels[m] for m in methods])
ax.set_ylabel("closed-loop replay TCP RMSE [mm]"); ax.set_title("A  Twin fidelity vs identification method")
ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=7); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

# ---- B
ax = axes[1]
tag = [t for t in tags if "x1.0" in t][0]
pm = [R[tag][m]["recovery"].get("payload_mass", [np.nan])[0] if m in R[tag] and R[tag][m]["recovery"] else np.nan for m in methods]
cz = [R[tag][m]["recovery"].get("payload_com", [[np.nan] * 3])[0][2] if m in R[tag] and R[tag][m]["recovery"] else np.nan for m in methods]
ax.axhline(0.8, color=C[0], lw=1, ls="--"); ax.axhline(0.06, color=C[1], lw=1, ls="--")
ax.plot(x, pm, "o", color=C[0], ms=6, label="payload mass [kg]")
ax.plot(x, cz, "s", color=C[1], ms=5, label="payload CoM z [m]")
ax.text(0.05, 0.83, "true 0.8 kg", color=C[0], fontsize=7); ax.text(0.05, 0.075, "true 0.06 m", color=C[1], fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([labels[m] for m in methods]); ax.set_ylim(-0.5, 1.5)
ax.set_title("B  Physical-coordinate recovery (noise ×1)"); ax.legend(frameon=False, loc="lower left", fontsize=7)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)

# ---- C
ax = axes[2]
cfile = [f for f in os.listdir("results") if f.startswith("contact_") and f.endswith(".json")]
if cfile:
    Cc = json.load(open(os.path.join("results", cfile[0])))
    t = np.array(Cc["ref"]["t"]); F = np.array(Cc["ref"]["F"])
    # the raw force chatters at ~100 Hz on every plant; show a 50 ms moving average (the metric-bearing envelope)
    def smooth(y, n=50):
        return np.convolve(y, np.ones(n) / n, mode="same")
    ax.plot(t, smooth(F), color=INK, lw=2, label="real system")
    series = [("nominal arm + nominal contact", C[1], "nominal twin"), ("calibrated arm + nominal contact", C[2], "calibrated arm (best-subset)"),
              ("oracle arm + true contact, matched integrator", C[0], "oracle twin")]
    for key, col, lab in series:
        if key in Cc and "F_trace" in Cc[key]:
            y = np.array(Cc[key]["F_trace"]); ax.plot(t[:len(y)], smooth(y), color=col, lw=1.6, label=lab)
    ax.set_xlabel("time [s]"); ax.set_ylabel("probe normal force, 50 ms mean [N]"); ax.set_title("C  Contact stage: press on a plate")
    ax.legend(frameon=False, fontsize=7, loc="upper right"); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
else:
    ax.text(0.5, 0.5, "contact results pending", ha="center", va="center", transform=ax.transAxes)

fig.tight_layout(w_pad=1.5)
fig.savefig("results/fig_twin_fidelity.png", dpi=220); fig.savefig("results/fig_twin_fidelity.pdf")
print("saved results/fig_twin_fidelity.png")
