"""Fig. 2: stability selection (group selection frequency over noise realisations, estimate distributions) and the
identifiability diagnostic (column-subspace R2 between body groups)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C1, C3 = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold"})
S = json.load(open("results/stability.json"))
groups = ["link7", "joint4", "joint6", "link3", "link2", "link4", "link1", "link5", "link6", "joint1", "joint2", "joint3", "joint5", "joint7", "armature"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), gridspec_kw=dict(width_ratios=[1.3, 1.0, 1.0]))
ax = axes[0]; x = np.arange(len(groups)); w = 0.2
C1b, C3b = "#9cc3ee", "#f5b596"
series = [(S.get("x1"), "MuJoCo x1", C1), (S.get("x3"), "MuJoCo x3", C3)]
try:
    I = json.load(open("results/isaac_stability_v2/stability.json"))["cases"]
    series += [(dict(summary=dict(freq=I["ref_x1"]["summary"]["group_counts"], n=I["ref_x1"]["summary"]["n"])), "Isaac Sim x1", C1b),
               (dict(summary=dict(freq=I["ref_x3"]["summary"]["group_counts"], n=I["ref_x3"]["summary"]["n"])), "Isaac Sim x3", C3b)]
except (OSError, KeyError):
    pass
for i, (blk, lab, c) in enumerate(series):
    if blk is None:
        continue
    f = blk["summary"]["freq"]; n = blk["summary"]["n"]
    ax.bar(x + (i - (len(series) - 1) / 2) * w, [f.get(g, 0) / n for g in groups], width=w - 0.02, color=c, linewidth=0, label=f"{lab} (n={n})")
ax.set_xticks(x); ax.set_xticklabels(groups, rotation=60, fontsize=7); ax.set_ylim(0, 1.18); ax.set_ylabel("selection frequency")
ax.set_title("A  Selected groups"); ax.legend(frameon=False, fontsize=6.5, loc="upper right", ncol=2); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax.axvspan(-0.5, 3.5, color=GRID, alpha=0.5, lw=0, zorder=0); ax.text(1.5, 1.1, "injected groups", ha="center", fontsize=7, color=INK2)
# B estimate distributions (normalised to truth)
ax = axes[1]
keys = [("payload_mass", 0.8, "payload\nmass"), ("com_z", 0.06, "CoM z"), ("j4_coul", 1.663, "j4\nCoulomb"), ("j4_visc", 0.34, "j4\nviscous"), ("link3_scale", 1.12, "link3\nmass")]
for i, (lvl, c) in enumerate((("x1", C1), ("x3", C3))):
    if lvl not in S:
        continue
    runs = S[lvl]["runs"]
    for k, (key, true, lab) in enumerate(keys):
        v = np.array([r[key] if key != "com_z" else r["com"][2] for r in runs]) / true
        ax.plot(np.full(len(v), k + (i - 0.5) * 0.25) + np.random.default_rng(k).uniform(-0.05, 0.05, len(v)), v, ".", color=c, ms=4, alpha=0.7)
ax.axhline(1.0, color=INK, lw=1, ls="--"); ax.set_ylim(0.4, 1.4); ax.set_ylabel("estimate / true")
ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, _, l in keys], fontsize=7); ax.set_title("B  MuJoCo estimates / true value")
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
# C identifiability: R2 between link groups
ax = axes[2]
if "identifiability" in S:
    D = S["identifiability"]; names = D["names"]; R2 = np.array(D["R2"], float)
    sel = ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "armature"]
    idx = [names.index(g) for g in sel]; M = R2[np.ix_(idx, idx)]
    ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(sel))); ax.set_xticklabels(sel, rotation=60, fontsize=7); ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel, fontsize=7)
    for i in range(len(sel)):
        for j in range(len(sel)):
            if i != j and not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color=INK if M[i, j] < 0.6 else "white")
    ax.set_title("C  Signature explained\nby another group", fontsize=8); ax.set_xlabel("explaining group"); ax.set_ylabel("group")
else:
    ax.text(0.5, 0.5, "identifiability pending", ha="center", va="center", transform=ax.transAxes)
fig.tight_layout(w_pad=1.5); fig.savefig("results/fig_stability.png", dpi=220); fig.savefig("results/fig_stability.pdf")
print("saved results/fig_stability.png")
