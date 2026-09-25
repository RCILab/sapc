"""Plot saved Forward-only stability results, without fitting or modifying the paper."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "results/isaac_stability")
    a = p.parse_args()
    result = json.loads((a.data / "stability.json").read_text(encoding="utf-8"))
    if result["status"] != "complete":
        raise ValueError("Plot only the completed preplanned study")
    names = [k for k in ("ref_x1", "ref_x3", "nom_x1") if k in result["cases"]]
    labels = {"ref_x1": "Injected gap\nnoise ×1", "ref_x3": "Injected gap\nnoise ×3", "nom_x1": "No injected gap\nnoise ×1"}
    colors = {"ref_x1": "#2677b6", "ref_x3": "#e48732", "nom_x1": "#62736b"}
    groups = ["link7", "joint4", "joint6", "link3", "link4", "link2", "link1", "link5", "link6",
              "joint1", "joint2", "joint3", "joint5", "joint7", "armature"]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(12.5, 6.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.2, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    freq = np.array([[result["cases"][name]["summary"]["group_frequency"][g] for name in names] for g in groups])
    ax.imshow(freq, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(names)), [labels[k] for k in names])
    ax.set_yticks(range(len(groups)), groups)
    for i, g in enumerate(groups):
        for j, name in enumerate(names):
            s = result["cases"][name]["summary"]
            ax.text(j, i, f"{s['group_counts'][g]}/{s['n']}", ha="center", va="center", color="white" if freq[i, j] > .6 else "black")
    ax.axhline(3.5, color="black", lw=.8)
    ax.set_title("A  Selection frequency\n(first four rows: injected groups)", loc="left")

    def distribution(axis, key, title, ylabel, selected, true=None):
        for i, name in enumerate(selected):
            vals = [r[key] for r in result["cases"][name]["runs"] if r[key] is not None]
            x = np.linspace(i-.09, i+.09, len(vals))
            axis.scatter(x, vals, s=24, alpha=.7, color=colors[name])
            axis.errorbar(i+.16, np.mean(vals), yerr=np.std(vals, ddof=1) if len(vals)>1 else 0,
                          color=colors[name], capsize=4, fmt="s", ms=4)
        if true is not None:
            axis.axhline(true, color="#444444", linestyle="--", linewidth=1, label="True value")
            axis.legend(frameon=False, fontsize=8)
        axis.set_xticks(range(len(selected)), [labels[k] for k in selected])
        axis.set_title(title, loc="left")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.2)

    refs = [k for k in names if k.startswith("ref")]
    distribution(fig.add_subplot(grid[0, 1]), "payload_mass_kg", "B  Recovered payload", "Mass [kg]", refs, .8)
    distribution(fig.add_subplot(grid[0, 2]), "com_error_mm", "C  Payload CoM error", "Euclidean error [mm]", refs)
    distribution(fig.add_subplot(grid[1, 1]), "tcp_rmse_mm", "D  Held-out replay", "TCP RMSE [mm]", names)
    distribution(fig.add_subplot(grid[1, 2]), "val_nrmse", "E  Held-out torque", "NRMSE", names)
    fig.suptitle("PhysX → MuJoCo: Forward attribution stability\nFixed trajectories; training measurement noise resampled; validation noise fixed at ×1", fontsize=12)
    for extension in ("png", "pdf"):
        fig.savefig(a.data / f"stability.{extension}", dpi=200)
    print(a.data / "stability.png")


if __name__ == "__main__":
    main()
