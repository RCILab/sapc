"""Task B: stability selection and identifiability diagnostics.

For N noise realisations (sensor-noise model re-drawn on the same real trajectories) run Forward group selection and
record the selected groups and the physical estimates.  Report selection frequency per group, mean +- std and 5-95 %
percentiles of payload mass / CoM / friction.  Then explain the link2/3/4 confusion with a column-subspace diagnostic:
for every pair of groups, R2 of one group's (standardised) columns projected onto the other's column space, plus
rank of each group alone and jointly, on the identification rows.
Usage: python run_stability.py [n_seeds_x1] [n_seeds_x3]
"""
import sys, json, time, numpy as np
import fr3_twin as ft, identify as idf

N1 = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N3 = int(sys.argv[2]) if len(sys.argv) > 2 else 10
TRAIN, T = [1, 2, 3], 20.0
logf = open("results/stability_log.txt", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush(); sys.stdout.flush()


nom = ft.make_nominal(); real = ft.make_real()
logs_tr = idf.collect(real, nom, TRAIN, T=T, noise_scale=1.0, noise_seed=0)
out = {}
for scale, n_seeds in ((1.0, N1), (3.0, N3)):
    runs = []
    for k in range(n_seeds):
        rng = np.random.default_rng(1000 + 100 * int(scale) + k)
        for lg in logs_tr:
            lg["q_meas"], lg["tau_meas"] = ft.add_sensor_noise(lg, rng, scale=scale)
        t0 = time.time()
        data_tr = idf.make_dataset(nom, logs_tr, source="sg", window=101)
        xi, info = idf.identify(data_tr, "Forward", model_nom=nom)
        rec = idf.recovery_table(nom, xi)
        runs.append(dict(seed=k, subset=info["subset"], payload_mass=rec["payload_mass"][0], com=rec["payload_com"][0],
                         j4_coul=rec["j4_coul"][0], j4_visc=rec["j4_visc"][0], j6_coul=rec["j6_coul"][0], link3_scale=rec["link3_mass_scale"][0],
                         path=info["path"]))
        say(f"[x{scale:g} seed {k}] {info['subset']} payload {rec['payload_mass'][0]:.3f} com {np.round(rec['payload_com'][0], 4).tolist()} j4 {rec['j4_coul'][0]:.2f}/{rec['j4_visc'][0]:.2f} | {time.time() - t0:.0f} s")
    freq = {g: sum(g in r["subset"] for r in runs) for g in idf.GROUPS}
    def stats(key, idx=None):
        v = np.array([r[key] if idx is None else r[key][idx] for r in runs], float)
        return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, p5=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))
    summ = dict(n=n_seeds, freq=freq, payload_mass=stats("payload_mass"), com_x=stats("com", 0), com_y=stats("com", 1), com_z=stats("com", 2),
                j4_coul=stats("j4_coul"), j4_visc=stats("j4_visc"), j6_coul=stats("j6_coul"), link3_scale=stats("link3_scale"))
    out[f"x{scale:g}"] = dict(runs=runs, summary=summ)
    say(f"=== noise x{scale:g}, {n_seeds} realisations: selection frequency {freq}")
    say(f"    payload mass {summ['payload_mass']['mean']:.3f} +- {summ['payload_mass']['std']:.3f} [{summ['payload_mass']['p5']:.3f}, {summ['payload_mass']['p95']:.3f}] (true 0.8); "
        f"CoM z {summ['com_z']['mean']:.4f} +- {summ['com_z']['std']:.4f} (true 0.06); j4 coul {summ['j4_coul']['mean']:.3f} +- {summ['j4_coul']['std']:.3f} (true 1.663); "
        f"j4 visc {summ['j4_visc']['mean']:.3f} +- {summ['j4_visc']['std']:.3f} (true 0.34); link3 scale {summ['link3_scale']['mean']:.3f} +- {summ['link3_scale']['std']:.3f} (true 1.12)")
    json.dump(out, open("results/stability.json", "w"), indent=1)

# ---- identifiability diagnostic on one noise-x1 dataset
rng = np.random.default_rng(1000 + 100)
for lg in logs_tr:
    lg["q_meas"], lg["tau_meas"] = ft.add_sensor_noise(lg, rng, scale=1.0)
data_tr = idf.make_dataset(nom, logs_tr, source="sg", window=101)
A, y = idf.stack(data_tr)
scale_c = A.std(axis=0); scale_c[scale_c < 1e-12] = np.inf; As = A / scale_c
names = list(idf.GROUPS.keys())
def basis(cols):
    M = As[:, cols]; M = M[:, np.abs(M).max(0) > 0]
    if M.shape[1] == 0:
        return np.zeros((As.shape[0], 0)), 0
    U, s, _ = np.linalg.svd(M, full_matrices=False); r = int((s > 1e-8 * s[0]).sum())
    return U[:, :r], r
B = {g: basis(idf.GROUPS[g]) for g in names}
rank = {g: B[g][1] for g in names}
R2 = np.zeros((len(names), len(names)))
for i, gi in enumerate(names):
    Mi = As[:, idf.GROUPS[gi]]; Mi = Mi[:, np.abs(Mi).max(0) > 0]
    for j, gj in enumerate(names):
        if i == j or Mi.shape[1] == 0 or B[gj][1] == 0:
            R2[i, j] = np.nan if i != j else 1.0; continue
        P = B[gj][0]; res = Mi - P @ (P.T @ Mi)
        R2[i, j] = 1 - (res ** 2).sum() / (Mi ** 2).sum()
# joint rank of pairs among links 2/3/4 and the true groups
def rank_of(groups):
    cols = sum((idf.GROUPS[g] for g in groups), []); M = As[:, cols]; M = M[:, np.abs(M).max(0) > 0]
    s = np.linalg.svd(M, compute_uv=False); return int((s > 1e-8 * s[0]).sum()), M.shape[1]
diag = dict(names=names, rank=rank, R2=R2.tolist(),
            pair_ranks={f"{a}+{b}": rank_of([a, b]) for a, b in (("link2", "link3"), ("link3", "link4"), ("link2", "link4"))},
            true_groups_rank=rank_of(["link3", "link7", "joint4", "joint6"]), true_plus_link4=rank_of(["link3", "link4", "link7", "joint4", "joint6"]))
out["identifiability"] = diag
json.dump(out, open("results/stability.json", "w"), indent=1)
say("=== identifiability: group ranks", rank)
for gi in ("link2", "link3", "link4", "link7", "joint4", "armature"):
    i = names.index(gi)
    say(f"    R2 of {gi:8s} columns explained by: " + ", ".join(f"{gj} {R2[i, names.index(gj)]:.2f}" for gj in ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "armature") if gj != gi))
say("    pair ranks:", diag["pair_ranks"], " true groups rank", diag["true_groups_rank"], " true+link4", diag["true_plus_link4"])
say("done")
