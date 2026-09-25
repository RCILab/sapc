"""Regression check after adding armature columns: Forward selection vs the true groups on same-engine data."""
import time, sys, numpy as np
import fr3_twin as ft, identify as idf


def say(*a):
    print(*a); sys.stdout.flush()


nom = ft.make_nominal(); real = ft.make_real()
logs = idf.collect(real, nom, [1, 2, 3], T=20.0, noise_scale=1.0)
data = idf.make_dataset(nom, logs, source="sg", window=101)
say("library columns:", data[0]["A"].shape)
t0 = time.time(); xi, info = idf.forward_groups(data, nom, verbose=say)
say("Forward", round(time.time() - t0), "s:", info["subset"], info["path"])
rec = idf.recovery_table(nom, xi)
say({k: rec[k] for k in ("payload_mass", "payload_com", "j4_coul", "j4_visc", "armature", "groups_found", "group_precision", "group_recall")})
xi_true = np.concatenate([ft.gap_true_dpi(nom, ft.GAP_TRUE).ravel(), ft.gap_true_fric(nom, ft.GAP_TRUE).ravel(), np.zeros(7)])
tw, rep = idf.twin_from(nom, xi_true); say("oracle twin with 112 cols:", rep)
tw, rep = idf.twin_from(nom, xi); say("forward twin:", rep)
say("forward replay:", idf.replay_metrics(tw, nom, idf.collect(real, nom, [11], T=10.0)[0]))
