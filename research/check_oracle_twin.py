"""Oracle twin: instantiate the TRUE gap through identify.instantiate and replay.  Isolates twin/replay bugs
from identification errors.  Also checks the pseudo-inertia projection."""
import numpy as np, sys
import fr3_twin as ft, identify as idf


def say(*a):
    print(*a); sys.stdout.flush()


nom = ft.make_nominal(); real = ft.make_real()
logs = idf.collect(real, nom, [11], T=10.0, noise_scale=1.0)
xi_true = np.concatenate([ft.gap_true_dpi(nom, ft.GAP_TRUE).ravel(), ft.gap_true_fric(nom, ft.GAP_TRUE).ravel()])
tw, rep = idf.twin_from(nom, xi_true); say("oracle twin:", rep)
say("oracle twin replay:", idf.replay_metrics(tw, nom, logs[0]))
say("oracle twin open-loop 0.2 s:", idf.openloop_metrics(tw, nom, logs[0], window=0.2))
tw2, _ = idf.twin_from(nom, xi_true, timestep=0.0005); tw2.model.opt.integrator = 1   # implicit, as the real system
say("oracle twin (0.5 ms implicit) replay:", idf.replay_metrics(tw2, nom, logs[0]))
tw0 = ft.Plant(ft.make_nominal()); say("nominal replay:", idf.replay_metrics(tw0, nom, logs[0]))
say("nominal open-loop 0.2 s:", idf.openloop_metrics(tw0, nom, logs[0], window=0.2))
pi = ft.body_pi(nom, "fr3_link7"); bad = pi.copy(); bad[4] = -0.01
p = ft.project_pi(bad); say("projection valid:", ft.params_from_pi(p)[4], "rel dist", np.linalg.norm(p - bad) / np.linalg.norm(bad))
say("true payload pi valid:", ft.params_from_pi(ft.body_pi(nom, "fr3_link7") + ft.gap_true_dpi(nom, ft.GAP_TRUE)[6])[4])
