"""E0: consistency checks of the test bed (pi conversions, regressor exactness, inverse/forward agreement)."""
import time, numpy as np, mujoco
import fr3_twin as ft

rng = np.random.default_rng(0)
# 1. pi round trip
for _ in range(5):
    m = rng.uniform(0.2, 3); c = rng.normal(size=3) * 0.05; w = np.sort(rng.uniform(1e-3, 5e-3, 3)); q = ft._rotquat(rng.normal(size=3), rng.uniform(0, 3))
    pi = ft.pi_from_params(m, c, w, q); m2, c2, w2, q2, ok = ft.params_from_pi(pi); pi2 = ft.pi_from_params(m2, c2, w2, q2)
    assert ok and np.allclose(pi, pi2, atol=1e-12), (pi, pi2)
print("pi round trip ok; basis cond =", np.linalg.cond(ft.P_BASIS))

# 2. regressor exactness: nominal + (payload, link3 scale) with no joint mods
nom = ft.make_nominal()
gap_in = dict(payload=ft.GAP_TRUE["payload"], link_mass_scale=ft.GAP_TRUE["link_mass_scale"])
R_in = ft.apply_gap(ft.make_nominal(), gap_in)
n = 200
q = ft.Q_HOME + rng.uniform(-0.8, 0.8, (n, 7)); qd = rng.normal(size=(n, 7)); qdd = rng.normal(size=(n, 7)) * 3
t0 = time.time(); Y, tau0 = ft.inertial_regressor(nom, q, qd, qdd); print("regressor time per sample [ms]", 1e3 * (time.time() - t0) / n)
dpi = ft.gap_true_dpi(nom, gap_in).ravel()
tauR = ft.inverse_batch(R_in, q, qd, qdd)
err = tauR - tau0 - Y @ dpi
print("regressor max |err| [Nm] =", np.abs(err).max(), " (torque scale", np.abs(tauR - tau0).max(), ")")

# 3. friction deltas: nominal + joint mods only
gap_f = dict(joints=ft.GAP_TRUE["joints"])
R_f = ft.apply_gap(ft.make_nominal(), gap_f)
tauF = ft.inverse_batch(R_f, q, qd, qdd) - tau0
xi = ft.gap_true_fric(nom, gap_f)
pred = ft.friction_torque(xi, qd)
print("friction delta max |err| [Nm] =", np.abs(tauF - pred).max(), " (scale", np.abs(tauF).max(), ") ; at |qd|>0.1:",
      np.abs((tauF - pred)[np.abs(qd) > 0.1]).max())

# 4. inverse vs forward on the nominal plant
traj = ft.Fourier(1)
ctrl = ft.PD(nom)
pl = ft.Plant(ft.make_nominal())
t0 = time.time(); log = pl.simulate(traj, ctrl, 3.0); print("sim 3 s wall [s]", time.time() - t0)
tau_inv = ft.inverse_batch(nom, log["q"], log["qd"], log["qdd"])
print("inverse-forward RMSE per joint [Nm]:", np.sqrt(((tau_inv - log["tau"]) ** 2).mean(axis=0)).round(4))
print("tracking RMSE [rad]:", np.sqrt(((log["q"] - log["q_ref"]) ** 2).mean(axis=0)).round(4))
# 5. real plant tracking
plR = ft.Plant(ft.make_real())
logR = plR.simulate(traj, ctrl, 3.0)
print("real tracking RMSE [rad]:", np.sqrt(((logR["q"] - logR["q_ref"]) ** 2).mean(axis=0)).round(4), " |tau| max", np.abs(logR["tau"]).max(axis=0).round(1))
tau_nomR = ft.inverse_batch(nom, logR["q"], logR["qd"], logR["qdd"])
print("real: residual tau_meas - tau_nom RMS per joint [Nm]:", np.sqrt(((logR["tau"] - tau_nomR) ** 2).mean(axis=0)).round(3))
