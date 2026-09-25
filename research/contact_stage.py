"""E4: contact stage.  After free-space calibration, identify the plate contact parameters (MuJoCo solref time
constant and damping ratio) from a probe-plate press and report force / penetration fidelity.

Real system: FR3 + GAP_TRUE, plate solref (0.002, 0.6), mu 0.3.   Nominal twin: plate solref (0.005, 1.0), mu 0.6.
Press: 1 s settle at home, then a Jacobian-resolved joint trajectory driving the probe 29 mm down (plate 26 mm
below the probe bottom at home, so 3 mm nominal penetration), hold 1 s, retract 1 s.  Same nominal-feedforward PD on every plant.
Twins: nominal | free-space calibrated (BestSubset or SINDy-phys) + nominal contact | calibrated + fitted contact
       | nominal arm + fitted contact | oracle (true arm + true contact).
Metrics: force RMSE [N], peak force error [N], impulse error [N s], penetration depth error [mm], TCP RMSE [mm].
"""
import sys, os, json, time, copy, itertools
import numpy as np
import mujoco
import fr3_twin as ft, identify as idf

METHOD = sys.argv[1] if len(sys.argv) > 1 else "BestSubset"
NOISE = 1.0
PLATE_TRUE = dict(solref=(0.002, 0.6), mu=0.3)
PLATE_NOM = dict(solref=(0.005, 1.0), mu=0.6)
GAP0, DESCENT, T_SETTLE = 0.026, 0.029, 1.0
logf = open("results/contact_log.txt", "a", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a); print(s); logf.write(s + "\n"); logf.flush(); sys.stdout.flush()


def plate_spec(solref, mu, pos):
    return dict(pos=pos, half=0.01, solref=solref, mu=mu, r_probe=0.01, solref_probe=(0.005, 1.0))


def probe_bottom_home():
    """Probe-centre position and Jacobian at home (plate placed far away), from a full mj_forward."""
    m = ft.load_model(contact=True, plate=plate_spec((0.005, 1.0), 0.6, (0.5, 0.0, -1.0)))
    d = mujoco.MjData(m); d.qpos[:] = ft.Q_HOME; mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "probe_center")
    p = d.site_xpos[sid].copy()
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, jacr, sid)
    return p, jacp


P_TCP, J_TCP = probe_bottom_home()          # probe centre at home
Z_BOTTOM = P_TCP[2] - 0.01                  # sphere radius 1 cm
PLATE_POS = (P_TCP[0], P_TCP[1], Z_BOTTOM - GAP0)


class Press:
    """q_d(t) = q_home + J^+ dp(t), dp = (0, 0, -z(t)); z: cosine ramp down (1 s), hold (1 s), ramp up (1 s)."""

    def __init__(self, descent=DESCENT, t_ramp=1.0, t_hold=1.0, t_settle=T_SETTLE):
        self.J_pinv = np.linalg.pinv(J_TCP)
        self.descent, self.t_ramp, self.t_hold, self.t_settle = descent, t_ramp, t_hold, t_settle

    def z(self, t):
        t = np.asarray(t, float) - self.t_settle          # settle at home first (the arm sags under its true load)
        t = np.maximum(t, 0.0)
        z = np.zeros_like(t); zd = np.zeros_like(t); zdd = np.zeros_like(t)
        A, tr, th = self.descent, self.t_ramp, self.t_hold
        w = np.pi / tr
        m1 = t < tr
        z[m1] = 0.5 * A * (1 - np.cos(w * t[m1])); zd[m1] = 0.5 * A * w * np.sin(w * t[m1]); zdd[m1] = 0.5 * A * w ** 2 * np.cos(w * t[m1])
        m2 = (t >= tr) & (t < tr + th)
        z[m2] = A
        m3 = (t >= tr + th) & (t < 2 * tr + th)
        s = t[m3] - tr - th
        z[m3] = 0.5 * A * (1 + np.cos(w * s)); zd[m3] = -0.5 * A * w * np.sin(w * s); zdd[m3] = -0.5 * A * w ** 2 * np.cos(w * s)
        return z, zd, zdd

    def __call__(self, t):
        z, zd, zdd = self.z(t)
        dp = np.stack([np.zeros_like(z), np.zeros_like(z), -z], axis=1)
        dv = np.stack([np.zeros_like(z), np.zeros_like(z), -zd], axis=1)
        da = np.stack([np.zeros_like(z), np.zeros_like(z), -zdd], axis=1)
        return ft.Q_HOME + dp @ self.J_pinv.T, dv @ self.J_pinv.T, da @ self.J_pinv.T


class ContactPlant(ft.Plant):
    """Plant with touch-sensor and probe-position logging."""

    def simulate(self, traj, controller, T, dt_ctrl=0.001):
        m, d = self.model, self.data
        nsub = int(round(dt_ctrl / m.opt.timestep)); nstep = int(round(T / dt_ctrl))
        q_d0, _, _ = traj(np.array([0.0]))
        self.reset(q_d0[0])
        F, P, Q, TAU = [], [], [], []
        qacc_prev = np.zeros(7)
        for k in range(nstep):
            t = k * dt_ctrl
            q_d, qd_d, qdd_d = traj(np.array([t]))
            tau = controller(d.qpos.copy(), d.qvel.copy(), q_d[0], qd_d[0], qdd_d[0])
            for _ in range(nsub):
                ex = self._extra(d.qpos, d.qvel, qacc_prev, tau)
                d.qfrc_applied[:] = tau + ex
                mujoco.mj_step(m, d)
                qacc_prev = d.qacc.copy()
            F.append(d.sensordata[0]); P.append(d.sensordata[1:4].copy()); Q.append(d.qpos.copy()); TAU.append(tau)
        return dict(t=np.arange(nstep) * dt_ctrl, F=np.array(F), p=np.array(P), q=np.array(Q), tau=np.array(TAU))


def make_plant(model_arm, plate, extra=None, timestep=None, integrator=None):
    """Build a contact scene whose arm parameters are copied from model_arm (free-space calibrated or real)."""
    m = ft.load_model(contact=True, plate=plate_spec(plate["solref"], plate["mu"], PLATE_POS),
                      timestep=model_arm.opt.timestep if timestep is None else timestep,
                      integrator=("implicitfast" if model_arm.opt.integrator == 3 else "implicit") if integrator is None else integrator)
    for b in range(model_arm.nbody):
        m.body_mass[b] = model_arm.body_mass[b]; m.body_ipos[b] = model_arm.body_ipos[b]
        m.body_inertia[b] = model_arm.body_inertia[b]; m.body_iquat[b] = model_arm.body_iquat[b]
    m.dof_frictionloss[:] = model_arm.dof_frictionloss; m.dof_damping[:] = model_arm.dof_damping; m.dof_armature[:] = model_arm.dof_armature
    mujoco.mj_setConst(m, mujoco.MjData(m))
    return ContactPlant(m, extra)


def force_metrics(log, ref):
    F, Fr = log["F"], ref["F"]
    n = min(len(F), len(Fr)); F, Fr = F[:n], Fr[:n]
    dt = ref["t"][1] - ref["t"][0]
    pen = (PLATE_POS[2] - (log["p"][:n, 2] - 0.01)); pen_r = (PLATE_POS[2] - (ref["p"][:n, 2] - 0.01))
    hold = slice(int(round((T_SETTLE + 1.2) / dt)), int(round((T_SETTLE + 1.9) / dt)))     # inside the 1 s hold
    return dict(F_rmse=float(np.sqrt(((F - Fr) ** 2).mean())), F_peak=float(F.max()), F_peak_ref=float(Fr.max()),
                F_hold=float(F[hold].mean()), F_hold_ref=float(Fr[hold].mean()), impulse=float(F.sum() * dt), impulse_ref=float(Fr.sum() * dt),
                F_peak_err=float(F.max() - Fr.max()), impulse_err=float((F.sum() - Fr.sum()) * dt),
                pen_max_mm=float(1e3 * pen.max()), pen_max_ref_mm=float(1e3 * pen_r.max()),
                tcp_rmse_mm=float(1e3 * np.sqrt(((log["p"][:n] - ref["p"][:n]) ** 2).sum(axis=1).mean())),
                onset=float(log["t"][np.argmax(F > 0.05)]) if (F > 0.05).any() else None,
                onset_ref=float(ref["t"][np.argmax(Fr > 0.05)]) if (Fr > 0.05).any() else None)


def fit_contact(model_arm, extra, ref, ctrl, press, T, grid_d=(0.001, 0.0015, 0.002, 0.003, 0.005, 0.008), grid_z=(0.3, 0.6, 1.0, 1.5), mu=0.6):
    """Grid search of the plate solref (time constant, damping ratio) minimising the force-trace RMSE."""
    best, table = None, []
    for dcon, zeta in itertools.product(grid_d, grid_z):
        pl = make_plant(model_arm, dict(solref=(dcon, zeta), mu=mu), extra)
        log = pl.simulate(press, ctrl, T)
        fm = force_metrics(log, ref)
        table.append((dcon, zeta, fm["F_rmse"], fm["F_peak"]))
        if best is None or fm["F_rmse"] < best[2]:
            best = (dcon, zeta, fm["F_rmse"])
    return best, table


if __name__ == "__main__":
    T = 4.0
    nom = ft.make_nominal(); real = ft.make_real()
    ctrl = ft.PD(nom); press = Press()
    say(f"=== E4 contact stage ({METHOD}) === plate top {PLATE_POS[2]:.4f} m, probe bottom at home {Z_BOTTOM:.4f} m")
    # real press
    real_c = make_plant(real, PLATE_TRUE)
    ref = real_c.simulate(press, ctrl, T)
    say(f"real: peak force {ref['F'].max():.2f} N, onset {ref['t'][np.argmax(ref['F'] > 0.05)]:.3f} s, max penetration {1e3 * (PLATE_POS[2] - (ref['p'][:, 2] - 0.01)).max():.3f} mm")
    # free-space calibration (from the same data protocol as run_gate: SG, noise x1)
    t0 = time.time()
    logs_tr = idf.collect(real, nom, [1, 2, 3], T=20.0, noise_scale=NOISE, noise_seed=17)
    data_tr = idf.make_dataset(nom, logs_tr, source="sg", window=101)
    # Always refit: a cached coefficient vector can silently outlive a change of the library or of the selection rule.
    xi, info = idf.identify(data_tr, METHOD, model_nom=nom)
    np.save(f"results/contact_xi_{METHOD}.npy", xi)
    model_cal, extra_cal, rep = idf.instantiate(nom, xi)
    say(f"free-space calibration ({METHOD}) {time.time() - t0:.0f} s: {info.get('subset', info.get('thr'))}; twin {rep}")
    xi_true = np.concatenate([ft.gap_true_dpi(nom, ft.GAP_TRUE).ravel(), ft.gap_true_fric(nom, ft.GAP_TRUE).ravel()])
    model_orc, extra_orc, _ = idf.instantiate(nom, xi_true)
    out = dict(ref_peak=float(ref["F"].max()))
    variants = {
        "nominal arm + nominal contact": (nom, None, PLATE_NOM),
        "calibrated arm + nominal contact": (model_cal, extra_cal, PLATE_NOM),
        "oracle arm + true contact": (model_orc, extra_orc, PLATE_TRUE),
        "oracle arm + true contact, matched integrator": (model_orc, extra_orc, PLATE_TRUE),
    }
    for name, (marm, ex, plate) in variants.items():
        kw = dict(timestep=0.0005, integrator="implicit") if "matched" in name else {}
        log = make_plant(marm, plate, ex, **kw).simulate(press, ctrl, T)
        out[name] = force_metrics(log, ref); out[name]["F_trace"] = log["F"].tolist()
        o = out[name]
        say(f"{name:46s} F rmse {o['F_rmse']:.2f} N | onset {o['onset']:.3f} (ref {o['onset_ref']:.3f}) | hold {o['F_hold']:.2f} (ref {o['F_hold_ref']:.2f}) N | impulse {o['impulse']:.2f} (ref {o['impulse_ref']:.2f}) N s | peak {o['F_peak']:.1f} (ref {o['F_peak_ref']:.1f}) | tcp {o['tcp_rmse_mm']:.3f} mm")
    for name, (marm, ex) in {"nominal arm + fitted contact": (nom, None), "calibrated arm + fitted contact": (model_cal, extra_cal)}.items():
        t0 = time.time()
        best, table = fit_contact(marm, ex, ref, ctrl, press, T)
        log = make_plant(marm, dict(solref=(best[0], best[1]), mu=0.6), ex).simulate(press, ctrl, T)
        out[name] = dict(force_metrics(log, ref), solref=[best[0], best[1]], grid=table, F_trace=log["F"].tolist())
        o = out[name]
        say(f"{name:46s} F rmse {o['F_rmse']:.2f} N | onset {o['onset']:.3f} | hold {o['F_hold']:.2f} N | impulse {o['impulse']:.2f} N s | peak {o['F_peak']:.1f} | tcp {o['tcp_rmse_mm']:.3f} mm | solref {best[0]:g}/{best[1]:g} (true {PLATE_TRUE['solref']}) | {time.time() - t0:.0f} s")
    out["ref"] = dict(t=ref["t"].tolist(), F=ref["F"].tolist())
    json.dump(out, open(f"results/contact_{METHOD}.json", "w"), indent=1)
    say("done")
