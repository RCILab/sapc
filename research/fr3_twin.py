"""FR3 digital-twin test bed (claude_try2).

Nominal simulator T0 : MuJoCo Menagerie fr3.xml, torque driven, contacts disabled in free space.
'Real' system R      : the same model with a deliberately introduced physics gap (payload on the
                       flange, joint friction/damping changes, link mass change, finer integrator).
Physics-gap library  : per sample (q, qd, qdd)
        tau_meas - tau_T0(q, qd, qdd) = Y_inert(q, qd, qdd) dpi + Phi_fric(qd) xi_f + e
    Y_inert : 7 x 70   inertial regressor of bodies link1..link7 (10 standard parameters each:
                       m, m c_x, m c_y, m c_z, I_xx, I_xy, I_xz, I_yy, I_yz, I_zz about the body origin),
                       extracted from the simulator itself: tau is linear in pi, so ten inverse-dynamics
                       calls with physically valid basis bodies give the regressor exactly (pi-basis trick).
                       A payload rigidly attached to link 7 is a change of link 7's pi.
    Phi_fric: 7 x 35   per-joint friction basis: Coulomb tanh(qd/v_s), viscous qd, Stribeck, qd|qd|, offset.
Everything SI.
"""
import os, re, copy
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
FR3_DIR = os.path.join(HERE, "models", "franka_fr3")
NJ = 7
LINKS = [f"fr3_link{i}" for i in range(1, 8)]
JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
TAU_LIM = np.array([87, 87, 87, 87, 12, 12, 12], float)
Q_HOME = np.array([0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
FRIC_NAMES = ["coul", "visc", "strib", "quad", "off"]
V_S, V_STR = 0.02, 0.1        # Coulomb smoothing and Stribeck velocity scales [rad/s]
PI_NAMES = ["m", "mcx", "mcy", "mcz", "Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"]


# ----------------------------------------------------------------------------- XML
def build_xml(timestep=0.001, integrator="implicitfast", contact=False, plate=None):
    """fr3.xml -> torque-driven variant: actuators removed (we drive qfrc_applied), tiny payload body on
    the flange (its parameters are set at the model level), contacts optionally disabled, optional plate."""
    src = open(os.path.join(FR3_DIR, "fr3.xml"), encoding="utf-8").read()
    src = re.sub(r"<actuator>.*?</actuator>", "", src, flags=re.S)
    src = re.sub(r"<keyframe>.*?</keyframe>", "", src, flags=re.S)
    flag = "" if contact else '<flag contact="disable"/>'
    src = src.replace('<option integrator="implicitfast"/>',
                      f'<option timestep="{timestep}" integrator="{integrator}" gravity="0 0 -9.81">{flag}</option>')
    payload = ('<site name="attachment_site" pos="0 0 0.107"/>'
               '<body name="payload" pos="0 0 0.107"><inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>'
               '<site name="tcp" pos="0 0 0" size="0.002"/>{probe}</body>')
    probe = ""
    if plate is not None:
        r = plate["r_probe"]
        probe = (f'<geom name="probe_geom" type="sphere" size="{r}" pos="0 0 {r}" mass="0.05" '
                 f'solref="{plate["solref_probe"][0]} {plate["solref_probe"][1]}" solimp="0.9 0.95 0.001" '
                 f'friction="{plate["mu"]} 0.005 0.0001" rgba="1 0 0 1"/>'
                 f'<site name="probe_center" pos="0 0 {r}" size="{1.2 * r}" type="sphere" rgba="1 0 0 0.2"/>')
    src = src.replace('<site name="attachment_site" pos="0 0 0.107"/>', payload.format(probe=probe))
    if plate is not None:
        p = plate
        body = (f'<body name="plate" pos="{p["pos"][0]} {p["pos"][1]} {p["pos"][2] - p["half"]}">'
                f'<geom name="plate_geom" type="box" size="0.15 0.15 {p["half"]}" friction="{p["mu"]} 0.005 0.0001" '
                f'solref="{p["solref"][0]} {p["solref"][1]}" solimp="0.9 0.95 0.001" rgba="0.6 0.6 0.6 1"/></body>')
        src = src.replace("</worldbody>", body + "</worldbody>")
        src = src.replace("</mujoco>", '<sensor><touch name="probe_touch" site="probe_center"/>'
                          '<framepos name="probe_pos" objtype="site" objname="probe_center"/></sensor></mujoco>')
    return src


def load_model(**kw):
    xml = build_xml(**kw)
    path = os.path.join(FR3_DIR, f"_generated_{os.getpid()}.xml")   # meshdir is relative to the file location; per-process name (parallel runs)
    open(path, "w", encoding="utf-8").write(xml)
    try:
        return mujoco.MjModel.from_xml_path(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ----------------------------------------------------------------------------- rigid-body parameters
def quat2mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def pi_from_params(mass, ipos, inertia_diag, iquat=(1, 0, 0, 0)):
    """Standard 10-vector (m, m c, I about the body origin) of a body given MuJoCo's (mass, ipos, inertia, iquat)."""
    c = np.asarray(ipos, float)
    R = quat2mat(iquat)
    Ic = R @ np.diag(np.asarray(inertia_diag, float)) @ R.T
    Io = Ic + mass * (c @ c * np.eye(3) - np.outer(c, c))
    return np.array([mass, mass * c[0], mass * c[1], mass * c[2],
                     Io[0, 0], Io[0, 1], Io[0, 2], Io[1, 1], Io[1, 2], Io[2, 2]])


def params_from_pi(pi, mass_floor=1e-6):
    """Inverse of pi_from_params.  Returns (mass, ipos, inertia_diag, iquat, valid).
    valid = positive mass, positive-definite CoM inertia satisfying the triangle inequalities."""
    pi = np.asarray(pi, float)
    m = pi[0]
    if m <= mass_floor:
        return m, np.zeros(3), np.full(3, 1e-9), np.array([1, 0, 0, 0.]), False
    c = pi[1:4] / m
    Io = np.array([[pi[4], pi[5], pi[6]], [pi[5], pi[7], pi[8]], [pi[6], pi[8], pi[9]]])
    Ic = Io - m * (c @ c * np.eye(3) - np.outer(c, c))
    w, V = np.linalg.eigh(Ic)
    if np.linalg.det(V) < 0:
        V[:, 0] *= -1
    valid = bool(w.min() > 0 and all(w[i] <= w[(i + 1) % 3] + w[(i + 2) % 3] + 1e-12 for i in range(3)))
    return m, c, w, mat2quat(V), valid


def pseudo_inertia(pi):
    """4x4 pseudo-inertia J(pi) = [[0.5 tr(Io) 1 - Io, h], [h^T, m]]; physically consistent iff J is PSD."""
    m, h = pi[0], pi[1:4]
    Io = np.array([[pi[4], pi[5], pi[6]], [pi[5], pi[7], pi[8]], [pi[6], pi[8], pi[9]]])
    J = np.zeros((4, 4))
    J[:3, :3] = 0.5 * np.trace(Io) * np.eye(3) - Io
    J[:3, 3] = h; J[3, :3] = h; J[3, 3] = m
    return J


def pi_from_pseudo(J):
    Sig, h, m = J[:3, :3], J[:3, 3], J[3, 3]
    Io = np.trace(Sig) * np.eye(3) - Sig
    return np.array([m, h[0], h[1], h[2], Io[0, 0], Io[0, 1], Io[0, 2], Io[1, 1], Io[1, 2], Io[2, 2]])


def project_pi(pi, eig_floor=1e-7, mass_floor=1e-3):
    """Nearest (Frobenius, in pseudo-inertia coordinates) physically consistent pi: eigenvalue clipping of J."""
    J = pseudo_inertia(np.asarray(pi, float))
    w, V = np.linalg.eigh(J)
    w = np.maximum(w, eig_floor)
    Jp = V @ np.diag(w) @ V.T
    pi_p = pi_from_pseudo(Jp)
    if pi_p[0] < mass_floor:                          # keep a tiny mass so MuJoCo accepts the body
        pi_p = pi_from_params(mass_floor, np.zeros(3), np.full(3, eig_floor))
    return pi_p


def body_pi(model, bname):
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    return pi_from_params(model.body_mass[b], model.body_ipos[b], model.body_inertia[b], model.body_iquat[b])


def set_body_pi(model, bname, pi):
    """Write a pi-vector into a body's MuJoCo inertial fields (must be physically valid).  Calls mj_setConst."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
    m, c, w, q, valid = params_from_pi(pi)
    if not valid:
        raise ValueError(f"pi of {bname} is not a valid rigid body")
    model.body_mass[b] = m
    model.body_ipos[b] = c
    model.body_inertia[b] = w
    model.body_iquat[b] = q
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return model


def _rotquat(axis, ang):
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), ang)
    return q


# Ten physically valid basis bodies whose pi-vectors span R^10.
BASIS_BODIES = [
    (1.0, (0, 0, 0), (1e-2, 1e-2, 1e-2), (1, 0, 0, 0)),
    (1.0, (0.1, 0, 0), (1e-2, 1e-2, 1e-2), (1, 0, 0, 0)),
    (1.0, (0, 0.1, 0), (1e-2, 1e-2, 1e-2), (1, 0, 0, 0)),
    (1.0, (0, 0, 0.1), (1e-2, 1e-2, 1e-2), (1, 0, 0, 0)),
    (1.0, (0, 0, 0), (2e-2, 1e-2, 1e-2), (1, 0, 0, 0)),
    (1.0, (0, 0, 0), (1e-2, 2e-2, 1e-2), (1, 0, 0, 0)),
    (1.0, (0, 0, 0), (1e-2, 1e-2, 2e-2), (1, 0, 0, 0)),
    (1.0, (0, 0, 0), (2e-2, 1e-2, 1e-2), _rotquat((0, 0, 1), np.pi / 4)),
    (1.0, (0, 0, 0), (1e-2, 2e-2, 1e-2), _rotquat((1, 0, 0), np.pi / 4)),
    (1.0, (0, 0, 0), (1e-2, 1e-2, 2e-2), _rotquat((0, 1, 0), np.pi / 4)),
]
P_BASIS = np.stack([pi_from_params(*b) for b in BASIS_BODIES], axis=1)      # 10 x 10, columns = pi_k
P_INV = np.linalg.inv(P_BASIS)


# ----------------------------------------------------------------------------- the physics gap
GAP_TRUE = dict(
    payload=dict(mass=0.8, com=(0.02, -0.015, 0.06), inertia=(1.5e-3, 1.5e-3, 0.8e-3)),   # in the payload frame at the flange
    joints={"fr3_joint4": dict(frictionloss=2.8, damping=0.55), "fr3_joint6": dict(frictionloss=1.1)},
    link_mass_scale={"fr3_link3": 1.12},
)


def apply_gap(model, gap):
    """Write a physics gap into a model in place (payload body, joint friction/damping, link mass scale)."""
    if gap.get("payload"):
        p = gap["payload"]
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        model.body_mass[b] = p["mass"]
        model.body_ipos[b] = np.asarray(p["com"], float)
        model.body_inertia[b] = np.asarray(p["inertia"], float)
        model.body_iquat[b] = np.array([1, 0, 0, 0.])
    for jn, mods in gap.get("joints", {}).items():
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        dof = model.jnt_dofadr[j]
        for key, val in mods.items():
            getattr(model, "dof_" + key)[dof] = val
    for bn, s in gap.get("link_mass_scale", {}).items():
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bn)
        model.body_mass[b] *= s
        model.body_inertia[b] *= s
    mujoco.mj_setConst(model, mujoco.MjData(model))
    return model


def gap_true_dpi(model_nom, gap):
    """True change of the 70 inertial parameters (link1..link7, in each link's own frame) implied by a gap.
    The payload sits at body 'payload' (pos p0, identity orientation in link 7's frame)."""
    dpi = np.zeros((NJ, 10))
    if gap.get("payload"):
        p = gap["payload"]
        b = mujoco.mj_name2id(model_nom, mujoco.mjtObj.mjOBJ_BODY, "payload")
        p0 = model_nom.body_pos[b]
        dpi[6] += pi_from_params(p["mass"], p0 + np.asarray(p["com"]), p["inertia"])
    for bn, s in gap.get("link_mass_scale", {}).items():
        i = LINKS.index(bn)
        dpi[i] += (s - 1.0) * body_pi(model_nom, bn)
    return dpi


def gap_true_arm(model_nom, gap):
    """True armature change (7,) implied by a gap: gap['armature'] = 'none' means the reference has no rotor inertia."""
    a = gap.get("armature")
    if a is None:
        return np.zeros(NJ)
    if isinstance(a, str) and a == "none":
        return -np.array([model_nom.dof_armature[model_nom.jnt_dofadr[mujoco.mj_name2id(model_nom, mujoco.mjtObj.mjOBJ_JOINT, jn)]] for jn in JOINTS])
    return np.asarray(a, float)


def gap_true_fric(model_nom, gap):
    """True friction-basis coefficients (7 x 5) implied by a gap: Coulomb and viscous deltas only."""
    xi = np.zeros((NJ, len(FRIC_NAMES)))
    for jn, mods in gap.get("joints", {}).items():
        j = JOINTS.index(jn)
        dof = model_nom.jnt_dofadr[mujoco.mj_name2id(model_nom, mujoco.mjtObj.mjOBJ_JOINT, jn)]
        if "frictionloss" in mods:
            xi[j, 0] = mods["frictionloss"] - model_nom.dof_frictionloss[dof]
        if "damping" in mods:
            xi[j, 1] = mods["damping"] - model_nom.dof_damping[dof]
    return xi


# ----------------------------------------------------------------------------- inverse dynamics and library
def inverse_batch(model, q, qd, qdd, data=None):
    """qfrc_inverse for a batch of samples (n x 7 each).  Includes MuJoCo's own passive and friction-loss forces."""
    data = mujoco.MjData(model) if data is None else data
    n = q.shape[0]
    out = np.empty((n, NJ))
    for i in range(n):
        data.qpos[:] = q[i]
        data.qvel[:] = qd[i]
        data.qacc[:] = qdd[i]
        mujoco.mj_inverse(model, data)
        out[i] = data.qfrc_inverse
    return out


class RegressorCache:
    """Pre-built basis models for the pi-basis trick (friction-loss constraint disabled: rigid-body part only)."""

    def __init__(self, model_nom, bodies=LINKS):
        self.bodies = list(bodies)
        self.m_nf = copy.copy(model_nom)
        self.m_nf.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS
        self.d_nf = mujoco.MjData(self.m_nf)
        self.models = []
        for bn in self.bodies:
            pib = body_pi(model_nom, bn)
            ms = []
            for k in range(10):
                mk = copy.copy(self.m_nf)
                set_body_pi(mk, bn, pib + P_BASIS[:, k])
                ms.append((mk, mujoco.MjData(mk)))
            self.models.append(ms)

    def __call__(self, q, qd, qdd):
        n = q.shape[0]
        tau_nf = inverse_batch(self.m_nf, q, qd, qdd, self.d_nf)
        Y = np.empty((n, NJ, 10 * len(self.bodies)))
        for bi, ms in enumerate(self.models):
            dT = np.empty((n, NJ, 10))
            for k, (mk, dk) in enumerate(ms):
                dT[:, :, k] = inverse_batch(mk, q, qd, qdd, dk) - tau_nf
            Y[:, :, 10 * bi:10 * (bi + 1)] = dT @ P_INV
        return Y


def inertial_regressor(model_nom, q, qd, qdd, bodies=LINKS, tau0=None, cache=None):
    """Y (n x 7 x 10*len(bodies)): d tau / d pi for each body, from the pi-basis trick."""
    if tau0 is None:
        tau0 = inverse_batch(model_nom, q, qd, qdd)
    cache = RegressorCache(model_nom, bodies) if cache is None else cache
    return cache(q, qd, qdd), tau0


def friction_basis(qd):
    """(n x 7 x 5) basis values per joint: Coulomb, viscous, Stribeck, quadratic, offset."""
    qd = np.asarray(qd, float)
    coul = np.tanh(qd / V_S)
    return np.stack([coul, qd, coul * np.exp(-(qd / V_STR) ** 2), qd * np.abs(qd), np.ones_like(qd)], axis=-1)


def friction_regressor(qd):
    """(n x 7 x 35) block-diagonal friction regressor: column 5*j + b acts on joint j only."""
    n = qd.shape[0]
    phi = friction_basis(qd)
    Phi = np.zeros((n, NJ, NJ * len(FRIC_NAMES)))
    for j in range(NJ):
        Phi[:, j, 5 * j:5 * (j + 1)] = phi[:, j, :]
    return Phi


def friction_torque(xi, qd):
    """Friction torque (n x 7) of coefficients xi (7 x 5) on velocities qd (n x 7)."""
    return np.einsum("njb,jb->nj", friction_basis(qd), xi)


def column_names():
    names = [f"{bn.replace('fr3_', '')}:{p}" for bn in LINKS for p in PI_NAMES]
    names += [f"j{j + 1}:{f}" for j in range(NJ) for f in FRIC_NAMES]
    names += [f"j{j + 1}:arm" for j in range(NJ)]
    return names


def armature_regressor(qdd):
    """(n x 7 x 7) rotor-inertia regressor: column j is qdd_j on row j (tau_j += armature_j qdd_j)."""
    n = qdd.shape[0]
    R = np.zeros((n, NJ, NJ))
    for j in range(NJ):
        R[:, j, j] = qdd[:, j]
    return R


def build_library(model_nom, q, qd, qdd):
    """Returns tau_nom (n x 7), A (n x 7 x 112): 70 inertial, 35 friction, 7 armature columns."""
    Y, tau0 = inertial_regressor(model_nom, q, qd, qdd)
    Phi = friction_regressor(qd)
    return tau0, np.concatenate([Y, Phi, armature_regressor(qdd)], axis=2)


# ----------------------------------------------------------------------------- excitation
class Fourier:
    """Finite Fourier series per joint (Swevers et al.), q(0) = q_home, zero initial velocity not enforced."""

    def __init__(self, seed, n_h=5, f0=0.1, amp=0.35, vmax=1.6, amax=8.0):
        rng = np.random.default_rng(seed)
        self.w = 2 * np.pi * f0
        self.n_h = n_h
        a = rng.standard_normal((NJ, n_h)) / np.arange(1, n_h + 1)
        b = rng.standard_normal((NJ, n_h)) / np.arange(1, n_h + 1)
        # scale each joint to a fraction of its range and respect velocity / acceleration caps
        t = np.linspace(0, 1 / f0, 2000)
        self.a, self.b = a, b
        q, qd, qdd = self._eval(t)
        lim = np.array([[-2.7437, 2.7437], [-1.7837, 1.7837], [-2.9007, 2.9007], [-3.0421, -0.1518],
                        [-2.8065, 2.8065], [0.5445, 4.5169], [-3.0159, 3.0159]])
        half = 0.5 * (lim[:, 1] - lim[:, 0])
        s = np.ones(NJ)
        for j in range(NJ):
            dev = np.abs(q[:, j] - Q_HOME[j]).max()
            s[j] = min(1.0, amp * half[j] / dev, vmax / np.abs(qd[:, j]).max(), amax / np.abs(qdd[:, j]).max())
        self.a, self.b = a * s[:, None], b * s[:, None]
        q, _, _ = self._eval(t)
        assert (q > lim[:, 0] + 0.05).all() and (q < lim[:, 1] - 0.05).all()

    def _eval(self, t):
        t = np.asarray(t, float)
        k = np.arange(1, self.n_h + 1)
        wk = self.w * k
        s, c = np.sin(np.outer(t, wk)), np.cos(np.outer(t, wk))
        q = (Q_HOME + (self.b / wk).sum(axis=1))[:, None] + (self.a / wk) @ s.T - (self.b / wk) @ c.T
        qd = self.a @ c.T + self.b @ s.T
        qdd = -(self.a * wk) @ s.T + (self.b * wk) @ c.T
        return q.T, qd.T, qdd.T

    def __call__(self, t):
        return self._eval(t)


DEFAULT_KP = np.array([300, 300, 300, 300, 120, 80, 40.])
DEFAULT_KD = np.array([30, 30, 30, 30, 12, 8, 4.])
# cross-engine stage: the Gazebo reference has no rotor inertia, so joint 7's inertia is ~1e-4 kg m^2 and a 1 kHz
# explicit PD needs Kd_7 <~ 0.1 to stay stable (Kd dt / I < 1).  Same gains on every plant of that stage.
KD_CROSS = np.array([30, 30, 30, 30, 12, 8, 0.05])


class PD:
    """Nominal-feedforward PD in joint space.  Uses the nominal simulator's inverse dynamics as feedforward."""

    def __init__(self, model_nom, kp=None, kd=None):
        self.model_nom = model_nom
        self.data_nom = mujoco.MjData(model_nom)
        self.kp = np.asarray(DEFAULT_KP if kp is None else kp, float)
        self.kd = np.asarray(DEFAULT_KD if kd is None else kd, float)

    def __call__(self, q, qd, q_d, qd_d, qdd_d):
        tau_ff = inverse_batch(self.model_nom, q_d[None], qd_d[None], qdd_d[None], self.data_nom)[0]
        tau = tau_ff + self.kp * (q_d - q) + self.kd * (qd_d - qd)
        return np.clip(tau, -TAU_LIM, TAU_LIM)


# ----------------------------------------------------------------------------- simulation
class Plant:
    """A MuJoCo model driven by qfrc_applied, with optional extra generalized force from a residual model
    (friction terms not native to MuJoCo, or a learned residual).  extra(q, qd, qdd_prev) -> tau_extra (7,)."""

    def __init__(self, model, extra=None):
        self.model = model
        self.data = mujoco.MjData(model)
        self.extra = extra
        self.nsub = None

    def _extra(self, q, qd, qacc_prev, tau):
        if self.extra is None:
            return 0.0
        if getattr(self.extra, "needs_tau", False):
            return self.extra(q, qd, tau)
        return self.extra(q, qd, qacc_prev)

    def reset(self, q0, qd0=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q0
        self.data.qvel[:] = 0.0 if qd0 is None else qd0
        mujoco.mj_forward(self.model, self.data)

    def simulate(self, traj, controller, T, dt_ctrl=0.001, log_every=1, q0=None):
        """Track traj(t) with controller for T seconds.  Torque held for dt_ctrl.  Returns a log dict."""
        m, d = self.model, self.data
        nsub = int(round(dt_ctrl / m.opt.timestep))
        nstep = int(round(T / dt_ctrl))
        q_d0, _, _ = traj(np.array([0.0]))
        self.reset(q_d0[0] if q0 is None else q0)
        t_log, q_log, qd_log, qdd_log, tau_log, qd_ref = [], [], [], [], [], []
        qacc_prev = np.zeros(NJ)
        for k in range(nstep):
            t = k * dt_ctrl
            q_d, qd_d, qdd_d = traj(np.array([t]))
            q, qd = d.qpos.copy(), d.qvel.copy()
            tau = controller(q, qd, q_d[0], qd_d[0], qdd_d[0])
            if k % log_every == 0:
                ex = self._extra(d.qpos, d.qvel, qacc_prev, tau)
                d.qfrc_applied[:] = tau + ex
                mujoco.mj_forward(m, d)              # instantaneous acceleration under the current torque
                t_log.append(t); q_log.append(q); qd_log.append(qd); tau_log.append(tau); qd_ref.append(q_d[0])
                qdd_log.append(d.qacc.copy())
            for _ in range(nsub):
                ex = self._extra(d.qpos, d.qvel, qacc_prev, tau)
                d.qfrc_applied[:] = tau + ex
                mujoco.mj_step(m, d)
                qacc_prev = d.qacc.copy()
        return dict(t=np.array(t_log), q=np.array(q_log), qd=np.array(qd_log), qdd=np.array(qdd_log),
                    tau=np.array(tau_log), q_ref=np.array(qd_ref))

    def replay_torque(self, q0, qd0, tau_seq, dt_ctrl=0.001):
        """Open-loop torque replay from (q0, qd0).  Returns q trajectory (n x 7)."""
        m, d = self.model, self.data
        nsub = int(round(dt_ctrl / m.opt.timestep))
        self.reset(q0, qd0)
        out = np.empty_like(tau_seq)
        qacc_prev = np.zeros(NJ)
        for k in range(tau_seq.shape[0]):
            out[k] = d.qpos
            for _ in range(nsub):
                ex = self._extra(d.qpos, d.qvel, qacc_prev, tau_seq[k])
                d.qfrc_applied[:] = tau_seq[k] + ex
                mujoco.mj_step(m, d)
                qacc_prev = d.qacc.copy()
        return out


def make_real(gap=GAP_TRUE, timestep=0.0005, integrator="implicit", **kw):
    return apply_gap(load_model(timestep=timestep, integrator=integrator, **kw), gap)


def make_nominal(timestep=0.001, integrator="implicitfast", **kw):
    return load_model(timestep=timestep, integrator=integrator, **kw)


# ----------------------------------------------------------------------------- sensors
def add_sensor_noise(log, rng, scale=1.0, q_quant=1e-4, q_sig=1e-5, tau_sig=0.05):
    """Encoder quantisation + Gaussian jitter on q, Gaussian noise on torque.  scale multiplies all noise."""
    q = log["q"] + scale * q_sig * rng.standard_normal(log["q"].shape)
    if scale > 0:
        q = np.round(q / (scale * q_quant)) * (scale * q_quant)
    tau = log["tau"] + scale * tau_sig * rng.standard_normal(log["tau"].shape)
    return q, tau
