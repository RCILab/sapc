"""Export the existing FR3 parameters and controller inputs; run with host Python.

No Isaac assets, URDF importer, network assets, or changes to existing experiments.
The rigid payload is merged into link 7 in standard inertial coordinates.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fr3_twin as ft


def export_model(reference):
    m = ft.make_real() if reference else ft.make_nominal()
    d = mujoco.MjData(m)
    d.qpos[:] = 0
    mujoco.mj_forward(m, d)
    bodies = []
    for name in ["fr3_link0"] + ft.LINKS:
        b = m.body(name).id
        pi = ft.body_pi(m, name)
        if name == "fr3_link7":
            p = m.body("payload").id
            pi += ft.pi_from_params(m.body_mass[p], m.body_pos[p] + m.body_ipos[p],
                                    m.body_inertia[p], m.body_iquat[p])
        if name == "fr3_link0":
            # Fixed to world: this otherwise massless base does not move.
            mass, com, inertia, quat = 1.0, np.zeros(3), np.ones(3) * 0.01, np.array([1., 0, 0, 0])
        else:
            mass, com, inertia, quat, valid = ft.params_from_pi(pi)
            if not valid:
                raise ValueError(f"Invalid inertia: {name}")
        bodies.append(dict(name=name, mass=float(mass), com=com.tolist(), inertia=inertia.tolist(),
                           inertia_quat=quat.tolist(), position=d.xpos[b].tolist(),
                           orientation=d.xquat[b].tolist()))
    joints = []
    for i, name in enumerate(ft.JOINTS):
        j = m.joint(name).id
        b, dof = m.jnt_bodyid[j], m.jnt_dofadr[j]
        joints.append(dict(name=name, parent=f"fr3_link{i}", child=f"fr3_link{i+1}",
                           position=m.body_pos[b].tolist(), orientation=m.body_quat[b].tolist(),
                           limits=m.jnt_range[j].tolist(), armature=float(m.dof_armature[dof]),
                           coulomb=float(m.dof_frictionloss[dof]), viscous=float(m.dof_damping[dof])))
    # Independent check points for the USD articulation's kinematics.
    checks = []
    for q in (ft.Q_HOME, ft.Fourier(11)(np.array([1.7]))[0][0]):
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        checks.append(dict(q=q.tolist(), tcp=d.site_xpos[m.site("tcp").id].tolist()))
    return dict(bodies=bodies, joints=joints, fk_checks=checks)


def prepare(out, duration=20.0):
    if duration < 1.0:
        raise ValueError("Use at least one second of motion")
    out.mkdir(parents=True, exist_ok=True)
    nom = ft.make_nominal()
    dt = 0.001
    # Compute press geometry without importing contact_stage (it opens an old log).
    d = mujoco.MjData(nom)
    d.qpos[:] = ft.Q_HOME
    mujoco.mj_forward(nom, d)
    b = nom.body("payload").id
    probe_local = np.array([0., 0., 0.01])
    probe = d.xpos[b] + d.xmat[b].reshape(3, 3) @ probe_local
    jac = np.zeros((3, nom.nv))
    mujoco.mj_jac(nom, d, jac, None, probe, b)
    plate = probe.copy()
    plate[2] -= 0.01 + 0.026
    manifest = dict(schema_version=2, fixed_base=True, engine="Isaac Sim 4.5.0 / PhysX", dt_control=dt,
                    dt_physics=0.0005, solver="TGS", position_iterations=16, velocity_iterations=4,
                    armature_mode="native_preserved", friction_mode="explicit_tanh_viscous",
                    friction_velocity=ft.V_S, native_joint_friction=0.0,
                    friction_note="Passive SI torque recomputed each physics substep; not PhysX load-dependent jointFriction.",
                    q_home=ft.Q_HOME.tolist(), tau_lim=ft.TAU_LIM.tolist(),
                    kp=ft.DEFAULT_KP.tolist(), kd=ft.KD_CROSS.tolist(),
                    tcp_local=[0, 0, 0.107], probe_local=[0, 0, 0.117], probe_radius=0.01,
                    plate_top=plate.tolist(), plate_half=[0.15, 0.15, 0.01],
                    contact_offset=0.0001, rest_offset=0.0,
                    nominal=export_model(False), reference=export_model(True),
                    gap=ft.GAP_TRUE, trajectories={})

    def write(name, q, qd, qdd):
        t = np.arange(len(q)) * dt
        filename = out / f"{name}.npz"
        np.savez(filename, t=t, q_d=q, qd_d=qd, qdd_d=qdd,
                 tau_ff=ft.inverse_batch(nom, q, qd, qdd), kp=ft.DEFAULT_KP,
                 kd=ft.KD_CROSS, tau_lim=ft.TAU_LIM, q_home=ft.Q_HOME)
        manifest["trajectories"][name] = dict(file=filename.name, samples=len(t),
                                            sha256=hashlib.sha256(filename.read_bytes()).hexdigest())

    t = np.arange(round(duration / dt)) * dt
    for seed in (1, 2, 3, 11, 12):
        write(f"seed{seed}", *ft.Fourier(seed)(t))
    ht = np.arange(2000) * dt
    write("hold", np.tile(ft.Q_HOME, (len(ht), 1)), np.zeros((len(ht), 7)), np.zeros((len(ht), 7)))
    t = np.arange(4000) * dt
    s = np.maximum(t - 1.0, 0)
    z, zd, zdd = np.zeros((3, len(t)))
    a = 0.029
    down, hold, up = s < 1, (s >= 1) & (s < 2), (s >= 2) & (s < 3)
    z[down] = a / 2 * (1 - np.cos(np.pi * s[down]))
    zd[down] = a / 2 * np.pi * np.sin(np.pi * s[down])
    zdd[down] = a / 2 * np.pi**2 * np.cos(np.pi * s[down])
    z[hold] = a
    z[up] = a / 2 * (1 + np.cos(np.pi * (s[up] - 2)))
    zd[up] = -a / 2 * np.pi * np.sin(np.pi * (s[up] - 2))
    zdd[up] = -a / 2 * np.pi**2 * np.cos(np.pi * (s[up] - 2))
    direction = -np.linalg.pinv(jac)[:, 2]
    write("press", ft.Q_HOME + z[:, None] * direction, zd[:, None] * direction, zdd[:, None] * direction)
    manifest["source_sha256"] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("fr3_twin.py", "models/franka_fr3/fr3.xml")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {out}: five {duration:g}s trajectories, hold, press, nominal/reference parameters")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "isaac" / "inputs")
    p.add_argument("--duration", type=float, default=20.0)
    a = p.parse_args()
    prepare(a.out.resolve(), a.duration)
