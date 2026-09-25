"""Generate Gazebo (SDF 1.9, DART) FR3 worlds from the MuJoCo Menagerie model (claude_try2, cross-engine stage E6).

- Link poses are baked at the MuJoCo 'home' configuration, so Gazebo joint angle 0 == q_home (q_mj = q_gz + q_home).
- Inertials copied from the MJCF (mass, CoM, principal inertia + orientation).  No visuals, no link collisions.
- Joint dynamics: viscous damping and Coulomb friction from the MJCF.  SDF has no rotor inertia (armature) and sdformat
  rejects the usual workaround (I_zz += armature violates the triangle inequality on slender links), so the Gazebo
  reference has NO rotor inertia, as real-world Gazebo FR3 models do.  This is a genuine engine gap the identification
  must discover (armature group).  PD gains are lowered (fr3_twin.PD_KD_X) so the 1 kHz discrete PD stays stable on the
  light wrist joints (effective inertia ~1e-3 kg m^2 without armature).
- Physics gap (reference world): payload link fixed to link 7 at the flange, joint-4/6 friction/damping, link-3 mass.
- Optional plate + spherical probe with a contact sensor for the press task.
Usage:  python gazebo/make_fr3_sdf.py            -> gazebo/fr3_nom.sdf, fr3_ref.sdf, fr3_nom_plate.sdf, fr3_ref_plate.sdf
"""
import os, sys
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fr3_twin as ft

WORLD = "fr3w"


def rpy(quat_wxyz):
    q = np.asarray(quat_wxyz, float)
    return Rot.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")


def pose_str(pos, quat_wxyz):
    r = rpy(quat_wxyz)
    return " ".join(f"{v:.9g}" for v in list(pos) + list(r))


def quat_mul(a, b):
    out = np.zeros(4); mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float)); return out


def inertial_xml(mass, ipos, iquat, inertia, armature=0.0):
    """SDF inertial block.  With armature > 0 the rotor inertia is folded into the link's inertia about the joint
    axis (link-frame z), the usual workaround for engines without rotor inertia; the full matrix is written in the
    link frame (identity inertial orientation)."""
    if armature <= 0:
        return (f"<inertial><pose>{pose_str(ipos, iquat)}</pose><mass>{mass:.9g}</mass>"
                f"<inertia><ixx>{inertia[0]:.9g}</ixx><ixy>0</ixy><ixz>0</ixz><iyy>{inertia[1]:.9g}</iyy><iyz>0</iyz><izz>{inertia[2]:.9g}</izz></inertia></inertial>")
    R = ft.quat2mat(iquat)
    I = R @ np.diag(np.asarray(inertia, float)) @ R.T + np.diag([0.0, 0.0, armature])
    return (f"<inertial><pose>{pose_str(ipos, (1, 0, 0, 0))}</pose><mass>{mass:.9g}</mass>"
            f"<inertia><ixx>{I[0, 0]:.9g}</ixx><ixy>{I[0, 1]:.9g}</ixy><ixz>{I[0, 2]:.9g}</ixz><iyy>{I[1, 1]:.9g}</iyy><iyz>{I[1, 2]:.9g}</iyz><izz>{I[2, 2]:.9g}</izz></inertia></inertial>")


def build(model, gap=None, plate=None, probe_r=0.01, armature_in_link=False):
    """model: MuJoCo nominal model (read geometry/inertials); gap: dict like fr3_twin.GAP_TRUE or None."""
    gap = gap or {}
    m = model
    names = ["fr3_link0"] + ft.LINKS
    ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names]
    mass_scale = gap.get("link_mass_scale", {})
    links, joints = [], []
    for k, (n, b) in enumerate(zip(names, ids)):
        mass, ipos, iquat, inertia = m.body_mass[b], m.body_ipos[b], m.body_iquat[b], m.body_inertia[b].copy()
        if n in mass_scale:
            mass = mass * mass_scale[n]; inertia = inertia * mass_scale[n]
        if k == 0:
            pose = "<pose>0 0 0 0 0 0</pose>"
        else:
            qz = np.array([np.cos(ft.Q_HOME[k - 1] / 2), 0, 0, np.sin(ft.Q_HOME[k - 1] / 2)])
            pose = f'<pose relative_to="{names[k - 1]}">{pose_str(m.body_pos[b], quat_mul(m.body_quat[b], qz))}</pose>'
        arm = 0.0
        if k > 0 and armature_in_link:
            arm = m.dof_armature[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, ft.JOINTS[k - 1])]]
        links.append(f'<link name="{n}">{pose}{inertial_xml(mass, ipos, iquat, inertia, arm)}</link>')
        if k > 0:
            jn = ft.JOINTS[k - 1]
            dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]
            damping, friction = m.dof_damping[dof], m.dof_frictionloss[dof]
            mods = gap.get("joints", {}).get(jn, {})
            damping = mods.get("damping", damping); friction = mods.get("frictionloss", friction)
            lo, hi = m.jnt_range[k - 1] - ft.Q_HOME[k - 1]
            joints.append(f'<joint name="{jn}" type="revolute"><parent>{names[k - 1]}</parent><child>{n}</child>'
                          f'<axis><xyz>0 0 1</xyz><limit><lower>{lo:.6g}</lower><upper>{hi:.6g}</upper><effort>{ft.TAU_LIM[k - 1]:g}</effort><velocity>5</velocity></limit>'
                          f'<dynamics><damping>{damping:.9g}</damping><friction>{friction:.9g}</friction></dynamics></axis></joint>')
    # payload / probe link fixed to link 7 at the flange
    p = gap.get("payload")
    pmass = p["mass"] if p else 1e-6
    pipos = np.asarray(p["com"], float) if p else np.zeros(3)
    pinertia = np.asarray(p["inertia"], float) if p else np.full(3, 1e-9)
    probe = ""
    if plate is not None:
        probe = (f'<collision name="probe_collision"><pose>0 0 {probe_r} 0 0 0</pose><geometry><sphere><radius>{probe_r}</radius></sphere></geometry>'
                 f'<surface><friction><ode><mu>{plate["mu"]}</mu><mu2>{plate["mu"]}</mu2></ode></friction>'
                 f'<contact><ode><kp>{plate["kp"]:g}</kp><kd>{plate["kd"]:g}</kd></ode></contact></surface></collision>'
                 f'<sensor name="probe_contact" type="contact"><contact><collision>probe_collision</collision></contact>'
                 f'<update_rate>1000</update_rate><topic>/probe_contact</topic><always_on>1</always_on></sensor>')
    links.append(f'<link name="payload"><pose relative_to="fr3_link7">0 0 0.107 0 0 0</pose>{inertial_xml(pmass, pipos, np.array([1, 0, 0, 0.]), pinertia)}{probe}</link>')
    ft_sensor = ('<sensor name="payload_ft" type="force_torque"><update_rate>1000</update_rate><always_on>1</always_on>'
                 '<force_torque><frame>child</frame><measure_direction>parent_to_child</measure_direction></force_torque></sensor>') if plate is not None else ""
    joints.append(f'<joint name="payload_fixed" type="fixed"><parent>fr3_link7</parent><child>payload</child>{ft_sensor}</joint>')
    plugins = ('<plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher"/>'
               + "".join(f'<plugin filename="gz-sim-apply-joint-force-system" name="gz::sim::systems::ApplyJointForce"><joint_name>{jn}</joint_name></plugin>' for jn in ft.JOINTS))
    model_xml = (f'<model name="fr3"><pose>0 0 0 0 0 0</pose><self_collide>false</self_collide>'
                 f'<joint name="world_fixed" type="fixed"><parent>world</parent><child>fr3_link0</child></joint>'
                 + "".join(links) + "".join(joints) + plugins + "</model>")
    plate_xml = ""
    if plate is not None:
        px, py, pz, half = plate["pos"][0], plate["pos"][1], plate["pos"][2] - plate["half"], plate["half"]
        plate_xml = (f'<model name="plate"><static>true</static><pose>{px:.6g} {py:.6g} {pz:.6g} 0 0 0</pose><link name="plate_link">'
                     f'<collision name="plate_collision"><geometry><box><size>0.3 0.3 {2 * half:.6g}</size></box></geometry>'
                     f'<surface><friction><ode><mu>{plate["mu"]}</mu><mu2>{plate["mu"]}</mu2></ode></friction>'
                     f'<contact><ode><kp>{plate["kp"]:g}</kp><kd>{plate["kd"]:g}</kd></ode></contact></surface></collision></link></model>')
    world = (f'<?xml version="1.0"?><sdf version="1.9"><world name="{WORLD}">'
             f'<physics name="1ms" type="dart"><max_step_size>0.001</max_step_size><real_time_factor>0</real_time_factor></physics>'
             f'<gravity>0 0 -9.81</gravity>'
             f'<plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"><engine><filename>gz-physics-dartsim-plugin</filename></engine></plugin>'
             f'<plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>'
             f'<plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>'
             f'<plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>'
             f'<plugin filename="gz-sim-forcetorque-system" name="gz::sim::systems::ForceTorque"/>'
             + model_xml + plate_xml + "</world></sdf>")
    return world


if __name__ == "__main__":
    nom = ft.make_nominal()
    sys.path.insert(0, os.path.dirname(HERE))
    import contact_stage as cs
    plate_geom = dict(pos=cs.PLATE_POS, half=0.01)
    variants = {
        "fr3_nom.sdf": dict(gap=None, plate=None),
        "fr3_ref.sdf": dict(gap=ft.GAP_TRUE, plate=None),
        "fr3_nom_plate.sdf": dict(gap=None, plate=dict(plate_geom, mu=0.6, kp=1e6, kd=100.0)),
        "fr3_ref_plate.sdf": dict(gap=ft.GAP_TRUE, plate=dict(plate_geom, mu=0.3, kp=3e6, kd=30.0)),
    }
    for fn, kw in variants.items():
        open(os.path.join(HERE, fn), "w", encoding="utf-8").write(build(nom, **kw))
        print("wrote", fn)
