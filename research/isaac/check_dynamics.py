"""Independent cross-engine mass-matrix/armature/gravity check.

Run on host: python isaac/check_dynamics.py
Uses PhysX runtime tensors, not just USD attributes, to verify the conversion.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def host():
    sys.path.insert(0, str(ROOT))
    import fr3_twin as ft
    import mujoco
    config = json.loads((ROOT / "isaac/inputs/manifest.json").read_text(encoding="utf-8"))
    checks = {}
    for variant in ("nom", "ref"):
        m = ft.make_nominal() if variant == "nom" else ft.make_real()
        m.dof_frictionloss[:] = 0
        m.dof_damping[:] = 0
        d = mujoco.MjData(m)
        rows = []
        for check in config["nominal" if variant == "nom" else "reference"]["fk_checks"]:
            d.qpos[:] = check["q"]
            mujoco.mj_forward(m, d)
            mass = np.zeros((7, 7))
            mujoco.mj_fullM(m, d, mass)
            rows.append(dict(q=check["q"], mass=mass.tolist(), gravity=d.qfrc_bias.tolist(), armature=m.dof_armature.tolist()))
        checks[variant] = rows
    out = ROOT / "results/isaac_dynamics_check"
    out.mkdir(parents=True, exist_ok=True)
    (out / "checked.json").unlink(missing_ok=True)
    (out / "expected.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    cmd = ["docker", "run", "--rm", "--gpus", "all", "-e", "ACCEPT_EULA=Y", "-e", "PRIVACY_CONSENT=N",
           "-v", str(ROOT) + ":/work", "-w", "/work", "--entrypoint", "/isaac-sim/python.sh",
           "nvcr.io/nvidia/isaac-sim:4.5.0", "-u", "/work/isaac/check_dynamics.py", "--inside"]
    with (out / "docker.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    print((out / "checked.json").read_text(encoding="utf-8"))


def inside():
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "multi_gpu": False, "fast_shutdown": True}, experience=str(Path(__file__).with_name("physics.kit")))
    try:
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.stage import create_new_stage
        from collect import build_stage
        config = json.loads((ROOT / "isaac/inputs/manifest.json").read_text())
        out = ROOT / "results/isaac_dynamics_check"
        checks = json.loads((out / "expected.json").read_text())
        result = {}
        for variant, rows in checks.items():
            World.clear_instance()
            create_new_stage()
            world = World(physics_dt=.0005, rendering_dt=.001, stage_units_in_meters=1., device="cpu")
            world.get_physics_context().enable_gpu_dynamics(False)
            world.get_physics_context().set_gravity(-9.81)
            build_stage(world.stage, config, variant, False)
            robot = world.scene.add(Articulation(prim_paths_expr="/World/FR3", name="robot"))
            world.reset()
            order = [robot.get_dof_index(f"fr3_joint{i}") for i in range(1, 8)]
            errors = []
            for row in rows:
                robot.set_joint_positions(np.array([row["q"]]), joint_indices=np.array(order))
                robot.set_joint_velocities(np.zeros((1, 7)))
                mass = robot.get_mass_matrices()[0][np.ix_(order, order)]
                if robot.get_mass_matrices().shape != (1, 7, 7):
                    raise RuntimeError("Not a fixed-base 7-DoF articulation")
                arm = robot.get_armatures()[0][order]
                gravity = robot.get_generalized_gravity_forces()[0][order]
                # PhysX reports rigid-body generalized inertia separately from rotor inertia.
                np.testing.assert_allclose(arm, row["armature"], atol=1e-7, rtol=1e-6)
                mass = mass + np.diag(arm)
                np.testing.assert_allclose(mass, row["mass"], atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(gravity, row["gravity"], atol=2e-4, rtol=2e-5)
                effort = np.array([.3, -.2, .4, -.3, .1, -.1, .05])
                robot.set_joint_efforts(np.array([gravity + effort]), joint_indices=np.array(order))
                world.step(render=False)
                acceleration = robot.get_joint_velocities()[0][order] / .0005
                expected_acceleration = np.linalg.solve(np.array(row["mass"]), effort)
                np.testing.assert_allclose(acceleration, expected_acceleration, atol=.01, rtol=.02)
                errors.append(dict(mass_max_abs=float(np.max(np.abs(mass-row["mass"]))),
                                   gravity_max_abs=float(np.max(np.abs(gravity-row["gravity"]))),
                                   armature_max_abs=float(np.max(np.abs(arm-row["armature"]))),
                                   acceleration_max_abs=float(np.max(np.abs(acceleration-expected_acceleration)))))
            result[variant] = errors
            world.stop()
        (out / "checked.json").write_text(json.dumps(result, indent=2))
        print("PASS: runtime PhysX mass, armature and gravity agree with MuJoCo", flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inside", action="store_true")
    a = p.parse_args()
    inside() if a.inside else host()
