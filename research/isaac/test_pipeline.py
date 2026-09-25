"""Contract and physical conversion checks: python -m unittest discover -s isaac -p test_*.py."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaac.prepare import export_model
import fr3_twin as ft
from run_isaac import read_log, event_metrics


class PhysicalExportTests(unittest.TestCase):
    def test_payload_merge_preserves_parallel_axis_inertia(self):
        for reference in (False, True):
            m = ft.make_real() if reference else ft.make_nominal()
            exported = export_model(reference)["bodies"][-1]
            ids = [m.body("fr3_link7").id, m.body("payload").id]
            masses = m.body_mass[ids]
            centers = [m.body_ipos[ids[0]], m.body_pos[ids[1]] + m.body_ipos[ids[1]]]
            center = np.average(centers, axis=0, weights=masses)
            inertia = np.zeros((3, 3))
            for b, mass, c in zip(ids, masses, centers):
                r = ft.quat2mat(m.body_iquat[b])
                dc = c-center
                inertia += r @ np.diag(m.body_inertia[b]) @ r.T + mass*(np.dot(dc, dc)*np.eye(3)-np.outer(dc, dc))
            er = ft.quat2mat(exported["inertia_quat"])
            np.testing.assert_allclose(exported["mass"], masses.sum(), atol=1e-12)
            np.testing.assert_allclose(exported["com"], center, atol=1e-12)
            np.testing.assert_allclose(er @ np.diag(exported["inertia"]) @ er.T, inertia, atol=1e-12)

    def test_joint_frames_reproduce_mujoco_fk(self):
        m = ft.make_nominal()
        exported = export_model(False)
        d = mujoco.MjData(m)
        for q in ft.Fourier(12)(np.array([0., .7, 3.1, 8.4]))[0]:
            pos, rot = np.zeros(3), np.eye(3)
            for j, angle in zip(exported["joints"], q):
                pos += rot @ j["position"]
                c, s = np.cos(angle), np.sin(angle)
                rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                rot = rot @ ft.quat2mat(j["orientation"]) @ rz
            tcp = pos + rot @ np.array([0., 0., .107])
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            np.testing.assert_allclose(tcp, d.site_xpos[m.site("tcp").id], atol=1e-12)

    def test_armature_preserved_and_four_group_gap(self):
        nom, ref = export_model(False), export_model(True)
        for a, b in zip(nom["joints"], ref["joints"]):
            self.assertEqual(a["armature"], b["armature"])
        changed_links = [a["name"] for a, b in zip(nom["bodies"], ref["bodies"]) if not np.isclose(a["mass"], b["mass"])]
        changed_joints = [a["name"] for a, b in zip(nom["joints"], ref["joints"]) if (a["coulomb"], a["viscous"]) != (b["coulomb"], b["viscous"])]
        self.assertEqual(changed_links, ["fr3_link3", "fr3_link7"])
        self.assertEqual(changed_joints, ["fr3_joint4", "fr3_joint6"])


class LogContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "test.npz"
        t = np.arange(1000)*.001
        q, qd, _ = ft.Fourier(11)(t)
        meta = dict(schema_version=2, fixed_base=True, engine="Isaac Sim 4.5.0 / PhysX", manifest_sha256="test",
                    dof_names=ft.JOINTS, fk_error_m=[1e-7], task="seed11")
        self.data = dict(t=t, q=q, qd=qd, tau=np.zeros_like(q), q_ref=q, F=np.zeros(len(t)), p=np.zeros((len(t), 3)),
                         kp=ft.DEFAULT_KP, kd=ft.KD_CROSS, tau_lim=ft.TAU_LIM, metadata=json.dumps(meta))

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_log_is_accepted(self):
        np.savez(self.path, **self.data)
        result = read_log(self.path, seed=11, expected_hash="test")
        self.assertEqual(result["q_meas"].shape, (1000, 7))

    def test_shifted_timestamp_rejected(self):
        self.data["t"] += .001
        np.savez(self.path, **self.data)
        with self.assertRaisesRegex(ValueError, "pre-step"):
            read_log(self.path, seed=11)

    def test_mixed_configuration_rejected(self):
        np.savez(self.path, **self.data)
        with self.assertRaisesRegex(ValueError, "Mixed experiment"):
            read_log(self.path, expected_hash="different")

    def test_nonfinite_and_wrong_seed_rejected(self):
        self.data["q"][4, 2] = np.nan
        np.savez(self.path, **self.data)
        with self.assertRaisesRegex(ValueError, "Invalid q"):
            read_log(self.path)
        self.data["q"][4, 2] = 0
        np.savez(self.path, **self.data)
        with self.assertRaisesRegex(ValueError, "Wrong trajectory seed"):
            read_log(self.path, seed=12)

    def test_no_contact_is_not_reported_as_time_zero(self):
        result = event_metrics(self.data["t"], self.data["F"])
        self.assertIsNone(result["onset"])
        self.assertEqual(result["impulse"], 0)

    def test_nan_tcp_cannot_pass_fk_audit(self):
        self.data["p"][0, 0] = np.nan
        np.savez(self.path, **self.data)
        with self.assertRaisesRegex(ValueError, "Invalid PhysX TCP"):
            read_log(self.path)


if __name__ == "__main__":
    unittest.main()
