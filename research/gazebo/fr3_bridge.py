"""Gazebo Harmonic bridge for the FR3 worlds (runs INSIDE the jas-gazebo:harmonic container).

Deterministic stepping: the world starts paused; each control step publishes seven joint forces and requests
multi_step = n physics steps (1 ms each), then waits for the joint_state message whose stamp reaches the target time.
Controller: tau = tau_ff[k] + Kp (q_d[k] - q) + Kd (qd_d[k] - qd), with q_d, qd_d, tau_ff precomputed (npz from
gazebo/make_traj.py, MuJoCo nominal inverse dynamics).  Joint angles are converted with q_mj = q_gz + q_home.

Usage:  python3 gazebo/fr3_bridge.py <world.sdf> <traj.npz> <out.npz> [--dt 0.001]
Output npz: t, q, qd, tau (commanded), q_ref, F (contact normal force if a contact sensor exists, else zeros).
"""
import sys, os, time, threading, subprocess, uuid, argparse
import numpy as np
from gz.transport13 import Node
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.contacts_pb2 import Contacts
from gz.msgs10.wrench_pb2 import Wrench

WORLD = "fr3w"
JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]


class FR3Gazebo:
    def __init__(self, world_path, log_path):
        os.environ["GZ_PARTITION"] = "dt_" + uuid.uuid4().hex
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(["gz", "sim", "-s", "--headless-rendering", "-v", "2", str(world_path)],
                                     stdout=self.log, stderr=subprocess.STDOUT)
        self.node = Node()
        self.cond = threading.Condition()
        self.t = -1.0; self.q = np.zeros(7); self.qd = np.zeros(7); self.F = 0.0; self.F_t = -1.0
        self.node.subscribe(Model, f"/world/{WORLD}/model/fr3/joint_state", self._on_state)
        self.node.subscribe(Contacts, "/probe_contact", self._on_contact)
        self.node.subscribe(Contacts, f"/world/{WORLD}/model/fr3/link/payload/sensor/probe_contact/contact", self._on_contact)
        self.W = np.zeros(3)
        self.node.subscribe(Wrench, f"/world/{WORLD}/model/fr3/joint/payload_fixed/sensor/payload_ft/forcetorque", self._on_wrench)
        self.pubs = [self.node.advertise(f"/model/fr3/joint/{j}/cmd_force", Double) for j in JOINTS]
        for _ in range(300):
            if f"/world/{WORLD}/control" in self.node.service_list() and all(p.has_connections() for p in self.pubs):
                break
            if self.proc.poll() is not None:
                raise RuntimeError("Gazebo exited; see log")
            time.sleep(0.1)
        else:
            raise RuntimeError("Gazebo discovery timeout: " + str(self.node.service_list()[:10]))
        self.steps = 0
        self.step(np.zeros(7), 1)          # one step to get the first state

    def _on_state(self, msg):
        q = np.zeros(7); qd = np.zeros(7)
        for j in msg.joint:
            if j.name in JOINTS:
                i = JOINTS.index(j.name); q[i] = j.axis1.position; qd[i] = j.axis1.velocity
        with self.cond:
            self.q, self.qd = q, qd
            self.t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
            self.cond.notify_all()

    def _on_contact(self, msg):
        f = 0.0
        for c in msg.contact:
            for w in c.wrench:
                f += abs(w.body_1_wrench.force.z)
        self.F = f
        self.F_t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9

    def _on_wrench(self, msg):
        self.W = np.array([msg.force.x, msg.force.y, msg.force.z])

    def step(self, tau, n=1):
        for p, v in zip(self.pubs, tau):
            m = Double(); m.data = float(v); p.publish(m)
        time.sleep(0.0003)                 # let the force messages land before the physics step
        target = (self.steps + n) * 0.001
        req = WorldControl(); req.pause = True; req.multi_step = n
        # the server performs the steps even when the reply is late; do not block on it (timeout in ms)
        self.node.request(f"/world/{WORLD}/control", req, WorldControl, Boolean, 2)
        deadline = time.monotonic() + 15
        with self.cond:
            while self.t < target - 1e-7:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    raise RuntimeError(f"physics timeout t={self.t} target={target} proc={self.proc.poll()}")
                self.cond.wait(min(remain, 0.1))
        self.steps += n
        return self.q.copy(), self.qd.copy()

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill(); self.proc.wait()
        self.log.close()


def run(world, traj, out, dt=0.001):
    T = np.load(traj)
    q_home, q_d, qd_d, tau_ff, kp, kd = T["q_home"], T["q_d"], T["qd_d"], T["tau_ff"], T["kp"], T["kd"]
    tau_lim = T["tau_lim"]
    n = q_d.shape[0]; nsub = int(round(dt / 0.001))
    g = FR3Gazebo(world, out + ".gzlog")
    t0 = time.time()
    Q, QD, TAU, F, W = np.zeros((n, 7)), np.zeros((n, 7)), np.zeros((n, 7)), np.zeros(n), np.zeros((n, 3))
    try:
        q, qd = g.q + q_home, g.qd
        for k in range(n):
            tau = np.clip(tau_ff[k] + kp * (q_d[k] - q) + kd * (qd_d[k] - qd), -tau_lim, tau_lim)
            Q[k], QD[k], TAU[k], F[k], W[k] = q, qd, tau, g.F, g.W
            qg, qd = g.step(tau, nsub)
            q = qg + q_home
            if k % 2000 == 0:
                print(f"  step {k}/{n}  sim t {g.t:.3f}  wall {time.time() - t0:.0f} s  |q_d-q|max {np.abs(q_d[k] - q).max():.4f}", flush=True)
    finally:
        g.close()
    np.savez(out, t=np.arange(n) * dt, q=Q, qd=QD, tau=TAU, q_ref=q_d, F=F, W=W)
    print("saved", out, "wall", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("world"); ap.add_argument("traj"); ap.add_argument("out"); ap.add_argument("--dt", type=float, default=0.001)
    a = ap.parse_args()
    run(a.world, a.traj, a.out, a.dt)
