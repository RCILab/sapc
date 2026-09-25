"""Inside the container: run the press for 2.6 s and print what the contact topic delivers."""
import sys, time, numpy as np
sys.path.insert(0, "/work/gazebo")
import fr3_bridge as fb
from gz.msgs10.contacts_pb2 import Contacts
seen = {"n": 0, "sample": None}
def on_contact(msg):
    seen["n"] += 1
    if msg.contact and seen["sample"] is None:
        c = msg.contact[0]
        seen["sample"] = dict(n_contact=len(msg.contact), n_wrench=len(c.wrench), n_pos=len(c.position), n_normal=len(c.normal), n_depth=len(c.depth),
                              c1=c.collision1.name, c2=c.collision2.name,
                              wrench=str(c.wrench[0])[:300] if c.wrench else None, depth=list(c.depth)[:3])
T = np.load("/work/gazebo/traj/press.npz")
g = fb.FR3Gazebo("/work/gazebo/fr3_ref_plate.sdf", "/work/results/gz/debug_contact.gzlog")
g.node.subscribe(Contacts, "/world/fr3w/model/fr3/link/payload/sensor/probe_contact/contact", on_contact)
print("topics with 'contact':", [t for t in g.node.topic_list() if "contact" in t], flush=True)
q_home, q_d, qd_d, tau_ff, kp, kd, lim = T["q_home"], T["q_d"], T["qd_d"], T["tau_ff"], T["kp"], T["kd"], T["tau_lim"]
q, qd = g.q + q_home, g.qd
for k in range(2600):
    tau = np.clip(tau_ff[k] + kp * (q_d[k] - q) + kd * (qd_d[k] - qd), -lim, lim)
    qg, qd = g.step(tau, 1); q = qg + q_home
print("messages received:", seen["n"], "sample:", seen["sample"], flush=True)
g.close()
