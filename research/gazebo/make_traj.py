"""Precompute controller references for the Gazebo bridge (Windows side, needs MuJoCo).

For each Fourier seed (and the press task): q_d, qd_d, qdd_d at 1 kHz and the nominal-model feedforward torque
tau_ff = ID_T0(q_d, qd_d, qdd_d) (MuJoCo nominal, with armature), plus PD gains.  Same controller as fr3_twin.PD.
Usage: python gazebo/make_traj.py     -> gazebo/traj/seed{1,2,3,11,12}.npz, gazebo/traj/press.npz
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import fr3_twin as ft
import contact_stage as cs

OUT = os.path.join(HERE, "traj")
os.makedirs(OUT, exist_ok=True)
nom = ft.make_nominal()
ft.DEFAULT_KD = ft.KD_CROSS                      # cross-engine gains (see fr3_twin.KD_CROSS)
pd = ft.PD(nom)


def write(name, traj, T):
    t = np.arange(int(round(T / 0.001))) * 0.001
    q_d, qd_d, qdd_d = traj(t)
    tau_ff = ft.inverse_batch(nom, q_d, qd_d, qdd_d)
    np.savez(os.path.join(OUT, name), t=t, q_d=q_d, qd_d=qd_d, qdd_d=qdd_d, tau_ff=tau_ff, kp=pd.kp, kd=pd.kd,
             q_home=ft.Q_HOME, tau_lim=ft.TAU_LIM)
    print("wrote", name, q_d.shape)


for sd in (1, 2, 3, 11, 12):
    write(f"seed{sd}.npz", ft.Fourier(sd), 20.0)
write("press.npz", cs.Press(), 4.0)
write("hold.npz", lambda t: (np.tile(ft.Q_HOME, (len(t), 1)), np.zeros((len(t), 7)), np.zeros((len(t), 7))), 2.0)
