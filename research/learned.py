"""Learned residual baselines (claude_try2).

MLPResidual : forward-dynamics residual  qdd = qdd_T0(q, qd, tau) + f(q, qd, tau).  Trained on the same samples as the
              physics library (targets qdd - qdd_T0), applied as a generalized force M_T0(q) f in the twin so that the
              simulation has no algebraic loop.  Torque metrics are reported through tau = tau_T0(q, qd, qdd) - M f.
"""
import mujoco
import numpy as np
import torch
import torch.nn as nn

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class MLPResidual:
    def __init__(self, model_nom, hidden=64, layers=2, steps=4000, lr=2e-3, wd=1e-4, seed=0, batch=2048):
        self.hidden, self.layers, self.steps, self.lr, self.wd, self.seed, self.batch = hidden, layers, steps, lr, wd, seed, batch
        self.model_nom = model_nom
        self.data_nom = mujoco.MjData(model_nom)

    def _feat(self, q, qd, tau):
        return np.concatenate([np.sin(q), np.cos(q), qd, tau], axis=1)

    def qdd_nom(self, q, qd, tau):
        m, d = self.model_nom, self.data_nom
        out = np.empty_like(q)
        for i in range(q.shape[0]):
            d.qpos[:] = q[i]; d.qvel[:] = qd[i]; d.qfrc_applied[:] = tau[i]
            mujoco.mj_forward(m, d); out[i] = d.qacc
        d.qfrc_applied[:] = 0.0
        return out

    def mass_mult(self, q, f):
        m, d = self.model_nom, self.data_nom
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        out = np.zeros(m.nv); mujoco.mj_mulM(m, d, out, np.asarray(f, float))
        return out

    def fit(self, data):
        torch.manual_seed(self.seed)
        X = np.concatenate([self._feat(d["q"], d["qd"], d["tau"]) for d in data])
        Y = np.concatenate([d["qdd"] - self.qdd_nom(d["q"], d["qd"], d["tau"]) for d in data])
        self.mx, self.sx = X.mean(0), X.std(0) + 1e-8; self.my, self.sy = Y.mean(0), Y.std(0) + 1e-8
        self.ylo, self.yhi = Y.min(0), Y.max(0)                     # outputs are clipped to the training range
        Xt = torch.tensor((X - self.mx) / self.sx, dtype=torch.float32, device=DEV)
        Yt = torch.tensor((Y - self.my) / self.sy, dtype=torch.float32, device=DEV)
        dims = [X.shape[1]] + [self.hidden] * self.layers + [7]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.SiLU())
        self.net = nn.Sequential(*mods).to(DEV)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.steps)
        n = Xt.shape[0]
        for s in range(self.steps):
            idx = torch.randint(0, n, (min(self.batch, n),), device=DEV)
            loss = ((self.net(Xt[idx]) - Yt[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        return self

    def predict(self, q, qd, tau):
        X = (self._feat(np.atleast_2d(q), np.atleast_2d(qd), np.atleast_2d(tau)) - self.mx) / self.sx
        with torch.no_grad():
            y = self.net(torch.tensor(X, dtype=torch.float32, device=DEV)).cpu().numpy()
        return np.clip(y * self.sy + self.my, self.ylo, self.yhi)

    def extra_fn(self):
        """Generalized-force hook for fr3_twin.Plant: M_T0(q) f(q, qd, tau) with the commanded torque."""
        def fn(q, qd, tau):
            return self.mass_mult(q, self.predict(q, qd, tau)[0])
        fn.needs_tau = True
        return fn

    def torque_metrics(self, data):
        """Residual torque left after the learned acceleration correction: r - (-M f) at the measured torque."""
        r = np.concatenate([d["r"] for d in data])
        pred = np.concatenate([np.stack([-self.mass_mult(d["q"][i], self.predict(d["q"][i], d["qd"][i], d["tau"][i])[0])
                                         for i in range(d["q"].shape[0])]) for d in data])
        res = r - pred
        return dict(rms_joint=np.sqrt((res ** 2).mean(0)).tolist(), rms_all=float(np.sqrt((res ** 2).mean())),
                    rms0_all=float(np.sqrt((r ** 2).mean())), nrmse=float(np.sqrt((res ** 2).mean()) / np.sqrt((r ** 2).mean())),
                    r2=float(1 - (res ** 2).sum() / ((r - r.mean(0)) ** 2).sum()))
